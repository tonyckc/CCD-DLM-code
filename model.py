
import os
import warnings
from typing import Optional, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers.utils import (
    is_torchdynamo_compiling,
)
from generation_utils import DreamGenerationConfig,DreamModelOutput

# Import the actual model class
from modeling_dream import DreamModel  # Adjust import based on actual file
import random
def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False,marginal_entropy=False):


    if temperature > 0:
        logits = logits / temperature

    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)


    probs = torch.softmax(logits, dim=-1)



    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10

        log_probs = torch.log(probs + epsilon)


        confidence = torch.sum(probs * log_probs, dim=-1)



    return confidence, x0



class ExtendedDreamModel(DreamModel):
    @torch.no_grad()
    def diffusion_generate_inference(
            self,
            inputs: Optional[torch.Tensor] = None,
            inputs_refine=None,
            return_query_embeddings=False,
            target_embeds=None,
            step_alloc_type="uniform",
            decoding_type="dream",
            seek_mode=False,
            expand_ratio=1.0,
            generation_config: Optional[DreamGenerationConfig] = None,
            **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
                hasattr(generation_config, "pad_token_id") and
                torch.any(input_ids == generation_config.pad_token_id) and
                attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        if decoding_type == "dream":
            result = self._sample_inference(
                input_ids,
                inputs_refine,
                attention_mask=attention_mask,
                generation_config=generation_config,
                generation_tokens_hook_func=generation_tokens_hook_func,
                generation_logits_hook_func=generation_logits_hook_func,
                return_query_embeddings=return_query_embeddings,
                step_alloc_type=step_alloc_type,
                target_embeds=target_embeds,
            )
        elif decoding_type == "ours":
            result = self._sample_inference_lookahead(
                input_ids,
                inputs_refine,
                attention_mask=attention_mask,
                generation_config=generation_config,
                generation_tokens_hook_func=generation_tokens_hook_func,
                generation_logits_hook_func=generation_logits_hook_func,
                return_query_embeddings=return_query_embeddings,
                step_alloc_type=step_alloc_type,
                seek_mode=seek_mode,
                expand_ratio = expand_ratio,
                target_embeds=target_embeds,
            )
        return result


    def _sample_inference(
            self,
            input_ids: torch.LongTensor,
            inputs_refine: torch.LongTensor,
            attention_mask: Optional[torch.LongTensor],
            generation_config: DreamGenerationConfig,
            generation_tokens_hook_func,
            generation_logits_hook_func,
            return_query_embeddings,
            target_embeds,
            step_alloc_type,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        histories = [] if (return_dict_in_generate and output_history) else None
        histories_hidden_states = [] if (return_dict_in_generate and output_history) else None

        eos_token_id = generation_config.eos_token_id

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        generate_tokens = x[0]
        #token_id_list = {id: tokenizer.decode([token]) for id, token in enumerate(generate_tokens)}

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"


        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(None, x, None)

        query_length = input_ids.shape[1]
        #print("query_length:", query_length)


        steps_plot = []
        mask_index_all = (x == mask_token_id)

        refine_buff = 0
        monitor_buffer = []
        tolerance_length = 3
        initial_unmask_token_length = 16
        minimal_unmask_token_length = 1
        decay_rate = 0.9
        unmask_token_length = initial_unmask_token_length
        effective_token = []
        for i in range(steps):

            mask_index = (x == mask_token_id)
            if step_alloc_type == 'uniform':
                t = timesteps[i]
                s = timesteps[i + 1]
                num_mask_token = mask_index.sum() / mask_index.shape[0]
                number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
            elif step_alloc_type == 'dynamic':
                t = timesteps[i]
                s = timesteps[i + 1]
                number_transfer_tokens = initial_unmask_token_length if i == 0 else unmask_token_length #token_schedule[i+3] #
            else:
                print('mismatch type')


            output = self(x, attention_mask, tok_idx)
            logits = output.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            # new output
            last_hidden_states = output.hidden_states

            # this allows user-defined logits control of the intermediate steps
            logits = generation_logits_hook_func(i, x, logits)



            mask_logits = logits[mask_index]




            if alg == 'origin':
                p_transfer = 1 - s / t if i < steps - 1 else 1
                x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                _, x0[transfer_index_t_s] = sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature,
                                                          top_p=top_p, top_k=top_k)
                x[mask_index] = x0.clone()
            else:
                if alg == 'maskgit_plus':

                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                elif alg == 'topk_margin':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k,
                                                   margin_confidence=True)
                elif alg == 'entropy':
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")


                steps_plot.append(number_transfer_tokens)

                full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                full_confidence[mask_index] = confidence
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)

                    else:
                        full_confidence = full_confidence / alg_temp
                        full_confidence = F.softmax(full_confidence, dim=-1)
                        transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)

                    x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()

                    row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)


                    x[row_indices, transfer_index] = x_[row_indices, transfer_index]

            # this allows user-defined token control of the intermediate steps
            x = generation_tokens_hook_func(i, x, logits)

            if histories is not None:
                histories.append(x.clone())

        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
                history_last_hidden_state=histories_hidden_states,
            )
        else:
            return x

    def _sample_inference_lookahead(
            self,
            input_ids: torch.LongTensor,
            inputs_refine: torch.LongTensor,
            attention_mask: Optional[torch.LongTensor],
            generation_config: DreamGenerationConfig,
            generation_tokens_hook_func,
            generation_logits_hook_func,
            return_query_embeddings,
            target_embeds,
            step_alloc_type,
            seek_mode,
            expand_ratio,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        generation_config.steps = steps
        generation_config.temperature = temperature
        generation_config.top_p = top_p
        generation_config.top_k = top_k
        generation_config.alg = alg
        generation_config.alg_temp = alg_temp

        histories = [] if (return_dict_in_generate and output_history) else None
        histories_hidden_states = [] if (return_dict_in_generate and output_history) else None

        eos_token_id = generation_config.eos_token_id

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)


        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(None, x, None)

        query_length = input_ids.shape[1]
        if self.device == torch.device("cuda:5"):
            print("query_length:", query_length)

        steps_plot = []
        mask_index_all = (x == mask_token_id)

        initial_unmask_token_length = 16
        minimal_unmask_token_length = 1
        decay_rate = 0.9
        unmask_token_length = initial_unmask_token_length
        mask_token_length = (max_length - input_ids.shape[1])


        verification_buffer = {"token_index": [], "token_id": [], "prediction": []}
        draft_buffer = {"token_index": [], "token_id": [], "prediction": []}
        uniform_onestep_number_transfer_tokens = int((max_length - input_ids.shape[1]) / steps)
        expand_ratio = expand_ratio
        verification_buffer_size_maximum = uniform_onestep_number_transfer_tokens + int(
            uniform_onestep_number_transfer_tokens * expand_ratio)
        verification_buffer_size_initial = 1

        stable_rank_mode_open = False

        seek_mode = seek_mode
        actual_update = uniform_onestep_number_transfer_tokens if seek_mode == False else None
        history_maximum_token_id_list = []

        warm_up_steps = 5

        sparse_count = 0


        actual_steps = []
        effective_token = []
        for i in range(steps):


            mask_index = (x == mask_token_id)

            num_mask_token = mask_index.sum() / mask_index.shape[0]




            if num_mask_token == 0:
                break

            # refine_buff = mask_index
            if step_alloc_type == 'uniform':
                t = timesteps[i]
                s = timesteps[i + 1]
                num_mask_token = mask_index.sum() / mask_index.shape[0]

                number_transfer_tokens = uniform_onestep_number_transfer_tokens
                # late stage
                if num_mask_token <= 0.1 * mask_token_length:
                    verification_buffer_size = verification_buffer_size_initial
                    draft_buffer_size = 2 * verification_buffer_size
                    n_top_verification = verification_buffer_size
                    n_top_draft = draft_buffer_size

                if i < (warm_up_steps - 1):
                    verification_buffer_size = verification_buffer_size_initial
                    draft_buffer_size = 2 * verification_buffer_size
                    n_top_verification = verification_buffer_size
                    n_top_draft = draft_buffer_size
                else:
                    verification_buffer_size = min(verification_buffer_size + 1,
                                                   verification_buffer_size_maximum) if i != 0 else verification_buffer_size_initial
                    draft_buffer_size = 2 * verification_buffer_size
                    n_top_verification = verification_buffer_size
                    n_top_draft = draft_buffer_size




            elif step_alloc_type == 'dynamic':
                t = timesteps[i]
                s = timesteps[i + 1]
                number_transfer_tokens = initial_unmask_token_length if i == 0 else unmask_token_length  # token_schedule[i+3] #

            else:
                print('mismatch type')



            output = self(x, attention_mask, tok_idx)
            logits = output.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            # new output
            last_hidden_states = output.hidden_states

            # this allows user-defined logits control of the intermediate steps
            logits = generation_logits_hook_func(i, x, logits)
            mask_logits = logits[mask_index]

            if i >= (warm_up_steps - 1):
                # TODO: if we need to use the samping value here.
                _, current_maximum__token_id = torch.softmax(logits / temperature, dim=-1).max(dim=-1) if temperature > 0 else torch.softmax(logits, dim=-1).max(dim=-1)
                history_maximum_token_id_list.append(current_maximum__token_id)

            if alg == 'origin':
                p_transfer = 1 - s / t if i < steps - 1 else 1
                x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                _, x0[transfer_index_t_s] = sample_tokens(mask_logits[transfer_index_t_s],
                                                          temperature=temperature,
                                                          top_p=top_p, top_k=top_k)
                x[mask_index] = x0.clone()
            else:
                if alg == 'maskgit_plus':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p,
                                                   top_k=top_k)
                elif alg == 'topk_margin':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p,
                                                   top_k=top_k,
                                                   margin_confidence=True)
                elif alg == 'entropy':
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k,
                                                   neg_entropy=True)
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")

                # check if the confidence is not reliable
                mask_indices = torch.argwhere(confidence == 0)  # shape n*k
                # many tokens have same confidence (0)
                stable_rank_mode_open = False  # True if mask_indices.shape[0] >= 0.1 * mask_token_length else False


                full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                full_confidence[mask_index] = confidence

                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:

                        if stable_rank_mode_open:
                            # transfer_index shape number_transfer_tokens*2 (2 is shape of full_confidence)
                            transfer_index = torch.argwhere(full_confidence == 0)[:number_transfer_tokens]
                            # TODO: batch size > 1 and number_transfer_tokens > 1
                            transfer_index = transfer_index[:, 1:]
                        else:
                            # transfer_index shape 1xtopk
                            _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)


                    else:
                        full_confidence = full_confidence / alg_temp
                        full_confidence = F.softmax(full_confidence, dim=-1)
                        transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)

                    x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()

                    row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(
                        transfer_index)

                    ##################
                    ##################
                    if i >= warm_up_steps and i < steps - 1:
                        # check top number_transfer_tokens candidate in or not in verification buffer
                        current_top_transfer_index = torch.squeeze(transfer_index)
                        # print("query length:",query_length)

                        number_transfer_tokens_current_lookahead = n_top_draft

                        if stable_rank_mode_open:
                            transfer_index_lookahead = torch.argwhere(full_confidence == 0)[
                                                       :number_transfer_tokens_current_lookahead]
                            transfer_index_lookahead = transfer_index_lookahead[:, 1:].permute(1, 0)
                        else:
                            _, transfer_index_lookahead = torch.topk(full_confidence,
                                                                     number_transfer_tokens_current_lookahead)

                        current_verification_token_index = transfer_index_lookahead[0][:n_top_verification]



                        # if at least one token in verification buffer
                        has_intersection = torch.isin(current_top_transfer_index,
                                                      verification_buffer["token_index"][-1]).any()

                        if has_intersection:  # last step, TODO: more history steps and one-step budgets (now support 1 token in one step)

                            number_transfer_tokens_current_lookahead = n_top_draft

                            if stable_rank_mode_open:
                                transfer_index_lookahead = torch.argwhere(full_confidence == 0)[
                                                           :number_transfer_tokens_current_lookahead]
                                transfer_index_lookahead = transfer_index_lookahead[:, 1:].permute(1, 0)
                            else:
                                _, transfer_index_lookahead = torch.topk(full_confidence,
                                                                         number_transfer_tokens_current_lookahead)

                            x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                            x_[mask_index] = x0.clone()

                            current_verification_token_index = transfer_index_lookahead[0][:n_top_verification]


                            verification_buffer_token_index = verification_buffer["token_index"][-1]

                            # where current_verification_token_index is in verification_buffer_token_index
                            mask = torch.isin(current_verification_token_index, verification_buffer_token_index)
                            # get the token indices according to mask
                            matching_tokens_index = current_verification_token_index[mask].unsqueeze(0)
                            # get the token  values
                            row_indices_matching_tokens_id_current = torch.arange(x.size(0),
                                                                                  device=self.device).unsqueeze(
                                1).expand_as(
                                matching_tokens_index)

                            matching_tokens_id_current = current_maximum__token_id[
                                row_indices_matching_tokens_id_current, matching_tokens_index]

                            # according to the current matching token index to retrieve history token index
                            # matching_tokens_id_history = draft_buffer["prediction"][-1][
                            #    row_indices_matching_tokens_id_current, matching_tokens_index]
                            matching_tokens_id_history = history_maximum_token_id_list[-2][
                                row_indices_matching_tokens_id_current, matching_tokens_index]

                            # like double check
                            mask_index_id_position = (matching_tokens_id_current == matching_tokens_id_history)

                            # find where match id, then get the position and id double match
                            matching_tokens_index_and_id = matching_tokens_index[mask_index_id_position].unsqueeze(0)

                            row_indices_matching_tokens_index_and_id = torch.arange(x.size(0),
                                                                                    device=self.device).unsqueeze(
                                1).expand_as(
                                matching_tokens_index_and_id)
                            # unmask
                            matching_pass_num = matching_tokens_index_and_id.shape[1]

                            actual_steps.append(matching_pass_num)

                            # no match, decode as original
                            if matching_pass_num < uniform_onestep_number_transfer_tokens:
                                # print("no match")
                                steps_plot.append(number_transfer_tokens)

                                # print("update #num:", number_transfer_tokens)

                                # no process
                                x[row_indices, transfer_index] = x_[row_indices, transfer_index]
                                # update verification and draft buffer

                                # check how many mask tokens has
                                mask_index_check = (x == mask_token_id)
                                num_mask_token_check = mask_index_check.sum() / mask_index_check.shape[0]

                                if num_mask_token_check == 0:
                                    # print("no mask token early stop")
                                    break

                                number_transfer_tokens_lookahead = n_top_draft + number_transfer_tokens  # plus a current top token

                                # to process edge tiem, avoid consider more tokens when mask tokens are less than number_transfer_tokens_lookahead
                                if number_transfer_tokens_lookahead > num_mask_token:
                                    # update n top draft and n top verification
                                    new_number_transfer_tokens_current_lookahead = int(num_mask_token.item())
                                    # update n top draft
                                    n_top_verification = n_top_verification if num_mask_token_check >= n_top_verification else int(
                                        num_mask_token_check.item())
                                    n_top_draft = int(num_mask_token_check.item())


                                if stable_rank_mode_open:
                                    transfer_index = torch.argwhere(full_confidence == 0)[
                                                     :number_transfer_tokens_lookahead]
                                    transfer_index = transfer_index[:, 1:].permute(1, 0)
                                else:
                                    _, transfer_index = torch.topk(full_confidence, number_transfer_tokens_lookahead)

                                x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                                x_[mask_index] = x0.clone()

                                row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(
                                    transfer_index)
                                #
                                draft_tokens = torch.squeeze(x_[row_indices, transfer_index])[
                                               number_transfer_tokens:]  # remove current decode token
                                token_index = torch.squeeze(transfer_index)[number_transfer_tokens:]
                                draft_buffer['token_index'].append(token_index)
                                draft_buffer['token_id'].append(draft_tokens)
                                draft_buffer["prediction"].append(x_)

                                verification_tokens = torch.squeeze(x_[row_indices, transfer_index])[
                                                      number_transfer_tokens:number_transfer_tokens + n_top_verification]  # first 1:1+n_top_verification as verification
                                token_index = torch.squeeze(transfer_index)[
                                              number_transfer_tokens:number_transfer_tokens + n_top_verification]
                                verification_buffer['token_index'].append(token_index)
                                verification_buffer['token_id'].append(verification_tokens)
                                continue
                            else:

                                if seek_mode:
                                    # update
                                    x[row_indices_matching_tokens_index_and_id, matching_tokens_index_and_id] = x_[
                                        row_indices_matching_tokens_index_and_id, matching_tokens_index_and_id]



                                else:  # equal to uniform
                                    row_indices_matching_tokens_index_and_id = row_indices_matching_tokens_index_and_id[
                                                                               :, :actual_update]
                                    matching_tokens_index_and_id = matching_tokens_index_and_id[:, :actual_update]
                                    x[row_indices_matching_tokens_index_and_id, matching_tokens_index_and_id] = x_[
                                        row_indices_matching_tokens_index_and_id, matching_tokens_index_and_id]


                            steps_plot.append(matching_pass_num)

                            # update verification and draft buffer
                            # check how many mask tokens has
                            mask_index_check = (x == mask_token_id)
                            num_mask_token_check = mask_index_check.sum() / mask_index_check.shape[0]

                            if num_mask_token_check == 0:

                                break

                            new_number_transfer_tokens_current_lookahead = n_top_draft + matching_pass_num
                            # avoid consider more tokens when mask tokens are less than number_transfer_tokens_lookahead

                            if new_number_transfer_tokens_current_lookahead > num_mask_token:
                                # update n top draft and n top verification
                                new_number_transfer_tokens_current_lookahead = int(num_mask_token.item())
                                # update n top draft
                                n_top_verification = n_top_verification if num_mask_token_check >= n_top_verification else int(
                                    num_mask_token_check.item())
                                n_top_draft = int(num_mask_token_check.item())



                            if stable_rank_mode_open:
                                transfer_index_new = torch.argwhere(full_confidence == 0)[
                                                     :new_number_transfer_tokens_current_lookahead]
                                transfer_index_new = transfer_index_new[:, 1:].permute(1, 0)
                            else:
                                _, transfer_index_new = torch.topk(full_confidence,
                                                                   new_number_transfer_tokens_current_lookahead)

                            # remove current decode token, and keep top n_top_draft always
                            mask = ~torch.isin(transfer_index_new.flatten(),
                                               matching_tokens_index_and_id.flatten()).unsqueeze(0)
                            draft_token_index = transfer_index_new[mask]

                            draft_buffer['token_index'].append(torch.squeeze(draft_token_index))
                            draft_buffer["prediction"].append(x_)

                            verification_buffer["token_index"].append(draft_token_index[:n_top_verification])

                        else:
                            # print("NOT found in verification buffer in {}".format(verification_buffer_size))
                            # no process
                            x[row_indices, transfer_index] = x_[row_indices, transfer_index]

                            # check how many mask tokens has
                            mask_index_check = (x == mask_token_id)
                            num_mask_token_check = mask_index_check.sum() / mask_index_check.shape[0]

                            steps_plot.append(number_transfer_tokens)
                            # update verification and draft buffer
                            if num_mask_token_check == 0:
                                # print("no mask token early stop")
                                break

                            number_transfer_tokens_lookahead = n_top_draft + number_transfer_tokens  # plus a current top token
                            # avoid consider more tokens when mask tokens are less than number_transfer_tokens_lookahead

                            if number_transfer_tokens_lookahead > num_mask_token:
                                # update n top draft and n top verification
                                new_number_transfer_tokens_current_lookahead = int(num_mask_token.item())
                                # update n top draft
                                n_top_verification = n_top_verification if num_mask_token_check >= n_top_verification else int(
                                    num_mask_token_check.item())
                                n_top_draft = int(num_mask_token_check.item())


                            if stable_rank_mode_open:
                                transfer_index = torch.argwhere(full_confidence == 0)[
                                                 :number_transfer_tokens_lookahead]
                                transfer_index = transfer_index[:, 1:].permute(1, 0)
                            else:

                                _, transfer_index = torch.topk(full_confidence, number_transfer_tokens_lookahead)

                            x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                            x_[mask_index] = x0.clone()

                            row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(
                                transfer_index)
                            #
                            draft_tokens = torch.squeeze(x_[row_indices, transfer_index])[
                                           number_transfer_tokens:]  # remove current decode token
                            token_index = torch.squeeze(transfer_index)[number_transfer_tokens:]
                            draft_buffer['token_index'].append(token_index)
                            draft_buffer['token_id'].append(draft_tokens)
                            draft_buffer["prediction"].append(x_)

                            verification_tokens = torch.squeeze(x_[row_indices, transfer_index])[
                                                  number_transfer_tokens:number_transfer_tokens + n_top_verification]  # first 1:1+n_top_verification as verification
                            token_index = torch.squeeze(transfer_index)[
                                          number_transfer_tokens:number_transfer_tokens + n_top_verification]
                            verification_buffer['token_index'].append(token_index)
                            verification_buffer['token_id'].append(verification_tokens)

                    else:

                        # print('regular decoding as a warming-up process')
                        x[row_indices, transfer_index] = x_[row_indices, transfer_index]
                        steps_plot.append(number_transfer_tokens)

                        if i == (warm_up_steps - 1) or i == (warm_up_steps - 2):  # prepare buffer

                            number_transfer_tokens_lookahead = n_top_draft + number_transfer_tokens  # plus a current top token

                            if stable_rank_mode_open:
                                transfer_index = torch.argwhere(full_confidence == 0)[
                                                 :number_transfer_tokens_lookahead]
                                transfer_index = transfer_index[:, 1:].permute(1, 0)
                            else:

                                _, transfer_index = torch.topk(full_confidence, number_transfer_tokens_lookahead)

                            x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                            x_[mask_index] = x0.clone()

                            row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(
                                transfer_index)
                            #
                            draft_tokens = torch.squeeze(x_[row_indices, transfer_index])[
                                           number_transfer_tokens:]  # remove current decode token
                            token_index = torch.squeeze(transfer_index)[number_transfer_tokens:]
                            draft_buffer['token_index'].append(token_index)
                            draft_buffer['token_id'].append(draft_tokens)
                            draft_buffer["prediction"].append(x_)

                            verification_tokens = torch.squeeze(x_[row_indices, transfer_index])[
                                                  number_transfer_tokens:number_transfer_tokens + n_top_verification]  # first 1:1+n_top_verification as verification

                            token_index = torch.squeeze(transfer_index)[
                                          number_transfer_tokens:number_transfer_tokens + n_top_verification]

                            verification_buffer['token_index'].append(token_index)

                            verification_buffer['token_id'].append(verification_tokens)

            # this allows user-defined token control of the intermediate steps
            x = generation_tokens_hook_func(i, x, logits)

            if histories is not None:
                histories.append(x.clone())


        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
                history_last_hidden_state=histories_hidden_states,
                finish_step=i,
            )
        else:
            return x

