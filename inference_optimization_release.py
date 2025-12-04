import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from transformers import AutoTokenizer
import json
from trip_metric import trip_metric
from tqdm import trange


from model import ExtendedDreamModel

import torch
import random


# Set random seed for reproducibility
seed = 45
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch_size = 1
city_list = [1] # from 1-8, which denotes the city #3 to #10
step = 256
decoding_type = "ours" # or "dream"

######################
# setting when decoding_type = "ours"
seek_mode =  True # in "ours", seek_mode = False as a uniform model ; seek_mode = True as a dynamic mode
expand_ratio = 4.0 # buffer size, 4 as a default
#######################
print("decoding_type:",decoding_type)

# set the model_path
model_path = "Dream-org/Dream-v0-Instruct-7B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model_cus = ExtendedDreamModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model_cus = model_cus.to("cuda").eval()


hard_acc_total = 0
for city in city_list:
    responses = []

    with open(
            "./planning_split_8/trip_planning_part{}_of_8.json".format(city)) as f:
        data = json.load(f)

    # two shots
    inputs_all = []
    for i in data.values():
        splits = i['prompt_5shot'].split("TASK:")
        inputs_all.append("TASK:".join(splits[:3] + [splits[-1]]))  #
        solution_text = splits[1].split('SOLUTION:')[-1]


    ###############################################################
    messages = [
        {"role": "user", "content": solution_text}
    ]

    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    input_ids = inputs.input_ids.to(device="cuda")
    attention_mask = inputs.attention_mask.to(device="cuda")

    total_step = 0
    for i in trange(0, len(inputs_all), batch_size):
            batch = inputs_all[i:i + batch_size]
            print('-'*20)
            messages = [
                {"role": "user", "content": batch[0]}
            ]
            inputs = tokenizer.apply_chat_template(
                messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
            )
            input_ids = inputs.input_ids.to(device="cuda")
            attention_mask = inputs.attention_mask.to(device="cuda")


            #input_ids = inputs_refine
            output = model_cus.diffusion_generate_inference(
                input_ids,
                None,
                attention_mask=attention_mask,
                steps=step,
                max_new_tokens=256,
                output_history=True,
                temperature=0,
                top_p=1.0,
                alg="entropy",
                return_dict_in_generate=True,
                output_hidden_states=True,
                target_embeds=None,
                step_alloc_type="uniform",
                decoding_type=decoding_type,
                seek_mode=seek_mode,
                expand_ratio=expand_ratio,
            )
            generations = [
                tokenizer.decode(g[len(p) :].tolist())
                for p, g in zip(input_ids, output.sequences)
            ]

            total_step += output.finish_step if decoding_type == 'ours' else 256

            responses.extend(generations)



    hard_acc = trip_metric(data, responses)
    hard_acc_total += hard_acc
    print("hard_acc:",hard_acc)
    print("average step=", total_step / len(inputs_all))
    average_step = total_step / len(inputs_all)
    f = open("./city_instruct_seek_mode_{}_{}_expand_{}_ours_v3.txt".format(seek_mode,decoding_type,expand_ratio), "a")
    f.write(f"expand_ratio:{expand_ratio}, city:{city}, average_step:{average_step},hard_acc:{hard_acc}\n")
    f.close()

print("average acc across {} cities:".format(len(city_list)), hard_acc_total / len(city_list))
