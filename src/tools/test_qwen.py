from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from transformers import Trainer, GPTQConfig, deepspeed

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

from tokenization_qwen import QWenTokenizer

from transformers.generation import GenerationConfig
import torch

torch.manual_seed(1234)

# Note: The default behavior now has injection attack prevention off.
tokenizer = QWenTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant.",
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens
    _separation = tokenizer(":").input_ids
    _img_token = tokenizer.img_end_id

    # Apply prompt templates
    input_ids, targets = [], []
    image_masks = []
    instruction_masks = []
    question_masks = []

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = (
            [im_start]
            + _system
            + tokenizer(system_message).input_ids
            + [im_end]
            + nl_tokens
        )
        input_id += system
        target += (
            [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        )
        assert len(input_id) == len(target)

        image_mask = [0] * len(input_id)
        instruction_mask = [0] * len(input_id)
        question_mask = [0] * len(input_id)

        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = (
                tokenizer(role).input_ids
                + nl_tokens
                + tokenizer(sentence["value"]).input_ids
                + [im_end]
                + nl_tokens
            )
            input_id += _input_id

            first_part = tokenizer(role).input_ids + nl_tokens
            second_part = tokenizer(sentence["value"]).input_ids
            last_part = [im_end] + nl_tokens

            if _img_token in second_part:
                image_part = second_part[: second_part.index(_img_token) + 2]
                instruction_and_question_part = second_part[
                    second_part.index(_img_token) + 2 :
                ]
            else:
                image_part = []
                instruction_and_question_part = second_part

            # split the second part with _separation token
            if _separation[0] in instruction_and_question_part:
                instruction_part = instruction_and_question_part[
                    : instruction_and_question_part.index(_separation[0])
                ]
                question_part = instruction_and_question_part[
                    instruction_and_question_part.index(_separation[0]) + 1 :
                ]
            else:
                instruction_part = []
                question_part = instruction_and_question_part

            # print("first_part", first_part)
            # print(tokenizer.decode(first_part))
            # print("image_part", image_part)
            # print(tokenizer.decode(image_part))
            # print("instruction_part", instruction_part)
            # print(tokenizer.decode(instruction_part))
            # print("question_part", question_part)
            # print(tokenizer.decode(question_part))
            # print("last_part", last_part)
            # print(tokenizer.decode(last_part))
            # input()

            image_mask += (
                [0] * len(first_part)
                + [1] * len(image_part)
                + [0] * len(instruction_part)
                + [0] * len(question_part)
                + [0] * len(last_part)
            )
            instruction_mask += (
                [0] * len(first_part)
                + [0] * len(image_part)
                + [1] * len(instruction_part)
                + [0] * len(question_part)
                + [0] * len(last_part)
            )
            question_mask += (
                [0] * len(first_part)
                + [0] * len(image_part)
                + [0] * len(instruction_part)
                + [1] * len(question_part)
                + [0] * len(last_part)
            )

            if role == "<|im_start|>user":
                _target = (
                    [im_start]
                    + [IGNORE_TOKEN_ID] * (len(_input_id) - 3)
                    + [im_end]
                    + nl_tokens
                )
            elif role == "<|im_start|>assistant":
                _target = (
                    [im_start]
                    + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids)
                    + _input_id[len(tokenizer(role).input_ids) + 1 : -2]
                    + [im_end]
                    + nl_tokens
                )
            else:
                raise NotImplementedError
            target += _target

        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
        image_masks.append(image_mask[:max_len])
        instruction_masks.append(instruction_mask[:max_len])
        question_masks.append(question_mask[:max_len])

    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.long)
    image_masks = torch.tensor(image_masks, dtype=torch.int)
    instruction_masks = torch.tensor(instruction_masks, dtype=torch.int)
    question_masks = torch.tensor(question_masks, dtype=torch.int)

    return dict(
        input_ids=input_ids.cuda(),
        labels=targets.cuda(),
        attention_mask=input_ids.ne(tokenizer.pad_token_id).cuda(),
        # image_mask=image_masks,
        # instruction_mask=instruction_masks,
        # question_mask=question_masks
    )


sources = [
    [
        {"from": "user", "value": "你好"},
        {"from": "assistant", "value": "我是Qwen-VL,一个支持视觉输入的大模型。"},
    ],
    # [
    #   {
    #     "from": "user",
    #     "value": "<img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>\nretrieve a document that is connected to this image: what is the breed of the dog in the picture?"
    #   },
    #   {
    #     "from": "user",
    #     "value": "<img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>\nretrieve a document that is connected to this image: what is the breed of the dog in the picture?"
    #   },
    # ]
    # [
    #   {
    #     "from": "user",
    #     "value": "Picture 1: <img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>\n图中的狗是什么品种？"
    #   },
    #   {
    #     "from": "assistant",
    #     "value": "图中是一只拉布拉多犬。"
    #   },
    #   {
    #     "from": "user",
    #     "value": "框出图中的格子衬衫"
    #   },
    #   {
    #     "from": "assistant",
    #     "value": "<ref>格子衬衫</ref><box>(588,499),(725,789)</box>"
    #   }
    # ]
]


tokenizer.pad_token_id = tokenizer.eod_id

encoding = preprocess(sources, tokenizer, 512)
print(encoding["input_ids"].shape)
input_ids = encoding["input_ids"]
# print(tokenizer.decode(input_ids[0]))
# print(input_ids)
# for id in input_ids[0]:
#     print(tokenizer.convert_tokens_to_ids([id]))
# print(encoding['labels'].shape)
# print(encoding['attention_mask'].shape)
# exit()

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-VL-Chat-Int4",
    device_map="cuda",
    torch_dtype=torch.float32,
    trust_remote_code=True,
    fp16=True,
)  # .eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
# model = AutoModelForCausalLM.from_pretrained(
#     "Qwen/Qwen-VL-Chat",
#     device_map="auto",
#     trust_remote_code=True,
#     fp16=True,
#     quantization_config=GPTQConfig(
#         bits=4, disable_exllama=True
#     ),
# )

for name, param in model.named_parameters():
    param.requires_grad = False


forward_results = model(**encoding)
print(forward_results.loss)
exit()
# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained(
    "Qwen/Qwen-VL-Chat", trust_remote_code=True
)

# 1st dialogue turn
query = tokenizer.from_list_format(
    [
        {
            "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        },  # Either a local path or an url
        {"text": "根据例子重写问题用于检索。图中的场景是什么？"},
    ]
)
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
# 图中是一名女子在沙滩上和狗玩耍，旁边是一只拉布拉多犬，它们处于沙滩上。

# 2nd dialogue turn
# response, history = model.chat(tokenizer, '框出图中击掌的位置', history=history)
# print(response)
# # <ref>击掌</ref><box>(536,509),(588,602)</box>
# image = tokenizer.draw_bbox_on_latest_picture(response, history)
# if image:
#   image.save('1.jpg')
# else:
#   print("no box")
