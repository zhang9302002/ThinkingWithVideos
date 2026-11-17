# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import math
import logging
import os
import re
import random
from tqdm import tqdm
from collections import defaultdict
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

from verl.tools.base_tool import initialize_tools_from_config
from functools import partial

from verl.utils.dataset.vision_utils import process_image, process_video, cached_process_video

logger = logging.getLogger(__name__)

import eval_prompts 

def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, *dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class RLHFDatasetMultiTurn(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False

        # Initialize reward list
        self.reward_list = config.multi_turn.get("reward_list", ["iou"])
        logger.info(f"Initialized reward_list: {self.reward_list} {type(self.reward_list)}")
        self.video_kwargs = dict(config.multi_turn.get("video_kwargs", {}))
        logger.info(f"Initialized video_kwargs: {self.video_kwargs}")

        # Initialize tools from config file
        self.max_turns = config.multi_turn.get("max_turns", 4)
        tool_config_path = config.multi_turn.get("tool_config_path", None)
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        logger.info(f"Initialized tools: {self.tools}")
        logger.info(f"Initialized tool_schemas: {self.tool_schemas}")
        # Initialize video base
        video_bases = config.multi_turn.get("video_base", None)
        if not isinstance(video_bases, (List, ListConfig)):
            video_bases = [video_bases]
        self.video_bases = video_bases
        self.system_prompt = self._build_system_prompt(self.tool_schemas)
        self.system_prompt_notool = self._build_system_prompt([])
        self.user_prompt_template = config.get("user_prompt_template", "TEMPORAL_GROUNDING_TEMPLATE")
        self.user_prompt_template = getattr(eval_prompts, self.user_prompt_template)
        logger.info(f"Initialized video bases: {self.video_bases}")
        logger.info(f"Initialized system prompt: {self.system_prompt}")
        logger.info(f"Initialized prompt template: {self.user_prompt_template}")
        print(f'In rl_dataset, self.user_prompt_template={self.user_prompt_template}')

        self._download()
        self._read_files_and_tokenize()

    def _build_system_prompt(self, tool_schemas):
        sys_prompt = "You are a helpful assistant."
        tool_schemas = [json.dumps(x) for x in tool_schemas]
        tool_schemas = '\n'.join(tool_schemas)
        tool_call_prompt = f"""
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_schemas}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
"""
        if len(tool_schemas) > 0:
            sys_prompt += "\n" + tool_call_prompt
        return str(sys_prompt)


    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _make_message(self, example, idx, video_base):
        data = {}
        max_frames = 64
        if 'video' in example:
            duration = example['duration']
            duration = round(duration, 2)
            prompt = self.user_prompt_template.format(input_text=example['text'], duration=duration)
            video_path = os.path.join(video_base, example['video'])
            video_content = {
                "type": "video",
                "video": video_path,
                "min_pixels": 4*28*28,
                "max_pixels": 224*224,
                "max_frames": max_frames,
            }
            if isinstance(self.video_kwargs, dict):
                video_content.update(self.video_kwargs)
            if 'video_kwargs' in example and isinstance(example['video_kwargs'], dict):
                video_content.update(example['video_kwargs'])
            data["videos"] = [video_content]
            user_content = [
                video_content,
                {"type": "text", "text": prompt}
            ]
        elif 'image' in example:
            img_template = self.user_prompt_template.replace("This is a video with duration {duration} seconds.", "This is an image.")
            prompt = img_template.format(input_text=example['text'])
            image_path = os.path.join(video_base, example['image'])
            image_content = {
                "type": "image",
                "image": image_path,
                "min_pixels": 4*28*28,
                "max_pixels": 448*448,
            }
            data["images"] = [image_content]
            video_path = None
            duration = 0.0
            user_content = [
                image_content,
                {"type": "text", "text": prompt}
            ]
        data.update({
            "data_source": example['data_source'],
            "prompt": [ 
                {
                    "role": "system",
                    "content": self.system_prompt if 'video' in example else self.system_prompt_notool,
                }, {
                    "role": "user",
                    "content": user_content,
                } 
            ],
            "ability": "temporal grounding",
            "reward_model": {"style": "rule", "ground_truth": example['solution']},
            "extra_info": {
                "split": 'train',
                "index": idx,
                "answer": example['solution'],
                "question": example['text'],
                "tools_kwargs": {
                    "video_path": video_path,
                    "duration": duration,
                    **self.video_kwargs,
                },
                "reward_list": self.reward_list,
            },
        })
        return data

    def _read_files_and_tokenize(self):
        dataframe_list = []
        idx = 0
        for json_file, video_base in zip(self.data_files, self.video_bases):
            dataframe = json.load(open(json_file))
            for item in dataframe:
                idx += 1
                new_item = self._make_message(
                    item, 
                    idx, 
                    video_base=str(video_base), 
                )
                dataframe_list.append(new_item)
        print(f"origin dataset len: {len(dataframe_list)}")

        dataframe_images = []
        dataframe_videos = []
        if self.filter_overlong_prompts:
            def filter_fn(doc):
                gt = doc["reward_model"]["ground_truth"]
                if isinstance(gt, list) and len(gt) == 2:
                    st, ed = gt
                    if st >= ed:
                        print(f'filter item out, {st=}, {ed=}')
                        return False
                return True
            for item in tqdm(dataframe_list, desc=f"Filter seq longer than {self.max_prompt_length} tokens"):
                if filter_fn(item):
                    if 'images' in item:
                        dataframe_images.append(item)
                    elif 'videos' in item:
                        dataframe_videos.append(item)
        print(f"filtered dataset: {len(dataframe_images)=}, {len(dataframe_videos)=}")

        # interleave them in one line
        if len(dataframe_images) > 0 and len(dataframe_videos) > 0:
            random.shuffle(dataframe_images)
            random.shuffle(dataframe_videos)
            # image : video == 1 : 1
            self.dataframe = [x for item in zip(dataframe_images, dataframe_videos) for x in item]
            # image : video == 1 : 3    
            # min_len = min(len(dataframe_images), len(dataframe_videos) // 3)
            # self.dataframe = []
            # for i in range(min_len):
            #     self.dataframe.append(dataframe_images[i])
            #     self.dataframe.extend(dataframe_videos[i*3:(i+1)*3])
            print(f"interleaved dataset len: {len(self.dataframe)} {self.dataframe[:3]}")
        else:
            self.dataframe = dataframe_images + dataframe_videos
            print(f"single modality dataset len: {len(self.dataframe)}")

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = row_dict.pop(self.prompt_key)
        model_inputs = {}

        if self.processor is not None:

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}
            mm_processor_kwargs = {}

            images = None
            if self.image_key in row_dict:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]
                multi_modal_data["image"] = images
                mm_processor_kwargs["fps"] = []

            videos = None
            if self.video_key in row_dict:
                videos = []
                multi_modal_data["video"] = []
                mm_processor_kwargs["fps"] = []
                for video in row_dict.pop(self.video_key):
                    # try to load from frame
                    video_path = video['video']
                    video_frame_path = video_path.split('.')[0]
                    fps = 2.0
                    if os.path.exists(video_frame_path):
                        frame_paths = os.listdir(video_frame_path)
                        frame_paths = sorted(frame_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                        frame_paths = [os.path.join(video_frame_path, frame_path) for frame_path in frame_paths]
                        total_frames = len(frame_paths)
                        # max_frames = video.get('max_frames', None)
                        # if max_frames is not None and total_frames > max_frames:
                        #     idx = torch.linspace(0, total_frames - 1, max_frames).round().long()
                        #     frame_paths = [frame_paths[i] for i in idx]
                        #     fps = max_frames / max(total_frames, 1e-6) * fps
                        video_path = frame_paths
                        video['video'] = video_path
                        video['fps'] = fps
                    # new_video, fps = cached_process_video(video)
                    new_video, fps = process_video(video)
                    videos.append(new_video)
                    multi_modal_data["video"].append(new_video)
                    mm_processor_kwargs["fps"].append(fps)

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, fps=mm_processor_kwargs["fps"], return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs["second_per_grid_ts"] = torch.tensor(model_inputs["second_per_grid_ts"])

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["mm_processor_kwargs"] = mm_processor_kwargs

            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            # row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and 'Qwen' in self.processor.image_processor.__class__.__name__:  # little trick, support qwen2.5-vl
            from verl.models.transformers.qwen2_vl import get_rope_index
            logger.debug(f'>>> in rl_dataset: start use get_rope_index, video_grid_thw={model_inputs.get("video_grid_thw")}, second_per_grid_ts={model_inputs.get("second_per_grid_ts")}')
            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
