# Based on https://github.com/haotian-liu/LLaVA.

import os
import json
import math
import torch
from torch.utils.data import Dataset, DataLoader
import random
import argparse
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu

# from qwen_vl_utils import process_vision_info
from verl.utils.dataset.video_vl_utils import process_vision_info, cached_process_vision_info

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, GenerationConfig, AutoConfig, AutoModelForVision2Seq
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig
from peft import AutoPeftModelForCausalLM
from collections import defaultdict

import sys
import re
import vllm

from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd_multi_turn_sync import parse_output
from verl.tools.base_tool import initialize_tools_from_config
import uuid

VIDEO_INFO_CACHE = {}
TEMPLATE = os.getenv("MY_PROMPT_TEMPLATE", default="Please find the visual event described by a sentence in the video, determining its starting and ending times. The format should be: 'The event happens in the start time - end time'. For example, The event 'person turn a light on' happens in the 24.30 - 30.42 seconds. Now I will give you the textual sentence: {input_text}. Please return its start time and end time.")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果你使用多个GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def auto_detect_world_size(model_path):
    pattern = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt$")
    world_sizes = set()
    ranks = set()
    for filename in os.listdir(model_path):
        m = pattern.match(filename)
        if m:
            ws = int(m.group(1))
            rank = int(m.group(2))
            world_sizes.add(ws)
            ranks.add(rank)
    if not world_sizes:
        return -100
    # 如果所有文件的world_size一致，直接用
    if len(world_sizes) == 1:
        return list(world_sizes)[0]
    # 否则用rank数量猜测
    return max(ranks) + 1

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    # chunk_size = math.ceil(len(lst) / n)  # integer division
    # return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]
    res = [[] for i in range(n)]
    for i, x in enumerate(lst):
        res[i % n].append(x)
    return res

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def load_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames

def ranki_print(s):
    print(f'[cuda:{args.chunk_idx}] {s}', flush=True)
    return
    if args.chunk_idx == 0:
        print(f'[cuda:{args.chunk_idx}] {s}', flush=True)


class VideoQADataset(Dataset):
    def __init__(self, args, processor):
        with open(args.gt_file) as file:
            gt_questions = json.load(file)
        gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
        if os.path.exists(args.answers_file):
            with open(args.answers_file, "r") as f:
                id_set = [json.loads(row)['id'] for row in f.readlines()]
                id_set = set(id_set)
                gt_questions = [sample for sample in gt_questions if sample['id'] not in id_set]
        self.data = gt_questions
        content_video = {
            "type": "video",
            "video": None,
        }
        content_image = {
            "type": "image",
            "image": None,
        }
        if args.max_frames is not None:
            content_video['max_frames'] = args.max_frames
        if args.max_pixels is not None:
            content_video['max_pixels'] = args.max_pixels
            content_image["max_pixels"] = 448*448
        if args.resized_height is not None:
            content_video['resized_height'] = args.resized_height
            content_image['resized_height'] = args.resized_height
        if args.resized_width is not None:
            content_video['resized_width'] = args.resized_width
            content_image['resized_width'] = args.resized_width
        self.content_video = content_video
        self.content_image = content_image
        self.video_dir = args.video_dir
        self.processor = processor
        tool_config_path = "verl/verl/tools/config/zoom_tool_config_new.yaml"
        tool_list = initialize_tools_from_config(tool_config_path)
        tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tools = {tool.name: tool for tool in tool_list}
        self.system_prompt = build_system_prompt(tool_schemas)
        self.no_number = args.no_number
        print(f'>>> no_number: {self.no_number}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        while idx < len(self.data):
            # try:
                sample = self.data[idx]
                if not 'question' in sample:
                    sample['question'] = sample['text']
                if not 'answer' in sample:
                    sample['answer'] = str(sample['solution'])
                
                if 'video' in sample or 'video_id' in sample:
                    video_name = sample['video_id'] if 'video_id' in sample else sample['video']
                    video_path, fps = None, 2.0
                    video_frame_path = os.path.join(self.video_dir, video_name.split('.')[0])
                    sample['video_path'] = os.path.join(self.video_dir, video_name)  # for tool call
                    if os.path.exists(video_frame_path):
                        frame_paths = os.listdir(video_frame_path)
                        frame_paths = sorted(frame_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                        frame_paths = [os.path.join(video_frame_path, frame_path) for frame_path in frame_paths]
                        total_frames = len(frame_paths)
                        if 'videommmu' in self.video_dir:
                            frame_paths.append(frame_paths[-1])
                        video_path = frame_paths
                        fps = len(frame_paths) / sample['duration']
                    else:
                        for ext in ['.mp4', '.webm', '.mkv', '.avi']:
                            path = os.path.join(self.video_dir, video_name.split('.')[0] + ext)
                            if os.path.exists(path):
                                video_path = path
                                break
                    if video_path is None:
                        raise FileNotFoundError(f"Video file for {video_name} not found.")
                    question = TEMPLATE.format(input_text=sample['question'], duration=sample['duration'])
                    content_mm = self.content_video.copy()
                    content_mm['video'] = video_path
                    content_mm['fps'] = fps
                    content_mm['draw_number'] = not self.no_number
                    content_mm['parallel'] = True
                elif 'image' in sample:
                    img_template = TEMPLATE.replace("This is a video with duration {duration} seconds.", "This is an image.")
                    question = img_template.format(input_text=sample['question'])
                    image_path = os.path.join(self.video_dir, sample['image'])
                    content_mm = self.content_image.copy()
                    content_mm['image'] = image_path
                else:
                    raise ValueError(f"Not a multimodal sample! Neither 'video' nor 'image' key found in sample: {sample}")
                messages = [
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            content_mm,
                            {"type": "text", "text": question},
                        ],
                    }
                ]
                if args.no_cache:
                    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
                else:
                    image_inputs, video_inputs, video_kwargs = cached_process_vision_info(messages, return_video_kwargs=True)
                mm_data = {}
                if image_inputs is not None:
                    mm_data["image"] = image_inputs
                if video_inputs is not None:
                    mm_data["video"] = video_inputs
                vllm_inputs = {
                    "messages": messages,
                    "multi_modal_data": mm_data,
                    "mm_processor_kwargs": {'fps': 2.0},  # set fps to 2.0
                }
                sample['vllm_inputs'] = vllm_inputs
                # ranki_print(f'In dataset, sample id: {sample["id"]}, video len = {len(video_inputs[0])} shape = {video_inputs[0][0].size} fps = {video_kwargs}')
                return sample
            
            # except Exception as e:
            #     print(f"[ERROR] idx={idx}, video={self.data[idx]['video_id']}, error: {e}")
            #     idx += 1
        raise RuntimeError("All samples from current idx onward failed.")

def execute_tools(tools, tool_call_arguments, tools_kwargs):
    tool_name = tool_call_arguments["name"]
    tool_arguments = tool_call_arguments["arguments"]
    try:
        assert tool_name in tools, f"Tool {tool_name} not found in tools list"
        tool = tools[tool_name]
        # this is a sync method
        tool_result = tool._execute(
            instance_id=str(uuid.uuid4()), 
            parameters=tool_arguments, 
            **tools_kwargs
        )
        tool_response, tool_reward_score, tool_metrics = tool_result
        return tool_response
    except Exception as e:
        return {
            "type": "error",
            "content": str(e),
        }
    
def build_system_prompt(tool_schemas):
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

def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    use_flash_attn = True
    qwen_path = 'models/Qwen2.5-VL-7B-Instruct'
    assert args.backend == 'vllm'
    llm = vllm.LLM(
        model=args.model_path,
        dtype='bfloat16',
        seed=args.seed,
        max_num_batched_tokens=8192,
        # disable_chunked_mm_input=True,
        # limit_mm_per_prompt={'image': 0, 'video': 2},
        gpu_memory_utilization=0.7,
        enforce_eager=True,
    )
    processor = AutoProcessor.from_pretrained(qwen_path, trust_remote_code=True, use_fast=True)
    ranki_print("Load model and processor success!")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except Exception as e:
            ranki_print(f'mkdir Except: {e}')
    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    args.answers_file = answers_file

    dataset = VideoQADataset(args, processor)
    batch_size = 8 // args.repeat_times
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True, 
        drop_last=False,
        collate_fn=lambda batch: batch, 
    )
    ans_file = open(answers_file, "a")

    # New batch vllm generation:
    target_template = ["</{tag}>", " </{tag}>", "</{tag}>\n", " </{tag}>\n", "</{tag}>\n\n", " </{tag}>\n\n"] 
    target_tags = ["tool_call", "answer"]
    target_sequences = [template.format(tag=tag) for tag in target_tags for template in target_template]
    sampling_params = vllm.SamplingParams(
        temperature=0.01,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=1024,
        stop=target_sequences,
        include_stop_str_in_output=True,  # Important
    )
    for samples in tqdm(dataloader, desc=f"cuda:{args.chunk_idx}"):
        # start multi-turn conversation
        cur_batch_size = len(samples)
        active_mask = torch.ones(cur_batch_size, dtype=torch.bool, device='cuda')  # if sample is active
        turns_stats = torch.ones(cur_batch_size, dtype=torch.int, device='cuda')  # number of turns
        valid_action_stats = torch.zeros(cur_batch_size, dtype=torch.int, device='cuda')  # number of valid actions
        valid_tool_call_stats = torch.zeros(cur_batch_size, dtype=torch.int, device='cuda')  # number of valid tools
        valid_tool_exec_stats = torch.zeros(batch_size, dtype=torch.int, device='cuda')  # number of valid tools
        active_num_list = [active_mask.sum().item()]
        response_ids = [[] for _ in range(cur_batch_size)]
        response_preds = ["" for _ in range(cur_batch_size)]

        max_turns = 2
        vllm_inputs = [sample['vllm_inputs'] for sample in samples]
        for step in range(max_turns + 1):
            ranki_print(f'>>> [step {step} / {max_turns + 1}]')
            active_id_list = torch.where(active_mask)[0].tolist()
            active_num = len(active_id_list)
            active = active_num > 0
            if not active:
                break
            active_vllm_inputs = []
            for idx in active_id_list:
                vid_len = [len(x) for x in vllm_inputs[idx]["multi_modal_data"]['video']]
                # ranki_print(f'before generation, {idx=} {vid_len=} {vllm_inputs[idx]["mm_processor_kwargs"]=} {vllm_inputs[idx]["messages"]=}')
                # ranki_print(f'before generation, {idx=} {vid_len=} {vllm_inputs[idx]["mm_processor_kwargs"]=}, sample_id = {samples[idx]["id"]}')
                curr_prompts = processor.apply_chat_template(vllm_inputs[idx]["messages"], add_generation_prompt=True, tokenize=False)
                active_vllm_inputs.append({
                    "prompt": curr_prompts,
                    "multi_modal_data": vllm_inputs[idx]["multi_modal_data"],
                    "mm_processor_kwargs": vllm_inputs[idx]["mm_processor_kwargs"],
                })
            outputs = llm.generate(
                prompts=active_vllm_inputs,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            outputs_flattened = [out for output in outputs for out in output.outputs]  # trick
            # process outputs
            for i, idx in enumerate(active_id_list):
                response_text = outputs_flattened[i].text
                curr_response_ids = outputs_flattened[i].token_ids
                response_ids[idx].extend(curr_response_ids)
                vllm_inputs[idx]["messages"].append({
                    "role": "assistant",
                    "content": response_text,
                })
                response_preds[idx] += response_text
                if step < max_turns:
                    # preprocess tool call
                    res = parse_output(response_text)
                    if res["type"] == "tool":
                        tool_call_arguments = res["content"]
                        tools_kwargs = {
                            "video_path": samples[idx]["video_path"],
                            "duration": samples[idx]["duration"],
                            "max_frames": 64,
                            "draw_number": not args.no_number,
                            "parallel": True,
                            "fps": 2,
                        }
                        tool_res = execute_tools(dataset.tools, tool_call_arguments, tools_kwargs)  # important
                        if tool_res["type"] == "result":
                            tool_result = tool_res["content"]
                            valid_tool_exec_stats[idx] += 1
                        elif tool_res["type"] == "error":
                            tool_result = tool_res["content"]
                        valid_action_stats[idx] += 1
                        valid_tool_call_stats[idx] += 1
                    elif res["type"] == "answer":
                        answer = res["content"]
                        tool_result = ""
                        active_mask[idx] = False
                        valid_action_stats[idx] += 1
                    elif res["type"] == "error":
                        error_msg = res["content"]
                        if error_msg == "Error: response parse failed":
                            tool_result = ""
                        else:
                            tool_result = error_msg
                    if tool_result != "":
                        if isinstance(tool_result, str):
                            new_content = tool_result
                            new_prompt_str = tool_result
                            new_videos = None
                            new_fps = None
                        elif isinstance(tool_result, dict):
                            teaser = f"Video clip from {tool_result['start_time']:.2f} to {tool_result['end_time']:.2f} seconds."
                            new_content = [
                                tool_result["ele"],
                                {
                                    "type": "text",
                                    "text": teaser
                                }
                            ]
                            new_prompt_str = "<|vision_start|><|video_pad|><|vision_end|>" + teaser
                            new_videos = [tool_result["video"]]
                            new_fps = [tool_result["fps"]]  # Here is a bug of VLLM V1 Engine
                            vllm_inputs[idx]["multi_modal_data"]["video"].extend(new_videos)
                            # vllm_inputs[idx]["mm_processor_kwargs"]["fps"].extend(new_fps)
                        new_message = {
                            "role": "tool",
                            "content": new_content,
                        }
                        vllm_inputs[idx]["messages"].append(new_message)
                        new_prompt = "<|im_end|>\n"
                        new_prompt += f"<|im_start|>{new_message['role']}\n"
                        new_prompt += new_prompt_str
                        new_prompt += "<|im_end|>\n"
                        new_prompt += "<|im_start|>assistant\n"
                        new_inputs = processor(text=[new_prompt], images=None, videos=new_videos, fps=new_fps, return_tensors="pt")  # a list [1, L]
                        new_token_ids = list(new_inputs.input_ids[0])
                        response_ids[idx].extend(new_token_ids)
                        video_lens = [len(x) for x in vllm_inputs[idx]["multi_modal_data"]["video"]]
                        ranki_print(f'after add tools, {idx=} {video_lens=}, {vllm_inputs[idx]["mm_processor_kwargs"]["fps"]=}')

        for idx, sample in enumerate(samples):
            pred_text = response_preds[idx]
            text = processor.apply_chat_template(vllm_inputs[idx]["messages"], add_generation_prompt=False, tokenize=False)
            sample_set = {
                'id': sample['id'],
                'question': sample['question'],
                'answer': sample['answer'],
                'pred': pred_text,
                'text': text,
                'meta_info': {
                    'active_mask': active_mask[idx].item(),
                    'turns_stats': turns_stats[idx].item(),
                    'valid_action_stats': valid_action_stats[idx].item(),
                    'valid_tool_call_stats': valid_tool_call_stats[idx].item(),
                    'valid_tool_exec_stats': valid_tool_exec_stats[idx].item(),
                }
            }
            if 'question_type' in sample:
                sample_set['question_type'] = sample['question_type']
            ans_file.write(json.dumps(sample_set) + "\n")
            ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max_pixels", type=int, default=224*224)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--resized_width", type=int, default=None)
    parser.add_argument("--resized_height", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--no-cache", action="store_true", default=False)
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "transformers"])
    parser.add_argument("--repeat_times", type=int, default=1)
    parser.add_argument("--no-number", action="store_true", default=False)
    
    global args
    args = parser.parse_args()
    args.seed = 42

    set_seed(args.seed)
    run_inference(args)
