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

TEMPLATE = os.getenv("MY_PROMPT_TEMPLATE", default="Please find the visual event described by a sentence in the video, determining its starting and ending times. The format should be: 'The event happens in the start time - end time'. For example, The event 'person turn a light on' happens in the 24.30 - 30.42 seconds. Now I will give you the textual sentence: {input_text}. Please return its start time and end time.")

VIDEO_INFO_CACHE = {}

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        while idx < len(self.data):
            try:
                sample = self.data[idx]
                if not 'question' in sample:
                    sample['question'] = sample['text']
                if not 'answer' in sample:
                    sample['answer'] = str(sample['solution'])
                
                if 'video' in sample or 'video_id' in sample:
                    video_name = sample['video_id'] if 'video_id' in sample else sample['video']
                    video_path, fps = None, 2.0
                    video_frame_path = os.path.join(self.video_dir, video_name.split('.')[0])
                    if os.path.exists(video_frame_path):
                        frame_paths = os.listdir(video_frame_path)
                        frame_paths = sorted(frame_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                        frame_paths = [os.path.join(video_frame_path, frame_path) for frame_path in frame_paths]
                        total_frames = len(frame_paths)
                        if args.max_frames is not None and total_frames > args.max_frames:
                            idx = torch.linspace(0, total_frames - 1, args.max_frames).round().long().tolist()
                            if 'videommmu' in self.video_dir:
                                idx.append(total_frames - 1)
                                idx.append(total_frames - 1)
                            frame_paths = [frame_paths[i] for i in idx]
                            fps = args.max_frames / max(total_frames, 1e-6) * fps
                        video_path = frame_paths
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
                        "role": "user",
                        "content": [
                            content_mm,
                            {"type": "text", "text": question},
                        ],
                    }
                ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                if args.no_cache:
                    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
                else:
                    image_inputs, video_inputs, video_kwargs = cached_process_vision_info(messages, return_video_kwargs=True)

                mm_data = {}
                if image_inputs is not None:
                    mm_data["image"] = image_inputs
                if video_inputs is not None:
                    mm_data["video"] = video_inputs
                llm_inputs = {
                    "prompt": text,
                    "multi_modal_data": mm_data,
                    "mm_processor_kwargs": video_kwargs,
                }
                sample['llm_inputs'] = llm_inputs

                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    fps=video_kwargs["fps"],
                    padding=True,
                    padding_side='left',
                    return_tensors="pt",
                )
                if video_inputs is not None:
                    ranki_print(f'In dataset, {inputs.input_ids.shape=}, {inputs.video_grid_thw=}')
                # 返回你需要的内容
                return sample
            
            except Exception as e:
                print(f"[ERROR] idx={idx}, video={self.data[idx]['video_id']}, error: {e}")
                idx += 1
        raise RuntimeError("All samples from current idx onward failed.")


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
        raise RuntimeError("未找到模型分片文件")
    # 如果所有文件的world_size一致，直接用
    if len(world_sizes) == 1:
        return list(world_sizes)[0]
    # 否则用rank数量猜测
    return max(ranks) + 1


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    use_flash_attn = True
    qwen_path = 'models/Qwen2.5-VL-7B-Instruct'
    try:
        if args.backend == 'vllm':
            llm = vllm.LLM(
                model=args.model_path,
                dtype='bfloat16',
                seed=args.seed,
                max_num_batched_tokens=8192,
                gpu_memory_utilization=0.7,
            )
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_path, 
                use_sliding_window=True,
                device_map="cuda", 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2" if use_flash_attn else "eager",
            ).eval()
    except Exception as e:
        ranki_print(f'Load model Except: {e}')
        model_path = args.model_path
        state_dict = defaultdict(list)
        world_size = auto_detect_world_size(model_path)
        for rank in range(world_size):
            filepath = f"{model_path}/model_world_size_{world_size}_rank_{rank}.pt"
            print('loading', filepath)
            this_state_dict = torch.load(filepath)
            for key, value in this_state_dict.items():
                state_dict[key].append(value.to_local())
        for key in state_dict:
            state_dict[key] = torch.cat(state_dict[key], dim=0)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_path, 
            use_sliding_window=True,
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if use_flash_attn else "eager",
        ).eval()
        model.load_state_dict(state_dict)
        model.eval()

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

    target_template = ["</{tag}>", " </{tag}>", "</{tag}>\n", " </{tag}>\n", "</{tag}>\n\n", " </{tag}>\n\n"] 
    target_tags = ["tool_call", "answer"]
    target_sequences = [template.format(tag=tag) for tag in target_tags for template in target_template]
    if args.repeat_times > 1:
        sampling_params = vllm.SamplingParams(
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.05,
            max_tokens=1024,
            stop=target_sequences,
            include_stop_str_in_output=True,  # Important
            n=args.repeat_times,
        )
    else:
        sampling_params = vllm.SamplingParams(
            temperature=0.01,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=1024,
            stop=target_sequences,
            include_stop_str_in_output=True,  # Important
        )

    for samples in tqdm(dataloader, desc=f"cuda:{args.chunk_idx}"):
        if args.backend == 'vllm':
            batch_llm_inputs = [sample['llm_inputs'] for sample in samples]
            outputs = llm.generate(batch_llm_inputs, sampling_params=sampling_params)
        else:
            text = [sample['llm_inputs']['prompt'] for sample in samples]
            image_inputs = [
                image for sample in samples if 'image' in sample['llm_inputs']['multi_modal_data']
                for image in sample['llm_inputs']['multi_modal_data']['image']
            ]
            video_inputs = [
                video for sample in samples if 'video' in sample['llm_inputs']['multi_modal_data']
                for video in sample['llm_inputs']['multi_modal_data']['video']
            ]
            fps_inputs = [
                fps for sample in samples if 'video' in sample['llm_inputs']['multi_modal_data']
                for fps in sample['llm_inputs']['mm_processor_kwargs']['fps']
            ]
            ranki_print(f'Inference: fps_inputs={fps_inputs}')
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                fps=fps_inputs,
                padding=True,
                padding_side='left',
                return_tensors="pt",
            )
            input_ids = inputs.input_ids.cuda()
            attention_mask = inputs.attention_mask.cuda()
            pixel_values_videos = inputs.pixel_values_videos.cuda()
            video_grid_thw = inputs.video_grid_thw.cuda()
            ranki_print(f'video_grid_thw={video_grid_thw}')
            with torch.inference_mode():
                if args.repeat_times > 1:
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values_videos=pixel_values_videos,
                        video_grid_thw=video_grid_thw,
                        max_new_tokens=1024,
                        temperature=1,
                        top_p=1,
                        do_sample=True,
                        num_return_sequences=args.repeat_times,
                        stop_strings=target_sequences,
                        tokenizer=processor.tokenizer,
                    )
                else:
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values_videos=pixel_values_videos,
                        video_grid_thw=video_grid_thw,
                        max_new_tokens=1024,
                        top_k=1,
                        do_sample=False,
                        stop_strings=target_sequences,
                        tokenizer=processor.tokenizer,
                    )
            generated_ids_trimmed = []
            for i in range(batch_size):
                in_ids = inputs.input_ids[i]
                for j in range(args.repeat_times):
                    out_ids = generated_ids[i * args.repeat_times + j]
                    out_ids_trimmed = out_ids[len(in_ids):]
                    generated_ids_trimmed.append(out_ids_trimmed)
            outputs = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        for sample_idx, sample in enumerate(samples):
            for repeat_id in range(args.repeat_times):
                if args.backend == 'vllm':
                    pred_text = outputs[sample_idx].outputs[repeat_id].text.strip()
                else:
                    idx = sample_idx * args.repeat_times + repeat_id
                    pred_text = outputs[idx].strip()
                sample_set = {
                    'id': sample['id'],
                    'repeat_id': repeat_id,
                    'question': sample['question'],
                    'answer': sample['answer'],
                    'pred': pred_text,
                    'text': sample['llm_inputs']['prompt'],
                }
                if 'solution' in sample:
                    sample_set['solution'] = sample['solution']
                if 'question_type' in sample:
                    sample_set['question_type'] = sample['question_type']
                ans_file.write(json.dumps(sample_set) + "\n")
                # ranki_print(f"input={sample['llm_inputs']['prompt']}, output={pred_text}")
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
    
    global args
    args = parser.parse_args()
    args.seed = 42

    set_seed(args.seed)
    run_inference(args)
