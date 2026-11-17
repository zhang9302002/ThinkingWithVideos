#    Copyright 2024 Flash-VStream Authors 
#
#    Licensed under the Apache License, Version 2.0 (the "License"); 
#    you may not use this file except in compliance with the License. 
#    You may obtain a copy of the License at 
#
#        http://www.apache.org/licenses/LICENSE-2.0 
#
#    Unless required by applicable law or agreed to in writing, software 
#    distributed under the License is distributed on an "AS IS" BASIS, 
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
#    See the License for the specific language governing permissions and 
#    limitations under the License. 

from collections import defaultdict
import csv
import json
import os
import argparse
import random
import re
import subprocess
import multiprocessing
import logging
import math
import ast


def get_dataset_info(dataset):
    dataset_list = [
        {'dataset_name': 'next_gqa', 'frame_dir': 'data/nextgqa/video_14400frames_fps2', 'data_file': 'data/nextgqa/test_set_grpo_src.json'},  # gqa task
        {'dataset_name': 'rextime', 'frame_dir': 'data/rextime/video_14400frames_fps2', 'data_file': 'data/rextime/test_set_grpo_exist_src.json'},  # gqa task
        {'dataset_name': 'rextime_val', 'frame_dir': 'data/rextime/video_14400frames_fps2', 'data_file': 'data/rextime/val_set_grpo_exist_src.json'},  # gqa task
    ]
    for d in dataset_list:
        if d['dataset_name'] == dataset:
            return d
    return None


def exec(cmd, sub=False, device=None):
    print(f'exec: {cmd}')
    if not sub:
        if isinstance(cmd, list):
            cmd = ' '.join(cmd)
        os.system(cmd)
    else:
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = device
        subprocess.run(cmd, env=my_env)

def launch_multi_gpu_eval(args, dataset_name, frame_dir, data_file, evaluation_name='evaluation'):
    model_path = args.model_path
    num_chunks = args.num_chunks
    output_dir = os.path.join(args.output_dir, evaluation_name, dataset_name)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    print(f'launch_multi_gpu_eval: args={args}')
    if not args.test:
        processes = []
        for idx in range(0, num_chunks):
            cmd = [ "python3", args.inference_file,
                "--model-path", model_path,
                "--video_dir", frame_dir,
                "--gt_file", data_file,
                "--output_dir", output_dir,
                "--output_name", "pred",
                "--num-chunks", str(num_chunks),
                "--chunk-idx", str(idx),
                "--max_pixels", str(args.max_pixels),
            ]
            if args.fps:
                cmd += ["--fps", str(args.fps)]
            if args.max_frames:
                cmd += ["--max_frames", str(args.max_frames)]
            if args.resized_height:
                cmd += ["--resized_height", str(args.resized_height)]
            if args.resized_width:
                cmd += ["--resized_width", str(args.resized_width)]
            if args.lora_path:
                cmd += ["--lora-path", args.lora_path]
            if args.no_cache:
                cmd += ["--no-cache"]
            logging.debug(f"Starting subprocess with command: {' '.join(cmd)}")
            # Start subprocess and capture output
            my_env = os.environ.copy()
            my_env["CUDA_VISIBLE_DEVICES"] = str(idx)
            p = subprocess.Popen(cmd, env=my_env)
            processes.append(p)
        for idx, p in enumerate(processes):
            stdout, stderr = p.communicate()
            logging.debug(f"Subprocess {idx} stdout: {stdout}")
            if stderr:
                logging.error(f"Subprocess {idx} stderr: {stderr}")
            if p.returncode != 0:
                logging.error(f"Subprocess {idx} failed with return code {p.returncode}")
            else:
                logging.debug(f"Subprocess {idx} completed successfully")
    return output_dir, data_file


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "From",
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[A-Z]", s):
        return ""

    matches = re.search(r"[A-Z]", s)
    if matches is None:
        return ""
    return matches[0]

def extract_answer(text):
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return text

def cal_iou_precision_recall(gt_list, pred_list):
    start1, end1 = gt_list
    start2, end2 = pred_list
    if start1 > end1:
        print(f'>> wrong annotation gt: {gt_list}, pred: {pred_list}')
        return -2, -2, -2
    elif end2 == -1 or start1 > end1 or start2 > end2:
        print(f'>> err gt: {gt_list}, pred: {pred_list}')
        return -1, -1, -1
    # 计算交集的起始和结束时间
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    # 计算交集和并集的长度
    eps = 1e-4
    inter_length = max(0, inter_end - inter_start)
    pred_length = end2 - start2 + eps
    gt_length = end1 - start1 + eps
    union_length = pred_length + gt_length - inter_length + eps
    # 计算 IoU
    iou = inter_length / union_length
    precision = inter_length / pred_length
    recall = inter_length / gt_length
    return iou, precision, recall

def extract_time_range(paragraph: str) -> list:
    candidates = re.split(r"[!?\n]", paragraph)

    timestamps = []
    time_regex = re.compile(r"\b(\d+\.\d+\b|\b\d+)\b")  # time formats (e.g., 18, 18.5)
    for sentence in candidates:
        time = re.findall(time_regex, sentence)
        new_times = []
        for tim in time or []:
            time_in_sec = float(tim)
            new_times.append(time_in_sec)
        new_timestamps = [(new_times[i], new_times[i + 1]) for i in range(0, len(new_times), 2) if i + 1 < len(new_times)]
        timestamps.extend(new_timestamps)
    for start, end in reversed(timestamps):
        if end < start:
            continue
        return start, end
    return -1, -1

def cal_mcq_reward(predict_str: str, ground_truth: str, extra_info: dict=None) -> float:
    pred_ans = extract_answer(predict_str)
    pred_ans = extract_characters_regex(pred_ans)
    gt_ans = extract_characters_regex(ground_truth)
    reward = 1.0 if pred_ans.strip() == gt_ans.strip() else 0.0
    log_data = {
        "reward_name": "mcq_reward_func",
        "custom_reward_func": reward,
    }
    return reward, log_data

def cal_iou_reward(predict_str: str, ground_truth: str, extra_info: dict=None):
    gt_list = extract_time_range(ground_truth)
    pred_list = extract_time_range(predict_str)
    iou, precision, recall = cal_iou_precision_recall(gt_list, pred_list)
    iou = max(0, iou)  # neglect special cases
    log_data = {
        "reward_name": "iou_reward_func",
        "custom_reward_func": iou,
    }
    return iou, log_data


def calc_eval_result(output_path, gt_file, num_chunks, data_path):
    if num_chunks > 1:
        pred_contents = []
        for _idx in range(num_chunks):
            file = os.path.join(output_path, f"{num_chunks}_{_idx}.json")
            pred_contents += [json.loads(line) for line in open(file)]
    else:
        file = os.path.join(output_path, f"pred.json")
        pred_contents = [json.loads(line) for line in open(file)]
    gt_contents = json.load(open(gt_file))
    gt_id_set = set([x['id'] for x in gt_contents])
    pred_contents = [x for x in pred_contents if x['id'] in gt_id_set]

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    parse_fail_cnt = 0
    max_zoom_round = 0
    if 'nextgqa' in gt_file:
        from eval_GQA_old import compute_iou
        info_file = 'data/nextgqa/test_set_grpo.json'
        info = json.load(open(info_file))
        info_dic = {x['id']: x for x in info}
    else:
        info_dic = {}
    for sample in pred_contents:
        acc, _ = cal_mcq_reward(sample['pred'], sample['answer'])
        if sample['id'] in info_dic:  # multiple gt range vs 1 predict range
            pred_list = extract_time_range(sample['pred'])
            iou = compute_iou([pred_list], info_dic[sample['id']]["solution"])
        else:
            iou, _ = cal_iou_reward(sample['pred'], sample['answer'])

        prediction_set[sample['id']] = {
            'iou': iou,
            'recall@0.3': 1 if iou >= 0.3 else 0,
            'recall@0.5': 1 if iou >= 0.5 else 0,
            'recall@0.7': 1 if iou >= 0.7 else 0,
            'acc': acc,
            **sample
        }
        
        
    
    json_path = os.path.join(output_path, 'result.json')
    with open(json_path, "w") as f:
        json.dump(prediction_set, f, indent=4)
    print("[main] All evaluation completed!")

    class ScoreMeter:
        def __init__(self):
            self.score_sum = 0
            self.count = 0
            self.yes_count = 0
            self.no_count = 0
            self.score_dict = {'yes': defaultdict(int), 'no': defaultdict(int)}

        def add_score(self, score, pred):
            self.score_sum += score
            self.count += 1
            pred_lower = pred.lower()
            if 'yes' in pred_lower:
                self.yes_count += 1
                self.score_dict['yes'][score] += 1
            elif 'no' in pred_lower:
                self.no_count += 1
                self.score_dict['no'][score] += 1

        def get_average_score(self):
            res = (self.score_sum / self.count) if self.count else 0
            return f"{res * 100:.6f}"

        def get_accuracy(self, response_type):
            if response_type == 'yes':
                res =  (self.yes_count / self.count) if self.count else 0
            elif response_type == 'no':
                res = (self.no_count / self.count) if self.count else 0
            else:
                res = 0
            return f"{res * 100:.6f}"
        
    meter_dic = defaultdict(lambda: ScoreMeter())
    cnt = 0
    for key, result in prediction_set.items():
        # Computing score
        cnt += 1
        meter_dic["recall@0.3"].add_score(result['recall@0.3'], 'yes')
        meter_dic["recall@0.5"].add_score(result['recall@0.5'], 'yes')
        meter_dic["recall@0.7"].add_score(result['recall@0.7'], 'yes')
        meter_dic["miou"].add_score(result['iou'], 'yes')
        meter_dic["acc"].add_score(result['acc'], 'yes')

    output = "Result:\n"
    output += f"total samples = {cnt}\n"
    output += f"parse fail = {parse_fail_cnt}\n"
    key_list = []
    value_list = []
    for key, meter in meter_dic.items():
        output += f"{key}, {meter.get_average_score()}\n"
        key_list.append(key)
        value_list.append(meter.get_average_score())
    output += "\n"

    for key in key_list:
        output += f"{key}, "
    output = output.rstrip(', ')  # Remove the trailing comma and space
    output += "\n"
    for value in value_list:
        output += f"{value}, "
    output = output.rstrip(', ')  # Remove the trailing comma and space
    output += "\n"
    
    print(output)
    csv_path = json_path.replace(".json", ".csv")
    with open(csv_path, 'w') as f:
        f.write(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default='~')
    parser.add_argument("--evaluation_name", type=str, default='evaluation')
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max_pixels", type=int, default=224*224)
    parser.add_argument("--resized_width", type=int, default=None)
    parser.add_argument("--resized_height", type=int, default=None)
    parser.add_argument("--prompt_type", type=str, help='not_used')
    parser.add_argument("--inference_file", type=str, default="inference.py")
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    info = get_dataset_info(args.dataset)
    if info is None:
        print(f'[main] ERROR {args.dataset} dataset was not found!')
        exit(0)
    print(f'[main] Execute {args.dataset} evaluation')
    out_dir, gt_file = launch_multi_gpu_eval(args, **info, evaluation_name=args.evaluation_name)
    print(f'[main] Execute {args.dataset} evaluation')
    calc_eval_result(out_dir, gt_file, args.num_chunks, info['data_file'])

