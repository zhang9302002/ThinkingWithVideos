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
import numpy as np


def get_dataset_info(dataset):
    dataset_list = [
        {'dataset_name': 'vsibench', 'frame_dir': 'data/vsibench/video_14400frames_fps2', 'data_file': 'data/vsibench/test_set_grpo.json'},  # vqa task
        {'dataset_name': 'mmvu', 'frame_dir': 'data/mmvu/video_14400frames_fps2', 'data_file': 'data/mmvu/valid_set_grpo.json'},  # vqa task
        {'dataset_name': 'videommmu', 'frame_dir': 'data/videommmu/video_14400frames_fps2', 'data_file': 'data/videommmu/test_set_grpo.json'},  # vqa task
        {'dataset_name': 'longvideo-reason', 'frame_dir': 'data/longvideo-reason/video_14400frames_fps2', 'data_file': 'data/longvideo-reason/test_set_grpo_src_exist.json'},  # vqa task
        {'dataset_name': 'lvbench', 'frame_dir': 'data/lvbench/video_14400frames_fps2', 'data_file': 'data/lvbench/test_qa.json'},  # vqa task
        {'dataset_name': 'videomme', 'frame_dir': 'data/videomme/video_14400frames_fps2', 'data_file': 'data/videomme/test_qa.json'},  # vqa task
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
            if args.no_number:
                cmd += ["--no-number"]
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
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:" "Best option:",
        "Answer",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[A-Z]", s):
        return ""

    matches = re.search(r"[A-Z]", s)
    if matches is None:
        return ""
    return matches[0]

NA_QUESTION_TYPES = [
    "object_abs_distance",  # vsi
    "object_counting",  # vsi
    "object_size_estimation",  # vsi
    "room_size_estimation",  # vsi
    "open-ended",  # mmvu
]

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

def extract_float(text):
    match = re.search(r'[-+]?\d*\.\d+|\d+', text)
    if match:
        return float(match.group())
    else:
        return None
    
def extract_answer(text):
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return text

def calc_eval_result(output_path, gt_file, num_chunks, data_path):
    if num_chunks > 1:
        pred_contents = []
        for _idx in range(num_chunks):
            file = os.path.join(output_path, f"{num_chunks}_{_idx}.json")
            try:
                for line in open(file):
                    pred_contents.append(json.loads(line))
            except:
                print(f"[main] parse fail: {file}, line: {line}")
                raise Exception
        
    else:
        file = os.path.join(output_path, f"pred.json")
        pred_contents = [json.loads(line) for line in open(file)]
    gt_contents = json.load(open(gt_file))
    gt_id_set = set([x['id'] for x in gt_contents])
    gt_id_dict = {x['id']: x for x in gt_contents}
    pred_contents = [x for x in pred_contents if x['id'] in gt_id_set]

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    parse_fail_cnt = 0
    max_zoom_round = 0
    for sample in pred_contents:
        try:
            ans = extract_answer(sample['pred'])
            acc = 0.0
            if 'question_type' in sample and sample['question_type'] in NA_QUESTION_TYPES:
                ans_float = extract_float(ans)
                if ans_float is None:
                    acc = 0.0
                else:
                    acc = mean_relative_accuracy(ans_float, float(sample["answer"]), 0.50, 0.95, 0.05)
            else:
                if extract_characters_regex(ans) == extract_characters_regex(sample["answer"]):
                    acc = 1.0
            prediction_set[sample['id']] = {
                'acc': acc,
                **sample
            }
        except Exception as e:
            parse_fail_cnt += 1
            print(f"[main] parse fail: {e}")
        
    
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
        meter_dic["acc"].add_score(result['acc'], 'yes')
        typ = None
        if 'question_type' in result:
            typ = result['question_type']
        elif 'question_type' in gt_id_dict[key]:
            typ = gt_id_dict[key]['question_type']
        if typ is not None:
            if isinstance(typ, str):
                meter_dic[typ].add_score(result['acc'], 'yes')
            elif isinstance(typ, list):
                for t in typ:
                    meter_dic[t].add_score(result['acc'], 'yes')

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
    parser.add_argument("--no_number", action="store_true")
    args = parser.parse_args()

    info = get_dataset_info(args.dataset)
    if info is None:
        print(f'[main] ERROR {args.dataset} dataset was not found!')
        exit(0)
    print(f'[main] Execute {args.dataset} evaluation')
    out_dir, gt_file = launch_multi_gpu_eval(args, **info, evaluation_name=args.evaluation_name)
    print(f'[main] Execute {args.dataset} evaluation')
    calc_eval_result(out_dir, gt_file, args.num_chunks, info['data_file'])

