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

from verl.utils.reward_score import default_compute_score
from tqdm import tqdm

def get_dataset_info(dataset):
    dataset_list = [
        {'dataset_name': 'charades_sta', 'frame_dir': './data/charades/video_14400frames_fps2', 'data_file': './data/charades/test_set.json'},
        {'dataset_name': 'charades_sta_train', 'frame_dir': './data/charades/video_14400frames_fps2', 'data_file': './data/charades/train_set_grpo_src.json'},
        {'dataset_name': 'actnet_tg', 'frame_dir': './data/actnet/video_14400frames_fps2', 'data_file': './data/actnet/test_set_grpo_new.json'},
        {'dataset_name': 'actnet_tg_full', 'frame_dir': './data/actnet/video_14400frames_fps2', 'data_file': './data/actnet/test_set_grpo_valid_new.json'},
        {'dataset_name': 'actnet_tg_train', 'frame_dir': './data/actnet/video_14400frames_fps2', 'data_file': './data/actnet/train_set_grpo_new_12k_src.json'},
        {'dataset_name': 'tvg_bench', 'frame_dir': './data/TimeR1-Dataset/tvgbench_data_cutted', 'data_file': './data/TimeR1-Dataset/test_set_grpo.json'},
        {'dataset_name': 'video_r1_01_video_train', 'frame_dir': './data/Video-R1-data/video_14400frames_fps2', 'data_file': './data/Video-R1-data/train_set_grpo_01_src_exist_video.json'},
        {'dataset_name': 'video_r1_01_image_train', 'frame_dir': './data/Video-R1-data', 'data_file': './data/Video-R1-data/train_set_grpo_01_src_exist_image.json'},
        {'dataset_name': 'nextgqa_train', 'frame_dir': './data/nextgqa/video_14400frames_fps2', 'data_file': './data/nextgqa/train_set_grpo_src.json'},
        {'dataset_name': 'rextime_train', 'frame_dir': './data/rextime/video_14400frames_fps2', 'data_file': './data/rextime/train_set_grpo_exist_src.json'},
        {'dataset_name': 'video_r1_image80k_0', 'frame_dir': './data/Video-R1-data', 'data_file': './data/Video-R1-data/train_set_grpo_image80k_src_exist_0.json'},
        {'dataset_name': 'video_r1_image80k_1', 'frame_dir': './data/Video-R1-data', 'data_file': './data/Video-R1-data/train_set_grpo_image80k_src_exist_1.json'},
        {'dataset_name': 'video_r1_image80k_2', 'frame_dir': './data/Video-R1-data', 'data_file': './data/Video-R1-data/train_set_grpo_image80k_src_exist_2.json'},
        {'dataset_name': 'video_r1_image80k_3', 'frame_dir': './data/Video-R1-data', 'data_file': './data/Video-R1-data/train_set_grpo_image80k_src_exist_3.json'},
        {'dataset_name': 'vidchapter_train_0', 'frame_dir': './data/vidchapters/video_14400frames_fps2', 'data_file': './data/vidchapters/train_set_grpo_src_0.json'},
        {'dataset_name': 'vidchapter_train_1', 'frame_dir': './data/vidchapters/video_14400frames_fps2', 'data_file': './data/vidchapters/train_set_grpo_src_1.json'},
        {'dataset_name': 'vidchapter_train_2', 'frame_dir': './data/vidchapters/video_14400frames_fps2', 'data_file': './data/vidchapters/train_set_grpo_src_2.json'},
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
            if args.backend:
                cmd += ["--backend", args.backend]
            if args.repeat_times:
                cmd += ["--repeat_times", str(args.repeat_times)]
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


def calc_eval_result(output_path, gt_file, num_chunks, data_path):
    if num_chunks > 1:
        pred_contents = []
        for _idx in tqdm(range(num_chunks), desc="Loading results"):
            file = os.path.join(output_path, f"{num_chunks}_{_idx}.json")
            pred_contents += [json.loads(line) for line in open(file)]
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
    for sample in tqdm(pred_contents, desc="Compute rewards"):
        info = gt_id_dict[sample['id']]
        res = default_compute_score(
            data_source=info['data_source'],
            solution_str=sample['pred'], 
            ground_truth=sample['answer'],
        )
        key = sample['id']
        if 'repeat_id' in sample:
            key += f'_{sample["repeat_id"]}'
        prediction_set[key] = {
            'reward_name': res["reward_name"],
            'custom_reward_func': res["custom_reward_func"],
            **sample
        }
        # parse meta infos
        if 'meta_info' in sample.keys():
            prediction_set[sample['id']].update({
                'meta_info': sample['meta_info']
            })
        
    
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
    for key, result in tqdm(prediction_set.items(), desc="Compute metrics"):
        # Computing score
        cnt += 1
        meter_dic["custom_reward_func"].add_score(result['custom_reward_func'], 'yes')
        reward_name = result['reward_name']
        meter_dic[reward_name].add_score(result['custom_reward_func'], 'yes')

        if 'meta_info' in result.keys():
            meta_info = result['meta_info']
            for k, v in meta_info.items():
                meter_dic[f"{k}"].add_score(v, 'yes')

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
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "transformers"])
    parser.add_argument("--repeat_times", type=int, default=None)
    args = parser.parse_args()

    info = get_dataset_info(args.dataset)
    if info is None:
        print(f'[main] ERROR {args.dataset} dataset was not found!')
        exit(0)
    print(f'[main] Execute {args.dataset} evaluation')
    out_dir, gt_file = launch_multi_gpu_eval(args, **info, evaluation_name=args.evaluation_name)
    print(f'[main] Execute {args.dataset} evaluation')
    calc_eval_result(out_dir, gt_file, args.num_chunks, info['data_file'])

