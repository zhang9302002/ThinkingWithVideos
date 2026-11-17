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


def get_dataset_info(dataset):
    dataset_list = [
        {'dataset_name': 'vidi', 'frame_dir': 'data/vidi/video_14400frames_fps2', 'data_file': 'data/vidi/test_set_grpo.json'},
        {'dataset_name': 'vidi_src', 'frame_dir': 'data/vidi/video_14400frames_fps2', 'data_file': 'data/vidi/test_set_grpo_src.json'},
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

def extract_answer(llm_message):
    answer = re.findall(r'[A-E]', llm_message)
    if len(answer) == 0:
        print('No answer found')
        answer = random.choice(['A', 'B', 'C', 'D', 'E'])
    else:
        answer = answer[0]
    map2idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    answer = map2idx[answer]
    return answer

def extract_time_range_old(time_range_str):
    # 定义一个正则表达式模式，用于匹配时间区间
    pattern = re.compile(r"From <(\d+)> to <(\d+)>")
    pattern = re.compile(r"From <(\d+)[^\d]* to <(\d+)[^\d]*")
    match = pattern.search(time_range_str)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        return start, end
    
    pattern = re.compile(r"From (\d+) to (\d+)")
    pattern = re.compile(r"From (\d+)[^\d]* to (\d+)[^\d]*")
    match = pattern.search(time_range_str)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        return start, end
    return -1, -1

# copied from lmms-eval
def extract_time_range_old2(paragraph):
    # If there is CoT, omit <think> ... </think>
    answer_match = re.search(r"<answer>(.*?)</answer>", paragraph)
    if answer_match:
        print(f'[Replace] before >>> {paragraph}')
        paragraph = answer_match.group(1).strip()
        print(f'[Replace] after <<< {paragraph}')

    prompt = "A specific example is : 20.8 - 30.0 seconds".lower()
    paragraph = paragraph.lower().replace(prompt, "").replace("to", "-")
    # Split text into sentences based on common delimiters
    sentences = re.split(r"[!?\n]", paragraph)

    # Keywords that might indicate the presence of time information
    keywords = ["starts", "ends", "happens in", "start time", "end time", "start", "end", "happen"]
    # filter sentences by keywords
    candidates = []
    for sentence in sentences:
        # If sentence contains one of the keywords
        if any(keyword in sentence for keyword in keywords):
            candidates.append(sentence)
    print(f'{sentences=}')
    print(f'{candidates=}')
    timestamps = []
    # Check for The given query happens in m - n (seconds)
    patterns = [r"(\d+\.*\d*)\s*-\s*(\d+\.*\d*)"]

    for time_pattern in patterns:
        time_matches = re.findall(time_pattern, paragraph)
        if time_matches:
            timestamps = [[float(start), float(end)] for start, end in time_matches]

    if len(sentences) == 0:
        return -1, -1
    # check for other formats e.g.:
    # 1 .Starting time: 0.8 seconds
    # Ending time: 1.1 seconds
    # 2. The start time for this event is 0 seconds, and the end time is 12 seconds.
    if len(timestamps) == 0:
        times = []
        time_regex = re.compile(r"\b(\d+\.\d+\b|\b\d+)\b")  # time formats (e.g., 18, 18.5)
        for sentence in candidates:
            time = re.findall(time_regex, sentence)
            print(f'{time=}')
            if time:
                time_in_sec = float(time[0])
                times.append(time_in_sec)
        times = times[: len(times) // 2 * 2]
        timestamps = [(times[i], times[i + 1]) for i in range(0, len(times), 2)]
    # Check for  examples like:
    # 3. The event 'person flipped the light switch near the door' starts at 00:00:18 and ends at 00:00:23.
    if len(timestamps) == 0:
        times = []
        time_regex = re.compile(r"\b((\d{1,2}:\d{2}:\d{2}))\b")  # time formats (e.g., 18:00, 00:18:05)
        for sentence in candidates:
            time = re.findall(time_regex, sentence)
            if time:
                t = time[0]
            else:
                continue
            # If time is in HH:MM:SS format, convert to seconds
            if t.count(":") == 2:
                h, m, s = map(int, t.split(":"))
                time_in_sec = h * 3600 + m * 60 + s
            elif t.count(":") == 1:
                m, s = map(int, t.split(":"))
                time_in_sec = m * 60 + s
            times.append(time_in_sec)
        times = times[: len(times) // 2 * 2]
        timestamps = [(times[i], times[i + 1]) for i in range(0, len(times), 2)]
    results = []
    for start, end in timestamps:
        if end < start:
            start, end = end, start
        return start, end
    return -1, -1

def extract_time_range(paragraph: str) -> list:
    candidates = re.split(r"[!?\n]", paragraph)
    # 1. try get every pair in each line
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

    # 2. try get every pair in reverse order, in paragraph
    time = re.findall(time_regex, paragraph)
    new_times = []
    for tim in time or []:
        time_in_sec = float(tim)
        new_times.append(time_in_sec)
    for i in range(0, len(new_times), 2):
        stidx = len(new_times) - i - 2
        edidx = len(new_times) - i - 1
        if stidx >= 0:
            start = new_times[stidx]
            end = new_times[edidx]
            if start <= end:
                return start, end
    return -1, -1

def extract_zoom_calls(output_text):
    pattern = r'<zoom>(.*?)</zoom>'
    matches = re.findall(pattern, output_text, re.DOTALL)
    zooms = []
    for match in matches:
        try:
            zoom_args = json.loads(match)
            start = float(zoom_args["start"])
            end = float(zoom_args["end"])
            zooms.append((start, end))
        except Exception as e:
            print(f">>> [parse_output] Error:", e)
            continue  # 跳过错误项
    return zooms

def cal_iou_precision_recall(gt_list, pred_list):
    start1, end1 = eval(gt_list)
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

from eval_GQA_old import compute_iou

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
    for sample in pred_contents:
        pred_list = extract_time_range(sample['pred'])  # in <answer> tag
        iou = compute_iou([pred_list], sample['answer'])
        precision = 0.0
        recall = 0.0
        if iou == -2:
            print(f"Parsing failed: ans={sample['answer']}, pred={sample['pred']}, gt error, jump it")
            continue
        elif iou == -1:
            print(f"Parsing failed: ans={sample['answer']}, pred={sample['pred']}")
            parse_fail_cnt += 1
            iou = 0
            precision = 0
            recall = 0
        prediction_set[sample['id']] = {
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'recall@0.3': 1 if iou >= 0.3 else 0,
            'recall@0.5': 1 if iou >= 0.5 else 0,
            'recall@0.7': 1 if iou >= 0.7 else 0,
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
        
    if 'vidi' in gt_file:
        pred_path = os.path.join(output_path, 'pred_results.json')
        pred_results = []
        for qid, sample in prediction_set.items():
            pred_list = extract_time_range(sample['pred'])
            pred_results.append({
                'query_id': int(qid),
                # 'video_id': sample['video_id'],
                # 'duration': sample['duration'],
                'query': sample['question'],
                'answer': [pred_list],
                # 'task': 'temporal_retrieval'
            })
        with open(pred_path, "w") as f:
            json.dump(pred_results, f, indent=4)
        print("[main] All evaluation completed!")
        cmd = f'python data/vidi/vidi/qa_eval.py --pred_path {pred_path} --gt_path data/vidi/vidi/VUE-TR_ground_truth.json --output_dir {output_path}'
        os.system(cmd)

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
        meter_dic["precision"].add_score(result['precision'], 'yes')
        meter_dic["recall"].add_score(result['recall'], 'yes')

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
    parser.add_argument("--backend", type=str, default=None)
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

