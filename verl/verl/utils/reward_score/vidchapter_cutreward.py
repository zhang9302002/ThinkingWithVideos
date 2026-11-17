# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import re
import os
import json
import logging

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
logfile = os.getenv("LOGGER_OUTPUT_FILE", "charades.log").replace("logging", "logging_rewards")
print(f'In charades_cutreward.py, {logfile=}')
handler = logging.FileHandler(logfile, mode='a')
formatter = logging.Formatter("%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s", datefmt="%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False  # 关键设置

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

def cut_threshold_linear_transform(iou, low=0.0, high=0.5):
    new_iou = (iou - low) / (high - low)
    new_iou = max(0.0, new_iou)
    new_iou = min(1.0, new_iou)
    return new_iou

def cal_iou_reward(predict_str: str, ground_truth: str, extra_info: dict=None):
    if isinstance(ground_truth, str):
        ground_truth = eval(ground_truth)
    pred_list = extract_time_range(predict_str)
    iou, precision, recall = cal_iou_precision_recall(ground_truth, pred_list)
    iou = max(0, iou)  # neglect special cases
    iou = cut_threshold_linear_transform(iou)  # cut reward
    log_data = {
        "reward_name": "iou_reward_func",
        "custom_reward_func": iou,
    }
    return iou, log_data

def count_tag_pairs(s, tag):
    count = 0
    pos = 0
    # 构造正则，确保匹配指定标签
    pattern = re.compile(r'<{tag}>(.*?)</{tag}>'.format(tag=re.escape(tag)), re.DOTALL)
    while pos < len(s):
        match = pattern.search(s, pos)
        if not match:
            break
        count += 1
        pos = match.end()  # 跳到当前pair的结尾，防止重叠
    return count

def format_toolrwd_reward(content):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<(.*?)>(.*?)</\1>"
    matches = re.finditer(pattern, content, re.DOTALL)
    tag_list = []
    last_pos = None
    score = 1
    for match in matches:
        tag = match.group(1)  
        tag_content = match.group(2).strip()  
        start_pos = match.start()  
        end_pos = match.end() 
        if not tag in ["think", "tool_call", "answer"]:
            score = 0
            break
        elif tag_list and tag_list[-1] == tag:
            score = 0
            break
        elif last_pos and not content[last_pos[1]: start_pos].strip() == "":
            score = 0
            break
        tag_list.append(tag)
        last_pos = (start_pos, end_pos)
    else:
        if not tag_list:
            score = 0
        elif not tag_list[0] == 'think':
            score = 0
        elif not tag_list[-1] == 'answer':
            score = 0
        else:
            score = 1
    return score

def has_repeats_regex(s):
    return False
    n = len(s)
    max_pattern_len = n // 10
    for pattern_len in range(1, max_pattern_len + 1):
        # 正则表达式：捕获一个长度为pattern_len的模式，并检测它是否连续出现20次
        pattern = r'((.|\n){%d})\1{9}' % pattern_len
        if re.search(pattern, s):
            return True
    return False

def cal_format_reward(predict_str: str, ground_truth: str, extra_info: dict=None) -> float:
    tool_call_count = 0
    if extra_info and "reward_bonus" in extra_info:
        if "valid_tool_exec_stats" in extra_info["reward_bonus"]:
            tool_call_count += int(extra_info["reward_bonus"]["valid_tool_exec_stats"])
    pure_format = format_toolrwd_reward(predict_str)
    fmt = 0
    if pure_format > 0:
        fmt += 1.0
    if tool_call_count >= 1:
        fmt += min(tool_call_count, 2) * 0.5
    if has_repeats_regex(predict_str):
        fmt = -10
    log_data = {
        "format_reward_func": fmt,
        "pure_format": pure_format,
        "tool_call_count": tool_call_count,
    }
    return fmt, log_data

REWARD_MAP = {
    "iou": cal_iou_reward,
    "format": cal_format_reward,
}

def compute_score(predict_str: str, ground_truth: str, extra_info: dict=None) -> float:
    reward_list = ["iou"]
    if extra_info is not None and "reward_list" in extra_info:
        reward_list = extra_info["reward_list"]

    log_data = {
        "score": 0.0, 
        "response": str(predict_str),
        "ground_truth": str(ground_truth),
    }
    final_reward = 0.0
    for reward_name in reward_list:
        reward_fn = REWARD_MAP[reward_name]
        reward, log = reward_fn(predict_str, ground_truth, extra_info)
        final_reward += reward
        log_data.update(log)
    log_data["score"] = final_reward
    logger.info("Compute Rewards: %s", json.dumps(log_data, ensure_ascii=False, indent=4))
    return log_data


if __name__ == '__main__':
    s = """
    happens in 18.00 - 23.00 seconds.
    happens in 00:00:19 - 00:00:23 seconds.
    from 00:00:19 to 00:00:28 seconds.
    """
    # s = "The event runs from 12.5 - 15.75 and then from 18 - 20."
    s = "<think>The person is holding a picture and is clearly identifiable as a human figure. The person enters the room from the right and moves towards the center of the room towards the washing machine. The body posture and movement is evidence of someone holding a picture. The action lasts approximately 2 seconds from the start of the video.</think> \n<tool_call>\n{\"name\": \"temporal_zoom\", \"arguments\": {\"start_time\": 1.82, \"end_time\": 8.94}}\n</tool_call><answer>The person turns off the light in the room after walking in and standing nearby the washing machine.</answer>"
    # s = "The start time for this event is 0 seconds, and the end time is 12 seconds."
    # times = extract_time_range(s)
    # print(times)

    format_score = cal_format_reward(s, None)
    print(format_score)
    print(has_repeats_regex("abababababababababababababababababab"))  # -1
    print(has_repeats_regex("abcdefgh"))  # 0
    s = "<think>Okay, let's break this down. I'm tasked with pinpointing the exact start and end times of the \"Kingsman: Secret Service\" segment in this video compilation. My expertise lies in film analysis, so I'll meticulously examine the timestamps.\n\nFirst, I'll scan through these initial observations frame by frame.  At `175.68s`, I see a man in a mask, which is certainly promising. Then, at `179.01s`, BAM – *that's* it! A \"KINGSMAN\" title card. Confident. Time stamp confirms the scene, now showing people gathering around what appears to be a red and maroon ambulance. Classic Jason Kingdom. I'll note those start and end times: start around `175.68s` and ending around `203.44s`.\n\nNext, key window. The woman combing her hair at `207.78s`.\n\nThen, some watery bits where he might be present, which is a nice complement. Finally, the scene *on* the train at `235.24s`.\n\nJust double-checking: It’s definitely \"Kingsman: Secret Service.\"\n\nTherefore, the start is clearly around `175.86s` with the masked scene, and the wrap-up seems to occur just before he actually *leaves*. Thus, a good endpoint would be approximately `200.00s`.\n 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成 自动生成"
    print(has_repeats_regex(s))