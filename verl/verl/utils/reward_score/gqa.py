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
from rouge_score import rouge_scorer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
logfile = os.getenv("LOGGER_OUTPUT_FILE", "gqa.log").replace("logging", "logging_rewards")
print(f'In gqa.py, {logfile=}')
handler = logging.FileHandler(logfile, mode='a')
formatter = logging.Formatter("%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s", datefmt="%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False  # 关键设置

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

def cal_open_reward(predict_str: str, ground_truth: str, extra_info: dict=None) -> float:
    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure
    pred_ans = extract_answer(predict_str)
    score = compute_rouge_score(ground_truth, pred_ans)
    reward = max(0.0, min(1.0, score))
    log_data = {
        "reward_name": "open_rouge_reward_func",
        "custom_reward_func": reward,
    }
    return reward, log_data

from .charades import cal_iou_precision_recall, extract_time_range
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

def cal_iou_free_form(predict_str: str, ground_truth: str, extra_info: dict=None):
    iou, iou_log = cal_iou_reward(predict_str, ground_truth, extra_info)
    free, free_log = cal_open_reward(predict_str, ground_truth, extra_info)
    reward = (iou + free)
    log_data = {
        "reward_name": "iou_free_form_reward_func",
        "custom_reward_func": reward,
    }
    return iou, log_data

def cal_iou_multiple_choice(predict_str: str, ground_truth: str, extra_info: dict=None):
    iou, iou_log = cal_iou_reward(predict_str, ground_truth, extra_info)
    mcq, mcq_log = cal_mcq_reward(predict_str, ground_truth, extra_info)
    reward = (iou + mcq)
    log_data = {
        "reward_name": "iou_multiple_choice_reward_func",
        "custom_reward_func": reward,
    }
    return iou, log_data

REWARD_MAP = {
    "format": cal_format_reward,
    "iou": cal_iou_reward,
    'multiple_choice': cal_mcq_reward,
    'free_form': cal_open_reward,
    'iou_free_form': cal_iou_free_form,
    'iou_multiple_choice': cal_iou_multiple_choice,
}

def compute_score(predict_str: str, ground_truth: str, extra_info: dict=None, problem_type: str=None) -> float:
    reward_list = ["format"]
    DATASET_MAP = {
        "nextgqa": "iou_multiple_choice",
        "rextime": "iou_free_form",
    }
    if problem_type in DATASET_MAP:
        dataset_reward_list = DATASET_MAP[problem_type]
        reward_list.append(dataset_reward_list)

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