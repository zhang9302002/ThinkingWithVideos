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

import logging
import os

import torch


def set_basic_config(level):
    """
    This function sets the global logging format and level. It will be called when import verl
    """
    # logging.basicConfig(format="%(levelname)s:%(asctime)s:%(message)s", level=level)

    print(f'In logging_utils, {os.getenv("VERL_LOGGING_LEVEL")=}')
    print(f'In logging_utils, {os.getenv("LOGGER_OUTPUT_FILE")=}')
    print(f'In logging_utils, {os.getenv("WANDB_DIR")=}')

    logging.basicConfig(
        format="%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s",    
        datefmt="%m-%d %H:%M:%S",
        level=os.getenv("VERL_LOGGING_LEVEL", "WARNING"),
        filename=os.getenv("LOGGER_OUTPUT_FILE", "my_log_file.log"),  # 指定日志文件名
        filemode="a"  # 追加模式，默认也是"a"，可以不写
    )



def log_to_file(string):
    print(string)
    if os.path.isdir("logs"):
        with open(f"logs/log_{torch.distributed.get_rank()}", "a+") as f:
            f.write(string + "\n")
