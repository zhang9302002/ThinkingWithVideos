# Copyright 2025 Amazon.com Inc and/or its affiliates
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
"""
test create_rl_sampler
"""

from collections.abc import Sized

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, RandomSampler

from verl.trainer.main_ppo import create_rl_sampler
from verl.utils.dataset.sampler import AbstractCurriculumSampler


class MultimodalSampler(AbstractCurriculumSampler):
    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
    ):
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(1)
        self.data_source = data_source
        self.generator = train_dataloader_generator
        self.T = 2

    def __iter__(self):
        n = len(self.data_source) // self.T
        perm = torch.randperm(n, generator=self.generator).tolist()
        for idx in perm:
            for i in range(self.T):
                yield idx*self.T + i

    def __len__(self) -> int:
        return len(self.data_source) // self.T * self.T

    def update(self, batch) -> None:
        return

