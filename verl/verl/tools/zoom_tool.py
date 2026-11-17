# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

import json
import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, Tuple, TypeVar
from uuid import uuid4

import ray
import ray.actor

from verl.tools.utils.search_r1_like_utils import perform_single_search_batch

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

from qwen_vl_utils import fetch_image, fetch_video
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ZoomTool(BaseTool):
    """Zoom tool for video temporal zoom-in

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the search tool
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize ZoomTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
            TODO
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        logger.info(f"Initialized ZoomTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": 0.0,
        }
        return instance_id
    
    def _execute(self, instance_id: str, parameters: dict[str, Any], video_path: str, duration: float, video: np.ndarray, **kwargs) -> Tuple[str, float, dict]:
        try:
            metrics = {}
            start_time = parameters.get("start_time")
            end_time = parameters.get("end_time")
            start_time = float(start_time)
            end_time = float(end_time)

            assert start_time < end_time, f"[error] start: {start_time} should be less than end: {end_time}"
            assert start_time >= 0, f"[error] start: {start_time} should be greater than 0"
            assert end_time <= duration, f"[error] end: {end_time} should be less than video duration: {duration}"
            assert end_time - start_time > 1.0, f"[error] end - start: {end_time} - {start_time} should be greater than 1.0"
            ele = {
                "type": "video",
                "video": video_path,
                "max_frames": kwargs.get("max_frames", 64),
                "min_pixels": kwargs.get("min_pixels", 4 * 28 * 28),
                "max_pixels": kwargs.get("max_pixels", 64 * 28 * 28),
                "video_start": start_time,
                "video_end": end_time,
            }
            # video, fps = fetch_video(ele, return_video_sample_fps=True)
            assert video.shape[1] == 3, f"[error] video shape: {video.shape}"
            T = video.shape[0]
            start_pos = int(start_time * T / duration)
            end_pos = int(end_time * T / duration)
            video = video[start_pos:end_pos, :, :, :]
            return_content = {
                "ele": ele,
                "video": video,
                "start_time": start_time,
                "end_time": end_time,
            }
            return {"type": "result", "content": return_content}, 0.0, metrics
        except Exception as e:
            logger.error(f"[ZoomTool] Execution failed: {e}, Received parameters: {parameters}")
            return {"type": "error", "content": f"Error: {self.name} tool execution failed: {e}"}, 0.0, {"error": str(e)}


    async def execute(self, instance_id: str, parameters: dict[str, Any], video_path: str, duration: float, video: np.ndarray, **kwargs) -> Tuple[str, float, dict]:
        """Execute the search tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query_list and optional timeout

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        result = self._execute(self, instance_id, parameters, video_path, duration, video, **kwargs)
        return result

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
