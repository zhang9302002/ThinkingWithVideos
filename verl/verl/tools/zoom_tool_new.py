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

from verl.utils.dataset.video_vl_utils import fetch_video

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class GetVideoClipFrameTool(BaseTool):
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

        logger.info(f"Initialized GetVideoClipFrameTool with config: {config}")

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

    def _execute(self, instance_id: str, parameters: dict[str, Any], video_path: str, duration: float, **kwargs) -> Tuple[str, float, dict]:
        """Execute the search tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query_list and optional timeout

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
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
            
            video_frame_path = video_path.split('.')[0]
            fps = 2.0
            if os.path.exists(video_frame_path):
                frame_paths = os.listdir(video_frame_path)
                frame_paths = sorted(frame_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                frame_paths = [os.path.join(video_frame_path, frame_path) for frame_path in frame_paths]
                total_frames = len(frame_paths)
                video_path = frame_paths
            else:
                raise ValueError(f"[error] video frame path {video_frame_path} not exists")
            ele = {
                "type": "video",
                "video": video_path,
                "max_frames": kwargs.get("max_frames", 64),
                "min_pixels": kwargs.get("min_pixels", 4 * 28 * 28),
                "max_pixels": kwargs.get("max_pixels", 64 * 28 * 28),
                "video_start": start_time,
                "video_end": end_time,
                "draw_number": kwargs.get("draw_number", False),
                "parallel": kwargs.get("parallel", True),
                "fps": kwargs.get("fps", 2),
            }
            video, fps = fetch_video(ele, return_video_sample_fps=True)
            return_content = {
                "ele": ele,
                "video": video,
                "fps": fps,
                "start_time": start_time,
                "end_time": end_time,
            }
            return {"type": "result", "content": return_content}, 0.0, metrics


        except Exception as e:
            logger.error(f"[ZoomTool] Execution failed: {e}, Received parameters: {parameters}")
            return {"type": "error", "content": f"Error: {self.name} tool execution failed: {e}"}, 0.0, {"error": str(e)}


    async def execute(self, instance_id: str, parameters: dict[str, Any], video_path: str, duration: float, **kwargs) -> Tuple[str, float, dict]:
        """Execute the search tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query_list and optional timeout

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        result = self._execute(instance_id, parameters, video_path, duration, **kwargs)
        return result
    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class GetVideoClipCaptionTool(GetVideoClipFrameTool):
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        logger.info(f"Initialized GetVideoClipCaptionTool with config: {config}")

    def _execute(self, instance_id: str, parameters: dict[str, Any], video_path: str, duration: float, **kwargs) -> Tuple[str, float, dict]:    
        try:
            start_time = parameters.get("start_time")
            end_time = parameters.get("end_time")
            start_time = float(start_time)
            end_time = float(end_time)
            assert start_time < end_time, f"[error] start: {start_time} should be less than end: {end_time}"
            assert start_time >= 0, f"[error] start: {start_time} should be greater than 0"
            assert end_time <= duration, f"[error] end: {end_time} should be less than video duration: {duration}"
            assert end_time - start_time > 1.0, f"[error] end - start: {end_time} - {start_time} should be greater than 1.0"
            video_frame_path = video_path.split('.')[0]
            fps = 2.0
            if os.path.exists(video_frame_path):
                frame_paths = os.listdir(video_frame_path)
                frame_paths = sorted(frame_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                frame_paths = [os.path.join(video_frame_path, frame_path) for frame_path in frame_paths]
                total_frames = len(frame_paths)
                video_path = frame_paths
            else:
                raise ValueError(f"[error] video frame path {video_frame_path} not exists")
            ele = {
                "type": "video",
                "video": video_path,
                "max_frames": kwargs.get("max_frames", 64),
                "min_pixels": kwargs.get("min_pixels", 4 * 28 * 28),
                "max_pixels": kwargs.get("max_pixels", 64 * 28 * 28),
                "video_start": start_time,
                "video_end": end_time,
                "draw_number": kwargs.get("draw_number", False),
                "parallel": kwargs.get("parallel", True),
                "fps": kwargs.get("fps", 2),
            }
            video, fps = fetch_video(ele, return_video_sample_fps=True)
            llm = kwargs.get("llm")
            sampling_params = kwargs.get("sampling_params")
            processor = kwargs.get("processor")
            text = "Please provide an objective and detailed caption for this video segment, focusing on accurately describing the visible subjects and their actions, the scene, and any changes, strictly based on the visual content. Do not make assumptions or add information not clearly present in the video."
            messages = [{
                "role": "user",
                "content": [
                    ele,
                    {"type": "text", "text": text},
                ],
            }]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            vllm_input = {
                "prompt": prompt,
                "multi_modal_data": {'video': video},
                "mm_processor_kwargs": {'fps': fps},
            }
            outputs = llm.generate([vllm_input], sampling_params=sampling_params, use_tqdm=False)
            output = outputs[0].outputs[0].text
            return {"type": "result", "content": output}, 0.0, {"success": True}

        except Exception as e:
            logger.error(f"[CaptionTool] Execution failed: {e}, Received parameters: {parameters}")
            return {"type": "error", "content": f"Error: {self.name} tool execution failed: {e}"}, 0.0, {"error": str(e)}



class AskVideoClipQuestionTool(GetVideoClipFrameTool):
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        logger.info(f"Initialized AskVideoClipQuestionTool with config: {config}")

    def _execute(self, instance_id: str, parameters: dict[str, Any], video_path: str, duration: float, **kwargs) -> Tuple[str, float, dict]:    
        try:
            start_time = parameters.get("start_time")
            end_time = parameters.get("end_time")
            start_time = float(start_time)
            end_time = float(end_time)
            assert start_time < end_time, f"[error] start: {start_time} should be less than end: {end_time}"
            assert start_time >= 0, f"[error] start: {start_time} should be greater than 0"
            assert end_time <= duration, f"[error] end: {end_time} should be less than video duration: {duration}"
            assert end_time - start_time > 1.0, f"[error] end - start: {end_time} - {start_time} should be greater than 1.0"
            video_frame_path = video_path.split('.')[0]
            fps = 2.0
            if os.path.exists(video_frame_path):
                frame_paths = os.listdir(video_frame_path)
                frame_paths = sorted(frame_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                frame_paths = [os.path.join(video_frame_path, frame_path) for frame_path in frame_paths]
                total_frames = len(frame_paths)
                video_path = frame_paths
            else:
                raise ValueError(f"[error] video frame path {video_frame_path} not exists")
            ele = {
                "type": "video",
                "video": video_path,
                "max_frames": kwargs.get("max_frames", 64),
                "min_pixels": kwargs.get("min_pixels", 4 * 28 * 28),
                "max_pixels": kwargs.get("max_pixels", 64 * 28 * 28),
                "video_start": start_time,
                "video_end": end_time,
                "draw_number": kwargs.get("draw_number", False),
                "parallel": kwargs.get("parallel", True),
                "fps": kwargs.get("fps", 2),
            }
            video, fps = fetch_video(ele, return_video_sample_fps=True)
            llm = kwargs.get("llm")
            processor = kwargs.get("processor")
            sampling_params = kwargs.get("sampling_params")
            question = parameters.get("question")
            text = f"Please answer the following question based only on the visual content of this video segment: {question} Describe only what is clearly visible in the frames. Do not make assumptions, guesses, or add any information that is not directly shown in the video."

            messages = [{
                "role": "user",
                "content": [
                    ele,
                    {"type": "text", "text": text},
                ],
            }]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            vllm_input = {
                "prompt": prompt,
                "multi_modal_data": {'video': video},
                "mm_processor_kwargs": {'fps': fps},
            }
            outputs = llm.generate([vllm_input], sampling_params=sampling_params, use_tqdm=False)
            output = outputs[0].outputs[0].text
            return {"type": "result", "content": output}, 0.0, {"success": True}

        except Exception as e:
            logger.error(f"[QuestionTool] Execution failed: {e}, Received parameters: {parameters}")
            return {"type": "error", "content": f"Error: {self.name} tool execution failed: {e}"}, 0.0, {"error": str(e)}

