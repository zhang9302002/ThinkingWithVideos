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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import json
import os
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length_return_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout

import re
import uuid
from verl.tools.base_tool import initialize_tools_from_config
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.debug.performance import _timer
from verl.models.transformers.qwen2_vl import get_rope_index

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    elif isinstance(value, np.ndarray):
        return np.repeat(value, repeats, axis=0)
    else:
        return [deepcopy(item) for item in value for _ in range(repeats)]


def parse_output(output_text):
    pattern = r'<tool_call>(.*?)</tool_call>'
    match = re.search(pattern, output_text, re.DOTALL)
    if match:
        try:
            tool_call_arguments = json.loads(match.group(1))
            assert "name" in tool_call_arguments, "'name' must be in tool_call_arguments"
            assert "arguments" in tool_call_arguments, "'arguments' must be in tool_call_arguments"
            assert isinstance(tool_call_arguments["name"], str), "'name' must be a string"
            assert isinstance(tool_call_arguments["arguments"], dict), "'arguments' must be a dictionary"
            return {
                "type": "tool",
                "content": tool_call_arguments
            }
        except Exception as e:
            return {
                "type": "error",
                "content": str(e),
            }
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, output_text, re.DOTALL)
    if match:
        return {
            "type": "answer",
            "content": match.group(1),
        }
    return {
        "type": "error",
        "content": "Error: response parse failed",
    }

class vLLMRolloutMultiTurnSync(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(model_hf_config.text_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        target_template = ["</{tag}>", " </{tag}>", "</{tag}>\n", " </{tag}>\n", "</{tag}>\n\n", " </{tag}>\n\n"] 
        target_tags = ["tool_call", "answer"]
        self.target_sequences = [template.format(tag=tag) for tag in target_tags for template in target_template ]

        self.single_turn_response_length = config.single_turn_response_length
        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=self.single_turn_response_length,
            stop=self.target_sequences,
            include_stop_str_in_output=True,  # Important
        )

        # # we may detokenize the result all together later
        # if vllm_version != "0.3.1":
        #     kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        logger.info(f"Initialized kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

        # Initialize tools from config file
        self.multi_turn_enable = config.multi_turn.enable
        self.max_turns = config.multi_turn.max_turns
        tool_config_path = config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        logger.info(f"Initialized tools: {self.tools}")
        logger.info(f"Initialized tool_schemas: {self.tool_schemas}")
        self.processor = hf_processor(model_path, trust_remote_code=True, use_fast=True)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def execute_tools(self, tool_call_arguments, tools_kwargs):
        tool_name = tool_call_arguments["name"]
        tool_arguments = tool_call_arguments["arguments"]
        try:
            assert tool_name in self.tools, f"Tool {tool_name} not found in tools list"
            tool = self.tools[tool_name]
            # this is a sync method
            tool_result = tool._execute(
                instance_id=str(uuid.uuid4()), 
                parameters=tool_arguments, 
                **tools_kwargs
            )
            tool_response, tool_reward_score, tool_metrics = tool_result
            return tool_response
        except Exception as e:
            return {
                "type": "error",
                "content": str(e),
            }


    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()
        
        generation_timing = {}
        with _timer("generation/preprocess", generation_timing):
            prompts_ids = prompts.batch["input_ids"]  # (bs, prompt_length)
            batch_size = prompts_ids.size(0)
            # left-padded attention_mask
            attention_mask = prompts.batch["attention_mask"]
            position_ids = prompts.batch["position_ids"]
            eos_token_id = prompts.meta_info["eos_token_id"]
            non_tensor_batch = prompts.non_tensor_batch
            vllm_inputs = []
            tools_kwargs = []
            for messages, multi_modal_data, mm_processor_kwargs, tools_kwarg in zip(non_tensor_batch.pop("raw_prompt"), non_tensor_batch.pop("multi_modal_data"), non_tensor_batch.pop("mm_processor_kwargs"), non_tensor_batch.pop("tools_kwargs")):
                messages = list(messages)  # here is another trap, has to be list() type
                vllm_inputs.append({"messages": messages, "multi_modal_data": multi_modal_data, "mm_processor_kwargs": mm_processor_kwargs})
                tools_kwarg.update({
                    "max_frames": 64,
                    "video": multi_modal_data["video"][0] if "video" in multi_modal_data else None,
                })
                tools_kwargs.append(tools_kwarg)
            multi_modal_inputs = []
            for mm_inputs in non_tensor_batch.pop("multi_modal_inputs"):
                multi_modal_inputs.append(dict(mm_inputs))
            extra_infos = []
            for info in non_tensor_batch.pop("extra_info"):
                extra_infos.append(dict(info))
            do_sample = prompts.meta_info.get("do_sample", True)
            is_validate = prompts.meta_info.get("validate", False)
            if not do_sample:
                kwargs = {
                    "best_of": 1,
                    "top_p": 1.0,
                    "top_k": -1,
                    "min_p": 0.0,
                    "temperature": 0,
                    "n": 1,  # if greedy, only 1 response
                }
            elif is_validate:
                # TODO: try **
                kwargs = {
                    "top_k": self.config.val_kwargs.top_k,
                    "top_p": self.config.val_kwargs.top_p,
                    "temperature": self.config.val_kwargs.temperature,
                    "n": 1,  # if validate, already repeat in ray_trainer
                }
            lora_requests = None
            if self.lora_kwargs:
                lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
                if len(lora_int_ids) > 0:
                    lora_int_id = lora_int_ids[0]
                    lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")] * batch_size
            repeat_times = self.sampling_params.n
            if repeat_times > 1 and do_sample:
                prompts_ids = _repeat_interleave(prompts_ids, repeat_times)
                attention_mask = _repeat_interleave(attention_mask, repeat_times)
                position_ids = _repeat_interleave(position_ids, repeat_times)
                batch_size = batch_size * repeat_times
                vllm_inputs = _repeat_interleave(vllm_inputs, repeat_times)
                tools_kwargs = _repeat_interleave(tools_kwargs, repeat_times)
                multi_modal_inputs = _repeat_interleave(multi_modal_inputs, repeat_times)
                extra_infos = _repeat_interleave(extra_infos, repeat_times)
            kwargs["n"] = 1

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            # manage repetitions, a, b, c, d => a, a, b, b, c, c, d, d
            # start multi-turn conversation
            active_mask = torch.ones(batch_size, dtype=torch.bool, device='cuda')  # if sample is active
            turns_stats = torch.ones(batch_size, dtype=torch.int, device='cuda')  # number of turns
            valid_action_stats = torch.zeros(batch_size, dtype=torch.int, device='cuda')  # number of valid actions
            valid_tool_call_stats = torch.zeros(batch_size, dtype=torch.int, device='cuda')  # number of valid tools
            valid_tool_exec_stats = torch.zeros(batch_size, dtype=torch.int, device='cuda')  # number of valid tools
            active_num_list = [active_mask.sum().item()]

            response_ids = [[] for _ in range(batch_size)]
            delta_position_ids = [[] for _ in range(batch_size)]
            st_indexs = [0 for _ in range(batch_size)]
            response_loss_masks = [[] for _ in range(batch_size)]
            rollout_log_probs = [[] for _ in range(batch_size)]

            for step in range(self.max_turns + 1):
                active_id_list = torch.where(active_mask)[0].tolist()
                active_num = len(active_id_list)
                active = active_num > 0
                if not active:
                    break

                with _timer("generation/generation", generation_timing):
                    active_vllm_inputs = []
                    for idx in active_id_list:
                        curr_prompts = self.processor.apply_chat_template(vllm_inputs[idx]["messages"], add_generation_prompt=True, tokenize=False)
                        active_vllm_inputs.append({
                            "prompt": curr_prompts,
                            "multi_modal_data": vllm_inputs[idx]["multi_modal_data"],
                            "mm_processor_kwargs": vllm_inputs[idx]["mm_processor_kwargs"],
                        })
                    outputs = self.inference_engine.generate(
                        prompts=active_vllm_inputs,
                        sampling_params=self.sampling_params,
                        lora_request=lora_requests,
                        use_tqdm=False,
                    )
                    outputs_flattened = [out for output in outputs for out in output.outputs]  # trick
                    assert active_num == len(outputs_flattened)

                for i, idx in enumerate(active_id_list):
                    with _timer("generation/preprocess_tool", generation_timing):
                        response_text = outputs_flattened[i].text
                        curr_response_ids = outputs_flattened[i].token_ids
                        curr_position_ids = torch.arange(0, len(curr_response_ids), device=position_ids.device).view(1, -1).expand(3, -1)
                        response_ids[idx].extend(curr_response_ids)
                        delta_position_ids[idx].append(curr_position_ids + st_indexs[idx])
                        st_indexs[idx] += len(curr_response_ids)
                        response_loss_masks[idx].extend([1] * len(curr_response_ids))
                        vllm_inputs[idx]["messages"].append({
                            "role": "assistant",
                            "content": response_text,
                        })
                        curr_log_prob = []
                        for j, logprob in enumerate(outputs_flattened[i].logprobs):
                            curr_log_prob.append(logprob[curr_response_ids[j]].logprob)
                        rollout_log_probs[idx].extend(curr_log_prob)
                    if step < self.max_turns and tools_kwargs[idx]['video_path'] is not None:
                        # preprocess tool call
                        res = parse_output(response_text)
                        if res["type"] == "tool":
                            tool_call_arguments = res["content"]
                            with _timer("generation/execute_tool", generation_timing):
                                tool_res = self.execute_tools(
                                    tool_call_arguments, {
                                        'llm': self.inference_engine,
                                        'sampling_params': self.sampling_params,
                                        "processor": self.processor,
                                        **tools_kwargs[idx],
                                    }
                                )
                            if tool_res["type"] == "result":
                                tool_result = tool_res["content"]
                                valid_tool_exec_stats[idx] += 1
                            elif tool_res["type"] == "error":
                                tool_result = tool_res["content"]
                            valid_action_stats[idx] += 1
                            valid_tool_call_stats[idx] += 1
                        elif res["type"] == "answer":
                            answer = res["content"]
                            tool_result = ""
                            active_mask[idx] = False
                            valid_action_stats[idx] += 1
                        elif res["type"] == "error":
                            error_msg = res["content"]
                            if error_msg == "Error: response parse failed":
                                tool_result = ""
                            else:
                                tool_result = error_msg

                        if tool_result != "":
                            with _timer("generation/postprocess_tool", generation_timing):
                                if isinstance(tool_result, str):  # text result, either caption, answer or error msg
                                    new_content = tool_result
                                    new_prompt_str = tool_result
                                    new_videos = None
                                    new_fps = None
                                elif isinstance(tool_result, dict):  # multimodal result, new video + text
                                    teaser = f"Video clip from {tool_result['start_time']:.2f} to {tool_result['end_time']:.2f} seconds."
                                    new_content = [
                                        tool_result["ele"],
                                        {
                                            "type": "text",
                                            "text": teaser
                                        }
                                    ]
                                    new_prompt_str = "<|vision_start|><|video_pad|><|vision_end|>" + teaser
                                    new_videos = [tool_result["video"]]
                                    new_fps = [tool_result["fps"]]
                                    vllm_inputs[idx]["multi_modal_data"]["video"].extend(new_videos)
                                    vllm_inputs[idx]["mm_processor_kwargs"]["fps"].extend(new_fps)
                                new_message = {
                                    "role": "tool",
                                    "content": new_content,
                                }
                                vllm_inputs[idx]["messages"].append(new_message)
                                new_prompt = "<|im_end|>\n"
                                new_prompt += f"<|im_start|>{new_message['role']}\n"
                                new_prompt += new_prompt_str
                                new_prompt += "<|im_end|>\n"
                                new_prompt += "<|im_start|>assistant\n"
                                new_inputs = self.processor(text=[new_prompt], images=None, videos=new_videos, fps=new_fps, return_tensors="pt")  # a list [1, L]
                                new_token_ids = list(new_inputs.input_ids[0])
                                new_position_ids = get_rope_index(
                                    self.processor,
                                    input_ids=new_inputs.input_ids[0],
                                    image_grid_thw=None,
                                    video_grid_thw=new_inputs.get("video_grid_thw"),
                                    second_per_grid_ts=new_inputs.get("second_per_grid_ts"),
                                    attention_mask=new_inputs.attention_mask[0],
                                ).to(prompts_ids.device)
                                response_ids[idx].extend(new_token_ids)
                                delta_position_ids[idx].append(new_position_ids + st_indexs[idx])
                                st_indexs[idx] += new_position_ids[:, -1].max() + 1  # a trap
                                response_loss_masks[idx].extend([0] * len(new_token_ids))
                                rollout_log_probs[idx].extend([-1] * len(new_token_ids))
                                if new_videos is not None:
                                    multi_modal_inputs[idx]["pixel_values_videos"] = torch.cat([multi_modal_inputs[idx]["pixel_values_videos"], new_inputs["pixel_values_videos"]], dim=0)
                                    multi_modal_inputs[idx]["video_grid_thw"] = torch.cat([multi_modal_inputs[idx]["video_grid_thw"], new_inputs["video_grid_thw"]], dim=0)
                                    multi_modal_inputs[idx]["second_per_grid_ts"] = torch.cat([multi_modal_inputs[idx]["second_per_grid_ts"], torch.tensor(new_inputs["second_per_grid_ts"])], dim=0)
                                    
                active_num_list.append(active_mask.sum().item())
                turns_stats[active_mask] += 1

            response_ids, response_attention_mask = pad_2d_list_to_length_return_mask(response_ids, self.pad_token_id, max_length=self.config.response_length, device=prompts_ids.device)
            response_loss_masks = pad_2d_list_to_length(response_loss_masks, 0, max_length=self.config.response_length).to(prompts_ids.device).to(torch.int64)
            rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.response_length).to(prompts_ids.device).to(torch.float32)

            # TODO response 3D rope, to check
            new_delta = torch.ones_like(response_ids).unsqueeze(1).repeat(1, 3, 1)  # [B, 3, L]
            for i, delta_id_list in enumerate(delta_position_ids):
                delta_id = torch.cat(delta_id_list, dim=1)
                mask = (response_attention_mask[i] == 1)
                new_delta[i, :, mask] = delta_id
            delta_position_id = new_delta

            sequence_ids = torch.cat([prompts_ids, response_ids], dim=-1)

        response_length = response_ids.size(1)
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        loss_mask = torch.cat((torch.zeros_like(attention_mask), response_loss_masks), dim=-1)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        sequence_lengths = attention_mask.sum(-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": prompts_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "rollout_log_probs": rollout_log_probs,  # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,  # [bsz, prompt_length + response_length]
                "active_mask": active_mask,
                "turns_stats": turns_stats,
                "valid_action_stats": valid_action_stats,
                "valid_tool_call_stats": valid_tool_call_stats,
                "valid_tool_exec_stats": valid_tool_exec_stats,
            },
            batch_size=batch_size,
        )
        
        non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs)
        for i, info in enumerate(extra_infos):
            info["reward_bonus"] = {
                "valid_tool_exec_stats": int(valid_tool_exec_stats[i]),
            }
        non_tensor_batch["extra_info"] = np.array(extra_infos)
        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={'generation_timing': generation_timing})

