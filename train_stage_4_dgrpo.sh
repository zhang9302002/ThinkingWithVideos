set -x

export DO_TRAIN=1
export DO_EVAL=1

export user_prompt_template=THINK_GENERAL_TOOL
export MY_PROMPT_TEMPLATE="This is a video with duration {duration} seconds.
You should first think the user's question step-by-step and then provides the user with the answer. 
Note that there must be an answer for each question and you must answer it.
# Instruction
1. Output your thought process within the <think> </think> tags, and output your answer within the <answer> </answer> tags.
2. You can call the provided tools ONCE to get more visual information within the <tool_call> </tool_call> tags. 
3. When you get the tool result, you need to integrate your initial reasoning with the new visual evidence from the tool, think step-by-step again and provide the final answer. 
# Output Format
<think> ... </think> <tool_call> ... </tool_call> <think> ... </think> <answer> ... </answer>
User Question: 
{input_text}"


export ENGINE=vllm
export ENGINE_MODE=multi_turn_sync
if [ "$ENGINE_MODE" != "sync" ]; then
    export VLLM_USE_V1=1
    export return_raw_chat=True
else
    export return_raw_chat=False
fi
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

# model arch
export model_path=STAGE_3_PRETRAINED_CKPT
export max_turns=2
export tool_config_path=verl/verl/tools/config/zoom_tool_config_new.yaml
# training
export n_gpus_per_node=8
export nnodes=2
export group_size=8
export rollout_batch_size=64
export update_batch_size=64  # one step per episode
export ppo_micro_batch_size_per_device=2  # divisor of group_size * update_batch_size / n_gpus_per_node
export prob_ref_micro_batch_size_per_device=4  # divisor of group_size * update_batch_size / n_gpus_per_node
# reward
export reward_list=[format,iou] ########### set to only iou reward
# data
export dataset=data_mtvr_cot_tool_rl
export dataset_json_path=[data/vidchapters/train_set_grpo_cutsrc_diff_exist_7k4_len256.json,data/longvideo-reason/train_set_grpo_src_exist_8k_len256.json]
export dataset_video_base=[data/vidchapters/video_14400frames_fps2,data/longvideo-reason/train_video_14400frames_fps2]
export max_prompt_length=3072
export max_response_length=7168
export single_turn_response_length=1024
# name
export project_name=sft_then_grpo_then_sft_then_grpo
export exp_suffix=thinking_tool_lr1e_6

# auto config
export EXP_NAME=qwen2_5_vl_7b_${exp_suffix}_${n_gpus_per_node}g_grpo_${dataset}_gs${group_size}_bs${rollout_batch_size}
export SAVE_PATH=outputs/${project_name}/${EXP_NAME}
export WANDB_API_KEY=YOUR_WANDB_API_KEY
export WANDB_MODE=offline
export WANDB_DIR=${SAVE_PATH}
export WANDB_CONFIG_DIR=${SAVE_PATH}
export DATE="$(TZ='Asia/Shanghai' date +%m%d_%H%M%S)"
# print, logging, and debug
export HYDRA_FULL_ERROR=1 
export PYTHONUNBUFFERED=1 
export VERL_LOGGING_LEVEL=INFO
export CONSOLE_OUTPUT_FILE=${SAVE_PATH}/${DATE}_verl_training.log
export LOGGER_OUTPUT_FILE=${SAVE_PATH}/${DATE}_verl_logging.log



# python verl/scripts/model_merger.py merge \
#     --backend fsdp \
#     --local_dir $CONTINUE_MODEL \
#     --target_dir $CONTINUE_MODEL_HF


# ray start --head --dashboard-host=0.0.0.0

if [ $DO_TRAIN -eq 1 ]; then
    mkdir -p $SAVE_PATH
    touch $CONSOLE_OUTPUT_FILE $LOGGER_OUTPUT_FILE
    chown -R tiger $SAVE_PATH 

    echo "start rl=grpo, write to ${SAVE_PATH}"
    echo "MY_PROMPT_TEMPLATE=$MY_PROMPT_TEMPLATE"

    export RAY_DEDUP_LOGS=0

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=dgrpo \
        data.train_files=$dataset_json_path \
        data.val_files=$dataset_json_path \
        data.train_batch_size=$rollout_batch_size \
        data.val_batch_size=$rollout_batch_size \
        data.max_prompt_length=$max_prompt_length \
        data.max_response_length=$max_response_length \
        data.return_raw_chat=$return_raw_chat \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.image_key=images \
        data.dataloader_num_workers=16 \
        data.custom_cls.path=verl/verl/utils/dataset/rl_dataset_multi_turn.py \
        data.custom_cls.name=RLHFDatasetMultiTurn \
        data.sampler.class_path=verl/verl/utils/dataset/rl_sampler_multimodal.py \
        data.sampler.class_name=MultimodalSampler \
        +data.user_prompt_template=$user_prompt_template \
        +data.multi_turn.max_turns=$max_turns \
        +data.multi_turn.tool_config_path=$tool_config_path \
        +data.multi_turn.video_base=$dataset_video_base \
        +data.multi_turn.reward_list=$reward_list \
        +data.multi_turn.video_kwargs.draw_number=true \
        +data.multi_turn.video_kwargs.parallel=true \
        +data.multi_turn.video_kwargs.max_frames=64 \
        reward_model.reward_manager=naive_multiturn \
        actor_rollout_ref.model.path=${model_path} \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.freeze_vision_tower=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=$update_batch_size \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_device \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$prob_ref_micro_batch_size_per_device \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=$ENGINE \
        actor_rollout_ref.rollout.mode=$ENGINE_MODE \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((($max_prompt_length + $max_response_length)*2)) \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.disable_log_stats=False \
        actor_rollout_ref.rollout.n=$group_size \
        +actor_rollout_ref.rollout.single_turn_response_length=$single_turn_response_length \
        actor_rollout_ref.rollout.multi_turn.enable=True \
        actor_rollout_ref.rollout.multi_turn.max_turns=$max_turns \
        actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
        +actor_rollout_ref.rollout.repetition_penalty=1.05 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$prob_ref_micro_batch_size_per_device \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name=$project_name \
        trainer.experiment_name=$EXP_NAME \
        trainer.n_gpus_per_node=$n_gpus_per_node \
        trainer.nnodes=$nnodes \
        trainer.save_freq=25 \
        trainer.test_freq=0 \
        trainer.total_epochs=1 \
        trainer.total_training_steps=null \
        trainer.default_local_dir=$SAVE_PATH \
        trainer.rollout_data_dir=$SAVE_PATH/rollouts \
        trainer.validation_data_dir=$SAVE_PATH/evaluations \
        trainer.val_before_train=False \
        trainer.balance_batch=True \
        hydra.run.dir=$SAVE_PATH \
        >> $CONSOLE_OUTPUT_FILE 2>&1 
fi


if [ $DO_EVAL -eq 1 ]; then
    export VLLM_USE_V1=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    
    max_pixels_expr='384*384'
    max_pixels=$((${max_pixels_expr}))
    max_frames=256
    for dataset in charades_sta_src actnet_tg_src
    do
        for step in 100 200 250
        do
            echo start eval ${dataset} at step ${step}
            suffix=step${step}
            python eval_temporal_grounding.py \
                --inference_file inference_vllm_multiturn_number.py \
                --model-path ${SAVE_PATH}/global_step_${step} \
                --dataset ${dataset} \
                --output_dir ${SAVE_PATH}/global_step_${step} \
                --evaluation_name evaluation_maxpix${max_pixels_expr}_number \
                --num_chunks ${n_gpus_per_node} \
                --max_pixels ${max_pixels} \
                --prompt_type TG_TRAIN_TEMPLATE \
                --no_cache \
                >> "${SAVE_PATH}/${DATE}_eval_qwen2vl7b_${dataset}_${suffix}.log" 2>&1 
        done
    done
    for dataset in next_gqa rextime_val
    do
        for step in 100 200 250
        do
            echo start eval ${dataset} at step ${step}
            suffix=step${step}
            python eval_GQA.py \
                --inference_file inference_vllm_multiturn_number.py \
                --model-path ${SAVE_PATH}/global_step_${step} \
                --dataset ${dataset} \
                --output_dir ${SAVE_PATH}/global_step_${step} \
                --evaluation_name evaluation_maxpix${max_pixels_expr}_number \
                --num_chunks ${n_gpus_per_node} \
                --max_pixels ${max_pixels} \
                --prompt_type TG_TRAIN_TEMPLATE \
                --no_cache \
                >> "${SAVE_PATH}/${DATE}_eval_qwen2vl7b_${dataset}_${suffix}.log" 2>&1 
        done
    done
    for dataset in mmvu videommmu vsibench
    do
        for step in 100 200 250
        do
            echo start eval ${dataset} at step ${step}
            suffix=step${step}
            python eval_VQA.py \
                --inference_file inference_vllm_multiturn_number.py \
                --model-path ${SAVE_PATH}/global_step_${step} \
                --dataset ${dataset} \
                --output_dir ${SAVE_PATH}/global_step_${step} \
                --evaluation_name evaluation_maxpix${max_pixels_expr}_maxfrm${max_frames}_number \
                --num_chunks ${n_gpus_per_node} \
                --max_pixels ${max_pixels} \
                --max_frames ${max_frames} \
                --prompt_type TG_TRAIN_TEMPLATE \
                --no_cache \
                >> "${SAVE_PATH}/${DATE}_eval_qwen2vl7b_${dataset}_${suffix}.log" 2>&1 
        done
    done


    max_pixels_expr='224*224'
    max_pixels=$((${max_pixels_expr}))
    max_frames=1024
    for dataset in vidchapter_src
    do
        for step in 100 200 250
        do
            echo start eval ${dataset} at step ${step}
            suffix=step${step}
            python eval_temporal_grounding.py \
                --inference_file inference_vllm_multiturn_number.py \
                --model-path ${SAVE_PATH}/global_step_${step} \
                --dataset ${dataset} \
                --output_dir ${SAVE_PATH}/global_step_${step} \
                --evaluation_name evaluation_maxpix${max_pixels_expr}_maxfrm${max_frames}_number \
                --num_chunks ${n_gpus_per_node} \
                --max_pixels ${max_pixels} \
                --max_frames ${max_frames} \
                --prompt_type TG_TRAIN_TEMPLATE \
                --no_cache \
                >> "${SAVE_PATH}/${DATE}_eval_qwen2vl7b_${dataset}_${suffix}.log" 2>&1 
        done
    done
    for dataset in vidi_src
    do
        for step in 100 200 250
        do
            echo start eval ${dataset} at step ${step}
            suffix=step${step}
            python eval_temporal_grounding_multi.py \
                --inference_file inference_vllm_multiturn_number.py \
                --model-path ${SAVE_PATH}/global_step_${step}/Qwen2.5-VL-7B-Instruct \
                --dataset ${dataset} \
                --output_dir ${SAVE_PATH}/global_step_${step} \
                --evaluation_name evaluation_maxpix${max_pixels_expr}_maxfrm${max_frames}_number \
                --num_chunks ${n_gpus_per_node} \
                --max_pixels ${max_pixels} \
                --max_frames ${max_frames} \
                --prompt_type TG_TRAIN_TEMPLATE \
                --no_cache \
                >> "${SAVE_PATH}/${DATE}_eval_qwen2vl7b_${dataset}_${suffix}.log" 2>&1 
        done
    done
    for dataset in longvideo-reason videomme
    do
        for step in 100 200 250
        do
            echo start eval ${dataset} 
            suffix=step${step}
            python eval_VQA.py \
                --inference_file inference_vllm_multiturn_number.py \
                --model-path ${SAVE_PATH}/global_step_${step} \
                --dataset ${dataset} \
                --output_dir ${SAVE_PATH}/global_step_${step} \
                --evaluation_name evaluation_maxpix${max_pixels_expr}_maxfrm${max_frames}_number \
                --num_chunks ${n_gpus_per_node} \
                --max_pixels ${max_pixels} \
                --max_frames ${max_frames} \
                --prompt_type TG_TRAIN_TEMPLATE \
                --no_cache \
                >> "${SAVE_PATH}/${DATE}_eval_qwen2vl7b_${dataset}_${suffix}.log" 2>&1 
        done
    done

fi



