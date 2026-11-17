set -x

export DO_TRAIN=0
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

# training
export nnodes=1
export n_gpus_per_node=8
export batch_size=256
export micro_batch_size_per_gpu=2
# model arch
export model_path=STAGE_2_PRETRAINED_CKPT
export max_turns=0
export tool_config_path=verl/verl/tools/config/zoom_tool_config_new.yaml
# data
export dataset=data_mtvr_cot_tool
export dataset_json_path=[data/vidchapters/train_set_sft_diff_exist_tool1_11k.json,data/longvideo-reason/train_set_sft_diff_exist_tool1_7k.json]
export dataset_json_path_val=[data/vidchapters/train_set_sft_diff_exist_tool1_11k_val.json,data/longvideo-reason/train_set_sft_diff_exist_tool1_7k_val.json]
export dataset_video_base=[data/vidchapters/video_14400frames_fps2,data/longvideo-reason/train_video_14400frames_fps2]
export max_prompt_length=10240
# name
export project_name=sft_then_grpo_then_sft
export exp_suffix=thinking_tool_lr1e_6

# auto config
export EXP_NAME=qwen2_5_vl_7b_${exp_suffix}_${n_gpus_per_node}g_sft_${dataset}_bs${batch_size}
export SAVE_PATH=outputs/${project_name}/${EXP_NAME}

export WANDB_API_KEY=YOUR_WANDB_API_KEY
export WANDB_MODE=offline
export WANDB_DIR=${SAVE_PATH}
export WANDB_CONFIG_DIR=${SAVE_PATH}
export DATE="$(TZ='Asia/Shanghai' date +%m%d_%H%M%S)"
# print, logging, and debug
export HYDRA_FULL_ERROR=1 
export PYTHONUNBUFFERED=1 
export VERL_LOGGING_LEVEL=DEBUG
export CONSOLE_OUTPUT_FILE=${SAVE_PATH}/${DATE}_verl_training.log
export LOGGER_OUTPUT_FILE=${SAVE_PATH}/${DATE}_verl_logging.log


if [ $DO_TRAIN -eq 1 ]; then
     mkdir -p $SAVE_PATH
     touch $CONSOLE_OUTPUT_FILE $LOGGER_OUTPUT_FILE
     chown -R tiger $SAVE_PATH 

     echo "start sft, write to ${SAVE_PATH}"
     echo "MY_PROMPT_TEMPLATE=$MY_PROMPT_TEMPLATE"

     torchrun \
            --nproc_per_node=$n_gpus_per_node \
            --standalone \
            -m verl.trainer.fsdp_sft_trainer \
            optim.lr=1e-5 \
            +data.max_prompt_length=$max_prompt_length \
            data.train_files=$dataset_json_path \
            data.val_files=$dataset_json_path_val \
            data.train_batch_size=$batch_size \
            data.micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
            data.prompt_key=prompt \
            +data.image_key=images \
            +data.video_key=videos \
            data.custom_cls.path=verl/verl/utils/dataset/sft_dataset_multi_turn.py \
            data.custom_cls.name=SFTDatasetMultiTurn \
            +data.user_prompt_template=$user_prompt_template \
            +data.response_dict_keys=['answer'] \
            +data.multi_turn.max_turns=$max_turns \
            +data.multi_turn.tool_config_path=$tool_config_path \
            +data.multi_turn.video_base=$dataset_video_base \
            +data.multi_turn.video_kwargs.draw_number=true \
            +data.multi_turn.video_kwargs.parallel=true \
            +data.multi_turn.video_kwargs.max_frames=128 \
            model.partial_pretrain=$model_path \
            model.use_liger=True \
            trainer.default_local_dir=$SAVE_PATH \
            trainer.project_name=$project_name \
            trainer.experiment_name=$EXP_NAME \
            trainer.logger=['console','wandb'] \
            trainer.save_freq=50 \
            trainer.test_freq=0 \
            trainer.total_epochs=1 \
            trainer.total_training_steps=null \
            use_remove_padding=true \
            >> $CONSOLE_OUTPUT_FILE 2>&1 

fi


if [ $DO_EVAL -eq 1 ]; then
    export VLLM_USE_V1=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export n_gpus_per_node=8

    max_pixels_expr='384*384'
    max_pixels=$((${max_pixels_expr}))
    max_frames=256
    for dataset in charades_sta_src actnet_tg_src
    do
        for step in 50 100
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
        for step in 50 100
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
        for step in 50 100
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
        for step in 50 100
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
        for step in 50 100
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
        for step in 50 100
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
