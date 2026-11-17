set -x

export DO_TRAIN=1
export DO_EVAL=1

export user_prompt_template=THINK_GENERAL
export MY_PROMPT_TEMPLATE="This is a video with duration {duration} seconds.
You should first think the user's question step-by-step and then provides the user with the answer. 
Note that there must be an answer for each question and you must answer it.
Output your thought process within the <think> </think> tags, and output your answer within the <answer> </answer> tags,
i.e., <think> ... </think><answer> ... </answer>.
User Question: 
{input_text}"

# training
export nnodes=1
export n_gpus_per_node=8
export batch_size=256
export micro_batch_size_per_gpu=4
# data
export dataset=data_mtvr_cot
export dataset_json_path=[data/charades/train_set_sft_new_10k6_src_diff.json,data/actnet/train_set_sft_new_9k5_src_diff.json,data/vidchapters/train_set_sft_diff_exist_4k3.json,data/nextgqa/train_set_sft_src_diff_3k.json,data/rextime/train_set_sft_exist_src_diff_3k7.json,data/Video-R1-data/train_set_sft_005_video.json,data/Video-R1-data/train_set_sft_005_image.json,data/longvideo-reason/train_set_sft_src_exist_8k.json]
export dataset_json_path_val=[data/charades/train_set_sft_new_10k6_src_diff_val.json,data/actnet/train_set_sft_new_9k5_src_diff_val.json,data/vidchapters/train_set_sft_diff_exist_4k3_val.json,data/nextgqa/train_set_sft_src_diff_3k_val.json,data/rextime/train_set_sft_exist_src_diff_3k7_val.json,data/Video-R1-data/train_set_sft_005_val_video.json,data/Video-R1-data/train_set_sft_005_val_image.json,data/longvideo-reason/train_set_sft_src_exist_8k_val.json]
export dataset_video_base=[data/charades/video_14400frames_fps2,data/actnet/video_14400frames_fps2,data/vidchapters/video_14400frames_fps2,data/nextgqa/video_14400frames_fps2,data/rextime/video_14400frames_fps2,data/Video-R1-data/video_14400frames_fps2,data/Video-R1-data,data/longvideo-reason/train_video_14400frames_fps2]
export max_prompt_length=4096
# name
export project_name=sft
export exp_suffix=thinking_lr1e_5

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
          +data.multi_turn.video_base=$dataset_video_base \
          +data.multi_turn.video_kwargs.draw_number=true \
          +data.multi_turn.video_kwargs.parallel=true \
          +data.multi_turn.video_kwargs.max_frames=128 \
          model.partial_pretrain=models/Qwen2.5-VL-7B-Instruct \
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
        for step in 100 200
        do
            echo start eval ${dataset} at step ${step}
            suffix=step${step}
            python eval_temporal_grounding.py \
                --inference_file inference_vllm_origin_number.py \
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
        for step in 100 200
        do
            echo start eval ${dataset} at step ${step}
            suffix=step${step}
            python eval_GQA.py \
                --inference_file inference_vllm_origin_number.py \
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
        for step in 100 200
        do
            echo start eval ${dataset} at step ${step}
            suffix=step${step}
            python eval_VQA.py \
                --inference_file inference_vllm_origin_number.py \
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
        for step in 100 200
        do
            echo start eval ${dataset} at step ${step}
            suffix=step${step}
            python eval_temporal_grounding.py \
                --inference_file inference_vllm_origin_number.py \
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
        for step in 100 200
        do
            echo start eval ${dataset} at step ${step}
            suffix=step${step}
            python eval_temporal_grounding_multi.py \
                --inference_file inference_vllm_origin_number.py \
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
        for step in 100 200
        do
            echo start eval ${dataset} 
            suffix=step${step}
            python eval_VQA.py \
                --inference_file inference_vllm_origin_number.py \
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

