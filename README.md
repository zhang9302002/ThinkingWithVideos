# Thinking With Videos: Multimodal Tool-Augmented Reinforcement Learning for Long Video Reasoning

<a href="https://zhang9302002.github.io/">Haoji Zhang</a><sup>\*</sup>,
<a href="https://gx77.github.io/GX/">Xin Gu</a><sup>\*</sup>,
Jiawen Li,
Chixiang Ma,
<a href="https://sulebai.github.io/">Sule Bai</a>,
<a href="https://lin-shan.com/">Chubin Zhang</a>,
Bowen Zhang,
Zhichao Zhou,
Dongliang He,
<a href="https://andytang15.github.io/">Yansong Tang</a><sup>&dagger;</sup>

<sup>\*</sup>Equal contributions, 
<sup>&dagger;</sup>Correspondence

<a href="https://zhang9302002.github.io/thinkingwithvideos-page/"><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href="https://arxiv.org/abs/2508.04416"><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href="https://huggingface.co/datasets/zhang9302002/MultiTaskVideoReasoning"><img src='https://img.shields.io/badge/Data-Huggingface-yellow'></a>

We proposed **VITAL**, a tool-augmented framework that enables advanced long video reasoning and temporal grounding.

We also introduce **MTVR**, a high-quality multi-task video reasoning training dataset.

<img src="assets/framework.png" alt="framework" width="100%">

## Contents
- [MTVR Dataset](#MTVR-Dataset)
- [Training and Evaluation](#Training-and-Evaluation)
- [Citation](#Citation)

## MTVR Dataset
The dataset is available [here](https://huggingface.co/datasets/zhang9302002/MultiTaskVideoReasoning).

## Training and Evaluation

### 1. Prepare coding environment:

This project is based on verl 0.4.0.dev, vllm 0.8.5.post1, transformers 4.51.1, torch 2.6.0 and python 3.10. Install dependencies:
```bash
bash setup.sh
```

### 2. Prepare model and data:

```bash
VITAL/
├── data/
│   ├── actnet/
│   ├── charades/
│   ├── longvideo-reason/
│   ├── mmvu/
│   ├── nextgqa/
│   ├── rextime/
│   ├── vidchapters/
│   ├── Video-R1-data/
│   ├── videomme/
│   ├── videommmu/
│   ├── vidi/
│   ├── vsibench/
├── models/
│   ├── Qwen2.5-VL-3B-Instruct/
│   ├── Qwen2.5-VL-7B-Instruct/
├── outputs/ # an empty folder to store the outputs
```

- Prepare Qwen2.5-VL-7B-Instruct model

```bash
mkdir -p models
huggingface-cli download --resume-download --local-dir-use-symlinks False  Qwen/Qwen2.5-VL-7B-Instruct --revision main --local-dir ./models/Qwen2.5-VL-7B-Instruct
```

- Prepare dataset for training and evaluation. You can download the following datasets from there official websites:

| Folder | Dataset | Source |
| --- | --- | --- |
| data/actnet | ActivityNet-MR | https://cs.stanford.edu/people/ranjaykrishna/densevid/ |
| data/charades | Charades-STA | https://github.com/jiyanggao/TALL |
| data/longvideo-reason | LongVideo-Reason | https://github.com/NVlabs/Long-RL/tree/main/longvideo-reason |
| data/mmvu | MMVU | https://github.com/yale-nlp/MMVU |
| data/nextgqa | NExT-GQA | https://github.com/doc-doc/NExT-GQA |
| data/rextime | ReXTime | https://huggingface.co/datasets/ReXTime/ReXTime |
| data/vidchapters | VidChapters-7M | https://github.com/antoyang/VidChapters |
| data/Video-R1-data | Video-R1 | https://huggingface.co/datasets/Video-R1/Video-R1-data |
| data/videomme | Video-MME | https://github.com/MME-Benchmarks/Video-MME |
| data/videommmu | Video-MMMU | https://github.com/EvolvingLMMs-Lab/VideoMMMU |
| data/vidi | VIDI / VUE-TR | https://github.com/bytedance/vidi |
| data/vsibench | VSI-Bench | https://github.com/vision-x-nyu/thinking-in-space |


### 3. Training and evaluation script

Run the following script to train and evaluate the model:
```bash
bash train_stage_1_sft.sh
bash train_stage_2_dgrpo.sh
bash train_stage_3_sft.sh
bash train_stage_4_dgrpo.sh
```
Note:
- You need to set the some configuration in the corresponding script, such as `PRETRAINED_CKPT`, `YOUR_WANDB_API_KEY`.


## Citation
If you find this project useful in your research, please consider citing:

```
@article{zhang2025thinking,
  title={Thinking With Videos: Multimodal Tool-Augmented Reinforcement Learning for Long Video Reasoning},
  author={Zhang, Haoji and Gu, Xin and Li, Jiawen and Ma, Chixiang and Bai, Sule and Zhang, Chubin and Zhang, Bowen and Zhou, Zhichao and He, Dongliang and Tang, Yansong},
  journal={arXiv preprint arXiv:2508.04416},
  year={2025}
}
```