# Express4D: Expressive, Friendly, and Extensible 4D Facial Motion Generation Benchmark

[![arXiv](https://img.shields.io/badge/arXiv-<2508.12438>-<COLOR>.svg)](https://arxiv.org/abs/2508.12438)
[![video](https://img.shields.io/badge/youtube-<Express4D>-<>.svg)](https://youtu.be/BhBdkzao4kA)

The official PyTorch implementation of the paper [**"Express4D: Expressive, Friendly, and Extensible 4D Facial Motion Generation Benchmark"**](https://arxiv.org/abs/2508.12438).

Please visit our [**webpage**](https://jaron1990.github.io/Express4D/) for more details.

![teaser](static/Presentation2.gif)

## Bibtex

If you find this code useful in your research, please cite:

```
@misc{aloni2025express4dexpressivefriendlyextensible,
title={Express4D: Expressive, Friendly, and Extensible 4D Facial Motion Generation Benchmark}, 
author={Yaron Aloni and Rotem Shalev-Arkushin and Yonatan Shafir and Guy Tevet and Ohad Fried and Amit Haim Bermano},
year={2025},
eprint={2508.12438},
archivePrefix={arXiv},
primaryClass={cs.GR},
url={https://arxiv.org/abs/2508.12438}
}
```

## Getting started

This code was tested on `Ubuntu 22.04.5 LTS` and requires:

* Python 3.10
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### 1. Setup environment
Setup conda env:
```shell
conda env create -f environment.yml
conda activate Express4D
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/GuyTevet/smplx.git
```

### 2. Download Express4D dataset

1. fill the google form: https://forms.gle/uSgMH7J6cpPC4oMY9
2. Copy all folders and files to <ROOT>/dataset/

### 3. download models:
1. trained model from: https://drive.google.com/drive/u/4/folders/1-FBG3nDVhJ0IHTJiT_mWRm0UJRsKfxws
2. evaluators from: https://drive.google.com/drive/u/4/folders/17MLLa55z-B-FFVwGQzNvSNB1g8C8asaq

### 4. Training:
```
python -m train.train_mdm --save_dir <OUTPUT_DIR> --dataset express4d --data_mode arkit --device <CUDA_DEVICE_ID> --save_interval <SAVE_EVERY_ITER> --cond_mode text --overwrite --use_ema  
```
optional flags:
```
--flip_face_on - flip horizontally
--visualize_during_training - visualize samples every save_interval
--eval_during_training - evaluate metrics every save_interval. requires also:
    --eval_mode full 
    --eval_model_name <evaluator_path to tex_mot_match root (in section 3.2)>
```

### 5. Sample:
```
python -m sample.generate_face --model_path <PATH_TO_MODE.pt> --device <CUDA_DEVICE_ID> --num_samples <SAMPLE_COUNT> --num_repetitions <DIFFERENT_SEEDS>
```

### 6. Train evaluators:
```
python -m data_loaders.humanml.train_decomp_v3 --dataset_name express4d --name <out_decomp_name> 
python -m data_loaders.humanml.train_tex_mot_match --dataset_name express4d --decomp_name <in_decomp_name (out from last command)> --name <out_evaluator_name>

```