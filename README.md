
# Lost in Interpolation: Why Predictive Feedback Fails in Diffusion Language Models

<div align="center">

**Accepted to Findings of EACL 2027**

Lavanya Nigam · Ishaan Bansal · Aryan Sood · Vidit Aggarwal · Gaurav Kumar Nayak

[![EACL 2027](https://img.shields.io/badge/EACL_2027-Findings-blue)](https://2027.eacl.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2608.06529-b31b1b.svg)](https://arxiv.org/abs/2608.06529)
[![PyTorch](https://img.shields.io/badge/PyTorch-Implementation-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)

**[Paper](https://arxiv.org/abs/2608.06529)**

</div>

Official PyTorch implementation of **Spherical Soft-Masking (S-SM)** for masked diffusion language models.

<div align="center">
  <img src="./assets/slerp_vs_lerp.png" width="60%"/>
</div>

## News

- **[November 2026]** Our paper has been accepted to **Findings of EACL 2027**!
- **[August 2026]** Our paper is available on [arXiv](https://arxiv.org/abs/2608.06529).


## Quickstart

### 1. Language Modeling

#### Installation

```bash
cd language

conda create -n sm-env python=3.12 -y
conda activate sm-env
conda install nvidia/label/cuda-12.4.0::cuda-toolkit

pip install -r requirements.txt
pip install flash_attn==2.7.4.post1 --no-build-isolation

bash setup.sh
wandb login
```

#### Continued Pretraining

Download the base MDLM checkpoint from [Google Drive](https://drive.google.com/drive/folders/16LuuptK7Xfk-vzhQYZBZ0SA-B-BFluau?usp=sharing).

```bash
NUM_GPUS=2 BASE_CKPT=/path/to/mdlm.ckpt \
bash scripts/run_slerp_multigpu.sh
```

#### Training from Scratch

```bash
bash scripts/run_slerp_multigpu.sh
```

The launcher automatically detects available GPUs and maintains a fixed effective global batch size across GPU counts.

#### Evaluation

Specify the trained checkpoint in the corresponding evaluation script.

```bash
# Standard MDLM sampling
bash scripts/03_gen_ppl_owt_mdlm_sm.sh 1024

# ReMDM sampling
bash scripts/04_gen_ppl_owt_remdm_cap_sm.sh 512
bash scripts/05_gen_ppl_owt_remdm_loop_sm.sh 1024
```

Generated samples and evaluation metrics (generative perplexity, entropy and MAUVE) are stored in the `generations/` directory.

For additional configurations, baselines and execution instructions, see:

- [Language Modeling README](./language/README.md)
- [Modal Setup](./language/MODAL_SETUP.md)
- [Verda Setup](./language/verda/README.md)

### 2. Code Generation

Built upon Dream-7B using PEFT / DoRA fine-tuning.

#### Installation

```bash
cd coding
pip install -r requirements.txt
huggingface-cli login
```

Replace `<YOUR_HF_USERNAME>` with your Hugging Face username in the relevant configuration files.

Follow the patching and model setup instructions in the [Code Generation README](./coding/README.md) before training.

#### Fine-tuning

```bash
cd train

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

python train.py \
    "./configs/base_configs/config_1.json" \
    "./configs/datasets_softmasking/config_coding.json" \
    ""
```

For baseline configurations, additional datasets, training seeds and HumanEval / EvalPlus evaluation, see [`coding/README.md`](./coding/README.md).

## Model Checkpoints

Earlier language modeling checkpoints are available on [Hugging Face](https://huggingface.co/lavanyanigam/soft-masking-checkpoints).

Checkpoint availability for the EACL 2027 version will be updated here.

## Acknowledgements

This codebase builds upon:

- [Duo](https://github.com/s-sahoo/duo)
- [ReMDM](https://github.com/kuleshov-group/remdm)
- [Dream-7B](https://github.com/DreamLM/Dream)
- [Soft-Masked Diffusion Language Models](https://openreview.net/forum?id=Gba02UMvrG)

## Citation

If you find our work useful, please cite:

```bibtex
@article{nigam2026lost,
  title={Lost in Interpolation: Why Predictive Feedback Fails in Diffusion Language Models},
  author={Nigam, Lavanya and Bansal, Ishaan and Sood, Aryan and Aggarwal, Vidit and Nayak, Gaurav Kumar},
  journal={arXiv preprint arXiv:2608.06529},
  year={2026}
}
```

The official ACL Anthology citation will be added upon publication.

## License

This repository is licensed under the [Apache License 2.0](./LICENSE).
