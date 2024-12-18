# Mochi-Diffusers-xDiT

This project is based on the Mochi algorithm of the diffuser and optimized and accelerated for multi GPUs inference using the Xufser framework.

## Installation

To set up the environment and install the required dependencies, follow these steps:

### 1. Create and activate the Conda environment:

```
conda create -n mochi-xfuser python=3.10
conda activate mochi-xfuser
```

### 2. Install PyTorch and related libraries:

```
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install the xDiT package:

```
cd xDiT
pip install -e .
```

[Optional] You can install flash-att==2.6.3 by this link:

[flash-attn==2.6.3](https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl)

```
pip install flash_attn-2.6.3+cu118torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### 4. Download the required files:

You can download the necessary files from the following link:

[Download diffusers-mochi.zip](https://huggingface.co/feizhengcong/mochi-1-preview-diffusers/blob/main/diffusers-mochi.zip)

Then extract and navigate into the `diffusers-mochi` folder:

```
cd diffusers-mochi
pip install -e .
```

### 5. Download the model parameters:

You can download the model parameters from this link:

[feizhengcong/mochi-1-preview-diffusers](https://huggingface.co/feizhengcong/mochi-1-preview-diffusers/tree/main)

Alternatively, you can use the following command to download them:

```
pip install -U huggingface_hub
huggingface-cli download --resume-download feizhengcong/mochi-1-preview-diffusers --local-dir feizhengcong/mochi-1-preview-diffusers --local-dir-use-symlinks False
```

## Usage

Once the setup is complete, you can run the example with the following command:

```
torchrun --nproc_per_node=4 examples/mochi_example.py --model 'feizhengcong/mochi-1-preview-diffusers' --ulysses_degree 2 --ring_degree 1 --use_cfg_parallel --height 480 --width 848 --num_frames 163 --prompt "your prompt" --num_inference_steps 50 --seed 42
```

## Related Links

- [xDiT](https://github.com/xdit-project/xDiT)
- [Mochi](https://github.com/genmoai/mochi)
- Mochi's Diffusers Integration Contributors:
  - [Zhengcong Fei](https://github.com/feizc)

