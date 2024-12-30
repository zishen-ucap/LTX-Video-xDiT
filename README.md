# Mochi-Diffusers-xDiT

This project is based on the [LTX-Video](https://github.com/Lightricks/LTX-Video) algorithm of the diffusers and optimized and accelerated for multi GPUs inference using the [xDiT](https://github.com/xdit-project/xDiT) framework.

## Installation

To set up the environment and install the required dependencies, follow these steps:

### 1. Create and activate the Conda environment:

```
conda create -n ltxvideo-xfuser python=3.10
conda activate ltxvideo-xfuser
```

### 2. Install PyTorch and related libraries:

```
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install the xDiT package:

```
git clone https://github.com/zishen-ucap/LTX-Video-xDiT.git
cd LTX-Video-xDiT
pip install -e .
```

You can install flash-att==2.6.3 by this link:

[flash-attn==2.6.3](https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl)

```
pip install flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```


### 4. Download the model parameters:

You can download the model parameters from this link:

[Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video)

Alternatively, you can use the following command to download them:

```
pip install -U huggingface_hub
huggingface-cli download --resume-download Lightricks/LTX-Video --local-dir Lightricks/LTX-Video --local-dir-use-symlinks False
```

## Usage

Once the setup is complete, you can run the example with the following command:

```
CUDA_VISIBLE_DEVICES=0,1,2,3, torchrun --nproc_per_node=4 examples/ltxvideo_example.py --model './Lightricks/LTX-Video' --ulysses_degree 1 --ring_degree 2 --use_cfg_parallel --height 512 --width 768 --num_frames 161 --prompt "Your prompt" --num_inference_steps 50 --seed 42
```


