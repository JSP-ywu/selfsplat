# SelfSokat

## Installation

To get started, create a virtual environment using Python 3.10+:

```bash
conda create -n selfsplat python=3.10 -y
conda activate selfsplat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Install RoPE2D

```bash
cd src/model/encoder/croco/croco_backbone/curope
python setup.py build_ext --inplace
```

</details>

## Acquiring Pre-trained Checkpoints and Datasets

You can find pre-trained checkpoints and datasets [here](https://huggingface.co/anonymous-submit/submission/tree/main). Put `CroCo_V2_ViTLarge_BaseDecoder.pth` in `checkpoints` directory. Unzip data subset in `dataset`.

## Running the Code

### Training

The main entry point is `src/main.py`. Call it via:

```bash
python3 -m src.main +experiment=re10k
```

This configuration requires a single GPU with 80 GB of VRAM (A100 or H100). To reduce memory usage, you can change the batch size as follows:

```bash
python3 -m src.main +experiment=re10k data_loader.train.batch_size=1
```

Our code supports multi-GPU training. The above batch size is the per-GPU batch size.

### Evaluation

To render frames from an existing checkpoint, run the following:

```bash
# Real Estate 10k
python3 -m src.main +experiment=re10k mode=test checkpointing.load=pretrained/re10k.ckpt

# ACID
python3 -m src.main +experiment=acid mode=test checkpointing.load=pretrained/acid.ckpt
```

## Camera Conventions

Our extrinsics are OpenCV-style camera-to-world matrices. This means that +Z is the camera look vector, +X is the camera right vector, and -Y is the camera up vector. Our intrinsics are normalized, meaning that the first row is divided by image width, and the second row is divided by image height.
