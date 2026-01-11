# GUMix-RS (Anonymous Code Release)

Training-free, zero-shot **open-vocabulary semantic segmentation** for **remote sensing** with pixel-wise **scale routing** (1.0× / 1.5×) guided by geometry–uncertainty cues.

> This repository is anonymized for double-blind review (no personal information).

---

## Overview (place your figure here)

<!-- You can replace the path with your own image -->
<p align="center">
  <img width="1316" height="540" alt="image" src="https://github.com/user-attachments/assets/3e03d1aa-cb84-415d-8f59-4e89762403ee" />

</p>

---

## Repository Layout (main components)

- `configs/` : dataset configs (`cfg_*.py`)
- `eval.py` : evaluate on one dataset
- `eval_all.py` : evaluate on all datasets
- `dist_test.sh` : multi-GPU launcher
- `sam2/`, `dinov3/`, `open_clip/`, `CropFormer/`, `eomt/` : dependencies / integrated modules
- `requirements.txt` : extra python dependencies

---

## Environment & Installation

**Python:** 3.10

### 1) Install MMCV/MMEngine/MMSeg via OpenMIM

```bash
pip install -U openmim
mim install mmengine==0.10.7
mim install mmcv==2.1.0
pip install mmsegmentation==1.2.2

```

### 2) Install remaining requirements
```bash
pip install -r requirements.txt
```

## Usage

Single-GPU
```bash
python eval.py --config config/cfg_DATASET.py
```
Multi-GPU
```bash
bash dist_test.sh config/cfg_DATASET.py NUM_GPU
```
Evaluate on all datasets
```bash
python eval_all.py
```


## Results





