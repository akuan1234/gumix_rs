# GUMix-RS (Anonymous Repo for Review)

**GUMix-RS** is a **training-free, zero-shot** open-vocabulary semantic segmentation framework tailored to **remote sensing** imagery.  
It resolves the *Scale–Uncertainty Paradox* via a **pixel-wise geometry–uncertainty energy** that routes inference between **1.0× (global context)** and **1.5× (local detail)**, combining:
- **GeoRSCLIP** (RS5M-ViT-H/14) for robust semantics,
- **SAM2** for instance-induced geometric complexity cues,
- **DINOv3-SAT** (ViT-L/16, SAT-493M pre-trained) for geometry-aware correlation refinement.

> This repository is anonymized for double-blind review and contains no personal identifiers.

---

## Performance Snapshot (Numbers)

| Setting | Task | Metric | Best non-ours | GUMix-RS | Relative gain |
|---|---:|---:|---:|---:|---:|
| SAM2\_32 | 8 multi-class RS datasets | Avg. mIoU (%) | 35.78 | **41.97** | **+17.3%** |
| SAM2\_8  | 8 multi-class RS datasets | Avg. mIoU (%) | 35.78 | **40.32** | **+12.7%** |
| SAM2\_8  | 4 binary extraction RS datasets | Avg. fg-IoU (%) | 23.34 | **26.21** | **+12.3%** |


<img width="1822" height="1225" alt="image" src="https://github.com/user-attachments/assets/9adf8c1b-c7ee-4c79-9d91-52298c15611e" />

<img width="2130" height="1235" alt="image" src="https://github.com/user-attachments/assets/ab5f64e1-f8cd-4bc8-bb46-2794127bce46" />

---


## Overview 

<p align="center">
  <img width="2204" height="1084" alt="image" src="https://github.com/user-attachments/assets/fa088fff-4f5a-475b-acec-bbe95229ad01" />


</p>

---

## Repository Layout (main components)

- `configs/` : dataset configs (`cfg_*.py`)
- `eval.py` : evaluate on one dataset
- `eval_all.py` : evaluate on all datasets
- `dist_test.sh` : multi-GPU launcher
- `sam2/, dinov3/`: dependencies / integrated modules
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
python eval.py --config configs/cfg_DATASET.py
```
Multi-GPU
```bash
bash dist_test.sh configs/cfg_DATASET.py NUM_GPU
```
Evaluate on all datasets
```bash
python eval_all.py
```


## Results

  <img width="1322" height="359" alt="image" src="https://github.com/user-attachments/assets/aada89d3-00a1-4a4a-b89e-14dfc0138327" />


  <img width="649" height="351" alt="image" src="https://github.com/user-attachments/assets/4ee84d1e-bf56-49b5-8075-724de240a3be" />


  <img width="646" height="721" alt="image" src="https://github.com/user-attachments/assets/0b59e68c-35b4-4dd8-a55f-275d4fea0f8a" />


  <img width="1034" height="455" alt="image" src="https://github.com/user-attachments/assets/3bef43fa-0f2f-4e51-b51d-587ffe00ae45" />


  <img width="735" height="476" alt="image" src="https://github.com/user-attachments/assets/cefcdb90-1d65-4130-8eae-5aceaddaef97" />



  <img width="757" height="455" alt="image" src="https://github.com/user-attachments/assets/20774870-6d7c-4b9c-b532-96aab2675c92" />

</p>



<p align="center">


</p>

<p align="center">


</p>

<p align="center">


</p>
