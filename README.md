# HCLmNet: Hybrid Continual Learning Multimodal Network for Lung Cancer Survival Prediction

This repository provides the official implementation of **HCLmNet**, a **Hybrid Continual Learning Multimodal Network** designed to address catastrophic forgetting in **lung cancer survival prediction** using clinical data and CT imaging.  
The framework integrates **Elastic Weight Consolidation (EWC)**, **Experience Replay (ER)**, **Instance-Level Correlation Replay (EICR)**, and **Class-Level Correlation Replay (ECCR)** to support adaptive model updates in real-world hospital environments where patient data continuously evolves.

---

## 🔍 Overview

Lung cancer survival prediction requires robust modeling of heterogeneous data and adaptability to new patient information.  
**HCLmNet** introduces:

- Multimodal learning using **Swin Transformer** (CT images) and **FCN/Tabular encoder** (clinical features)  
- Cross-attention fusion for deep inter-modal interactions  
- A hybrid continual learning strategy (EWC + ER + EICR + ECCR)  
- CoxPH survival risk estimation & ordinal classification support  
- Significant improvement in survival prediction metrics  
  - **C-Index: 0.84**  
  - **MAE: 140 days**  
  - **Forgetting: 0.08**  

---

## 🧠 Key Features

- **Swin Transformer** for CT-based feature extraction  
- **Clinical Feature Encoder** for tabular EHR information  
- **Cross-Attention Fusion Layer** for multimodal alignment  
- **Hybrid Continual Learning** (EWC, ER, EICR, ECCR)  
- **CoxPH Survival Layer** with time-to-event modeling  
- **Modular training pipeline** with configuration files  
- **Reproducible experiments** aligned with the PLOS ONE manuscript  

---

## 📁 Repository Structure

```
HCLmNet-LungCancer-Survival/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── configs/
│ ├── model_config.yaml
│ ├── training_config.yaml
│ └── continual_learning_config.yaml
│
├── data/
│ └── sample/ # small sample (NOT real patient data)
│
├── src/
│ ├── models/
│ │ ├── swin_encoder.py
│ │ ├── xlnet_clinical_encoder.py
│ │ ├── dna_encoder_lstm.py
│ │ └── fusion_cross_attention.py
│ │
│ ├── continual_learning/
│ │ ├── ewc.py
│ │ ├── replay_buffer.py
│ │ ├── eicr.py
│ │ └── eccr.py
│ │
│ ├── survival/
│ │ ├── coxph_layer.py
│ │ └── loss_functions.py
│ │
│ ├── train.py
│ ├── evaluate.py
│ └── utils.py
│
├── notebooks/
│ ├── preprocessing.ipynb
│ ├── training_pipeline.ipynb
│ └── results_visualization.ipynb
│
├── figures/
│ ├── model_architecture.png
│ ├── calibration_curve.png
│ └── survival_curves.png
│
└── docs/
├── installation.md
├── usage_instructions.md
└── continual_learning_explanation.md
```

## 📄 Citation

If you use this code or any part of the HCLmNet framework in your research, please cite the following manuscript:

```bibtex
@article{bappi2025hclmnet,
  title={Hybrid Continual Learning Multimodal Network for Lung Cancer Survival Prediction},
  author={Bappi, Ilias and et al.},
  journal={Manuscript under review},
  year={2025}
}
