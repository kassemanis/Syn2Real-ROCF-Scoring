# Synthetic-to-real Attentive Deep Learning for Alzheimer's Assessment

[![Paper](https://img.shields.io/badge/paper-Journal%20of%20Biomedical%20Informatics-blue)](https://www.sciencedirect.com/journal/journal-of-biomedical-informatics) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation for the paper: **"Synthetic-to-real attentive deep learning for Alzheimer's assessment: a domain-agnostic framework for ROCF scoring"**.

Our work introduces a framework to automate the scoring of the Rey-Osterrieth Complex Figure (ROCF) test, a key tool in cognitive assessment for Alzheimer's disease. We tackle the critical challenges of data scarcity and domain shift by combining a novel synthetic data generation pipeline with a domain-agnostic deep learning model, **ROCF-Net**.


## Framework Overview

Our framework consists of two main stages:

### Synthetic Data Generation

We generate synthetic ROCF drawings in a multi-step process designed to mimic real-world variations. The main parts of this process are:

* **Unit Decomposition & Prototype Generation:** The ground-truth ROCF figure is broken down into its 18 standard scoring units, and based on scoring criteria, these units are probabilistically selected and geometrically distorted to create a prototype drawing (see `Syn2Real-ROCF-Scoring/rocf_synthesis/prototype_generation.py`).
  
* **Geometry and Style Transfer:** The style (e.g., stroke texture, line quality) from a real patient drawing is transferred to the prototype, creating a realistic final image with an associated score ([refer to the official implementation](https://github.com/sunniesuhyoung/DST)).

### ROCF-Net Scoring Model

ROCF-Net is an end-to-end deep learning model for regressing the final ROCF score from an input drawing. Its architecture is based on a ResNet50 backbone with key modification:
* **MixStyle Layers:** To improve domain generalization, feature statistics are mixed between instances during training, making the model robust to variations in drawing style ([refer to the official implementation](https://github.com/KaiyangZhou/mixstyle-release)).

* **Laplacian Attention (LaplacianAtt):** A custom attention module that uses Laplacian filtering to detect edges, forcing the model to focus on the structural components of the drawing relevant for scoring.


If you find this work useful in your research, please consider citing our paper:

```bash
@article{Bouali2025ROCFNet,
  title   = {Synthetic-to-real attentive deep learning for Alzheimer's assessment: a domain-agnostic framework for ROCF scoring},
  author  = {Kassem Anis Bouali and Elena Šikudová},
  journal = {Journal of Biomedical Informatics},
  year    = {2025},
  note    = {Preprint}
}
```


### Acknowledgements
We acknowledge financial support from Charles University through the SVV-260699 internal grant. The data used in this study were obtained from the project "Štandardizácia neuropsychologickej batérie NEUROPSY" (APVV-15-0686).
