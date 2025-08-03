# 🧠 MoE Experiments – Mixture of Experts Transformer in PyTorch  
*A minimal and educational implementation of a Mixture of Experts (MoE) Transformer inspired by LLaMA‑4.*  

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)]()

---

## 📚 Table of Contents
- [📖 About the Project](#-about-the-project)
- [📂 Repository Structure](#-repository-structure)
- [🧠 Mixture of Experts (MoE)](#-mixture-of-experts-moe)
- [📊 Metrics to Analyze MoE](#-metrics-to-analyze-moe-behavior)
- [📜 Dataset: TinyStories](#-tinystories-dataset)
- [🚀 Quick Start](#-quick-start)
- [📈 Experiments & Results](#-experiments--results)
- [🔮 Future Work](#-future-work)
- [📖 Citation](#-citation)

---

## 📖 About the Project
This repository implements a **Mixture of Experts (MoE) Transformer** in PyTorch for educational purposes.  
It is inspired by **LLaMA‑4's MoE architecture**, featuring:

✅ **Top‑k expert routing (Top‑1, Top‑2, Random)**  
✅ Modular design for easy experimentation  
✅ Mini dataset pipeline using the **TinyStories dataset**  
✅ Detailed analysis of **routing entropy, expert utilization, and specialization**  

The goal is to provide **hands‑on understanding** of how MoE models work, making it a **starting point for scaling to larger LLMs**.

---

## 📂 Repository Structure

| File | Description |
|------|------------|
| `model.py` | Defines the MoE Transformer architecture, router, experts, and forward logic |
| `prepare_data.py` | Data preparation utilities (character-level & tiktoken tokenizers) |
| `train.py` | Training loop with loss computation, accuracy metrics, and logging |
| `utils.py` | Helper functions for saving/loading models and utilities |
| `moe-transformer.ipynb` | Notebook for training and visualizing loss/metrics |
| `moe-analyze.ipynb` | Notebook for analyzing expert utilization and entropy |
| `saved_models/` | Folder to store trained checkpoints |
| `LICENSE` | MIT License |
| `README.md` | This file |

---

## 🧠 Mixture of Experts (MoE)

**MoE (Mixture of Experts)** is a technique where a **router dynamically selects a subset of experts (MLPs)** to process each token.  
This allows **increased model capacity** without increasing per‑token computation.

🔹 **Key Idea:**  
- Router assigns each token to **Top‑k experts**  
- Only selected experts compute forward pass  
- Improves **efficiency, scalability, and specialization**

---

### 🔀 Routing Strategies
| Routing Type   | Description |
|---------------|------------|
| **Top‑1**     | Each token routed to the most probable expert |
| **Top‑2**     | Each token routed to 2 experts → smoother gradients |
| **Random**    | Experts selected randomly (sanity check) |

---

## 📊 Metrics to Analyze MoE Behavior

### 1️⃣ **Routing Entropy**
Measures router **confidence** in expert selection.

- Low entropy → router is confident (may be overconfident)  
- High entropy → router is uncertain

Formula:  
$$H_i = -\sum_{j=1}^{E} p_{i,j} \log(p_{i,j}+\epsilon)$$

---

### 2️⃣ **Expert Utilization**
Measures how **evenly tokens are distributed** among experts.

- Low std dev → Balanced expert usage  
- High std dev → Some experts dominate  

Formula:  
$$Utilization_i = \frac{\text{tokens sent to expert } i}{\text{total tokens}}$$

---

## 📜 TinyStories Dataset
We use [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) – a collection of **short, simple stories** perfect for rapid prototyping and analyzing expert routing behaviors in MoE models.

---

## 🚀 Quick Start

### 🔧 Installation
```bash
git clone https://github.com/<your-username>/moe-experiments.git
cd moe-experiments
pip install .
