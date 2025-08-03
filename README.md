# 🧠 MoE Experiments – Mixture of Experts Transformer in PyTorch  
*A simple Mixture of Experts (MoE) Transformer inspired by LLaMA‑4 for educational purposes.*  

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)]()

---

## 🚀 Quick Start

### 🔧 Installation
```bash
git clone https://github.com/<your-username>/moe-experiments.git
cd moe-experiments
pip install -r requirements.txt

## 📚 Table of Contents
- [📖 What is Being Done](#-what-is-being-done)
- [📂 Repository Contents](#-repository-contents)
- [🧠 Mixture of Experts (MoE)](#-mixture-of-experts-moe)
- [📊 Metrics to Analyze MoE Behavior](#-metrics-to-analyze-moe-behavior)
- [📜 TinyStories Dataset](#-tinystories-dataset)
- [🧪 Designing the Experiment Grid](#-designing-the-experiment-grid)
- [📈 Results](#-results)
- [🚀 Quick Start](#-quick-start)
- [📖 License](#-license)

---

## 📖 What is Being Done

This project explores the **Mixture of Experts (MoE) Transformer**, focusing on how different routing strategies (**top‑1, top‑2, random**) affect **expert utilization, routing entropy, and model performance**.

It is designed for **interpretability and educational clarity**, with detailed analysis of **routing and expert behavior**.  
While MoE architectures are widely used in large‑scale LLMs, here we build a **small‑scale LLM** to provide hands‑on understanding of **expert specialization and routing dynamics**—a foundation for scaling up to larger MoE models.

---

## 📂 Repository Contents

| File / Folder         | Description |
|-----------------------|-------------|
| **model.py**          | Defines the MoE Transformer architecture, router, experts, and all model logic |
| **prepare_data.py**   | Data utilities (character‑level & tiktoken tokenizers, dataset classes for *Alice in Wonderland* and *TinyStories*) |
| **train.py**          | Training loop, loss computation, top‑k accuracy, logging of routing entropy & expert utilization |
| **utils.py**          | Helper functions for saving/loading models and other utilities |
| **moe-transformer.ipynb** | Notebook for training the MoE Transformer & visualizing training curves |
| **moe-analyze.ipynb** | Notebook for analyzing routing entropy, expert utilization, and performance |
| **saved_models/**     | Directory for trained checkpoints |
| **LICENSE**           | MIT License |
| **README.md**         | This file |

---

## 🧠 Mixture of Experts (MoE)

**MoE (Mixture of Experts)** is a neural network technique where a **router selects a subset of specialized “experts” (MLPs)** for each token.  
In Transformers, MoE layers replace or augment standard feed‑forward layers.  

🔹 Only a few experts process each token → **more parameters without increasing per‑token compute**  
🔹 Enables **scaling and specialization** → widely used in large LLMs  

---

## 📊 Metrics to Analyze MoE Behavior

### 🔀 Routing Entropy
Measures **router confidence** in expert selection.

- Low entropy → router is confident (often overconfident)  
- High entropy → router is uncertain  

Formula:  

$$H_i = -\sum_{j=1}^{E} p_{i,j} \cdot \log(p_{i,j} + \epsilon)$$  

Average routing entropy over $N$ tokens:  

$$H = \frac{1}{N} \sum_{i=1}^{N} H_i$$  

---

### 📦 Expert Utilization
Measures **how evenly tokens are distributed** among experts.

- Low std dev → experts used almost equally  
- High std dev → some experts dominate  

Formula:  

$$Utilization_i = \frac{\text{tokens routed to expert } i}{\text{total tokens}}$$

---

## 📜 TinyStories Dataset
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) is a collection of **short, simple stories** ideal for **language modeling research**.  
Its **small size & diverse vocabulary** make it perfect for **rapid MoE prototyping and analysis**.

---

## 🧪 Designing the Experiment Grid

| # Experts | Routing Type   | Notes |
|-----------|---------------|-------|
| 4         | Top‑1         | Standard gating; each token routed to its top expert |
| 4         | Top‑2         | Each token routed to two experts → smoother gradients, better utilization |
| 4         | Top‑1 Random  | Randomly selects one expert per token → sanity check |
| 4         | Top‑2 Random  | Randomly selects two experts per token → sanity check |

### **Top‑1 vs Top‑2**
- Top‑2 smooths gradients and improves utilization balance, but costs more compute.

### **Random Routing**
- Serves as a sanity check to test whether learned routing actually helps.

---

## 📈 Results

### 📝 Generated Text  
**Prompt:** `Once upon a time`  

**Generated Text:**  
> Once upon a time, there was a little girl named Lily. She loved to play and eat carrots. One day, she  
found some her friend named Sue. Sue was very weak and always discussed to her aunt with her favorite basketball.  
>  
> Lily ran to his friend, Tommy, came over to sandwiches. Sue was good at Max

---

### 📉 Loss, Routing Entropy, and Expert Utilization (Top‑2 Routing)
- **Loss vs Training Step:**  
  ![Loss vs Training Step](images/loss_vs_step.png)

- **Routing Entropy vs Training Step:**  
  ![Routing Entropy vs Training Step](images/routing_entropy_vs_step.png)

- **Expert Utilization vs Training Step:**  
  ![Expert Utilization vs Training Step](images/expert_utilization_vs_step.png)

**Key Observations:**  
✅ Routing entropy decreases over training → router becomes more confident  
✅ Early layers show higher entropy → later layers specialize more  
✅ Expert utilization is even in shallow layers but skewed in deeper layers  
✅ After ~500 steps, loss stabilizes but entropy continues decreasing  

📌 **Adding a routing entropy penalty to the loss could improve balance and specialization.**

---

### 🔥 Experiment Grid Results

| Model           | Test Loss | Top‑3 Accuracy | Routing Entropy (5 layers)             | Utilization Std Dev (5 layers) |
|-----------------|-----------|----------------|----------------------------------------|--------------------------------|
| Top‑1           | 2.46      | 66.79%         | Low [1.11 1.08 1.06 0.94 0.78]        | High [0.02 0.11 0.12 0.11 0.09] |
| Top‑1 Random    | 2.53      | 65.57%         | High [1.27 1.27 1.27 1.27 1.27]       | Even [0.01 0.01 0.01 0.01 0.01] |
| Top‑2           | 2.42      | 67.22%         | Low [1.06 1.02 0.95 0.78 0.65]        | High [0.03 0.02 0.07 0.05 0.11] |
| Top‑2 Random    | 2.53      | 65.63%         | High [1.27 1.27 1.27 1.27 1.27]       | Even [0.01 0.01 0.01 0.01 0.01] |

🔹 **Top‑k routing outperforms random routing**  
🔹 **Top‑2 routing performs best** (lower loss, higher accuracy)  
🔹 Random routing works surprisingly well → task may be too simple for full expert specialization.

---
