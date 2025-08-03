# moe-experiments
This project implements a simple Mixture of Experts (MoE) Transformer in PyTorch for educational purposes. 
Inspired by LLaMA-4 architecture, it features top-k expert routing, modular design, and a mini dataset 
pipeline using the Tiny Stories dataset.

This repository is inspired by [FareedKhan-dev/train-llama4](https://github.com/FareedKhan-dev/train-llama4).

---

## Repository Contents

- **model.py**: Defines the MoE Transformer architecture, including the router, experts, and all model 
logic.
- **prepare_data.py**: Data preparation utilities, including character-level and tiktoken-based tokenizers, 
and dataset classes for both 'a sample paragraph from Alice in Wonderland' and 'TinyStories'.
- **train.py**: Training loop, loss and top-k accuracy computation, and logging of routing entropy and 
expert utilization.
- **utils.py**: Utility functions for saving/loading models and other helpers.
- **moe-transformer.ipynb**: Jupyter notebook for training the MoE Transformer model and visualizing 
training curves.
- **moe-analyze.ipynb**: Jupyter notebook for analyzing routing entropy, expert utilization, and model 
performance.
- **saved_models/**: Directory for storing trained model checkpoints.
- **LICENSE**: MIT License.
- **README.md**: This file.

## What is Being Done

This repository explores the Mixture of Experts (MoE) Transformer, focusing on how different routing 
strategies (top-1, top-2, random) affect expert utilization, routing entropy, and model performance. 
The project is designed for interpretability and educational clarity, with detailed analysis of routing 
and expert behavior. 

Typically, MoE architectures are used in large-scale language models (LLMs) to efficiently scale model 
capacity. However, in this repository, we implement a small-scale LLM to provide a hands-on understanding 
of how MoE works. We analyze and interpret the trained MoE model, aiming to uncover insights into expert 
specialization and routing dynamics. This serves as a foundation for building and experimenting with 
larger-scale MoE Transformer models in the future.

---

## Procedure

### Mixture of Experts (MoE)
MoE (Mixture of Experts) is a neural network technique where a router selects a subset of specialized 
"experts" (MLPs) for each input token. In Transformer architectures, MoE layers typically replace or 
augment standard feedforward layers. The router dynamically assigns each token to one or more experts, 
so only a few experts process each token, allowing the model to scale up total parameters without increasing 
per-token computation. MoE increases model capacity and diversity, enabling efficient scaling and specialization, 
which is especially useful for large language models.

### Metrics to analyze MoE behavior

#### Routing Entropy
Routing entropy measures the confidence of the router in its expert selection. 

Low entropy -> router is confident (often overconfident)
**Formula:**

For a token $i$ with gating probabilities $p_i = [p_{i,1}, p_{i,2}, ..., p_{i,E}]$ over $E$ experts:

$$
H_i = -\sum_{j=1}^{E} p_{i,j} \cdot \log(p_{i,j} + \epsilon)
$$

where $\epsilon$ is a small constant to prevent $\log(0)$. The gating probabilities are computed 
by applying the softmax function to the logits (the outputs of the router network).

Average routing entropy over all $N$ tokens:

$$
H = \frac{1}{N} \sum_{i=1}^{N} H_i
$$

where $N$ is the number of tokens.

#### Expert Utilization

Expert utilization quantifies how evenly tokens are distributed among experts. Ideally, we would like 
that all experts are used equally. The standard deviation of expert utilization can be used to measure 
this balance.

Low std dev → Experts are used almost equally. The router (or random routing) spreads tokens evenly.

High std dev → Some experts get many tokens, others get few. The router is biased toward certain experts.

**Formula:**
    Utilization of expert $i$ is given by
    Utilization_i = \frac{\text{tokens routed to expert } i}{\text{total tokens}}

### TinyStories Dataset
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) is a collection of short, simple 
stories designed for language modeling research. Its small size, simplicity, and diverse vocabulary 
make it ideal for Mixture of Experts (MoE) experiments, as it enables rapid prototyping and clear analysis 
of expert routing and specialization behaviors in a controlled setting.

---

## Designing the Experiment Grid

| # Experts | Routing Type   | Notes                                                        |
|-----------|---------------|--------------------------------------------------------------|
| 4         | Top-1         | Standard gating; each token routed to its top expert         |
| 4         | Top-2         | Each token routed to two experts; smoother gradients, better utilization |
| 4         | Top-1 Random  | Randomly selects one expert per token; sanity check          |
| 4         | Top-2 Random  | Randomly selects two experts per token; sanity check         |

**Top-1 vs. Top-2:**
Top-2 generally smooths gradients better and improves utilization balance but costs more compute (two 
experts per token).

**Random Routing:**
Randomly selects experts — useful as a sanity check to see whether intelligent routing really helps.

## Results

### Generated Text
**Prompt:** `Once upon a time`

**Generated Text:**

Once upon a time, there was a little girl named Lily. She loved to play and eat carrots. One day, she 
found some her friend named Sue. Sue was very weak and always discussed to her aunt with her favorite basketball.

Lily ran to his friend, Tommy, came over to sandwiches. Sue was good at Max

### Analyzing loss, routing entropy, and expert utilization during training
Below plots show the evolution of loss, routing entropy, and expert utilization during training for 
Top-2 routing

- **Loss vs Training Step:**
    ![Loss vs Training Step](images/loss_vs_step.png)

- **Routing Entropy vs Training Step:**  
    ![Routing Entropy vs Training Step](images/routing_entropy_vs_step.png)

- **Expert Utilization vs Training Step:**  
    ![Expert Utilization vs Training Step](images/expert_utilization_vs_step.png)

**Interpretation**
- The routing entropy decreases during the training, indicating that the network's confidence to 
predict the next token keeps on increasing.
- Early layers show higher routing entropy (more uncertainty), while later layers are more confident 
(lower entropy), reflecting their greater impact on predictions.
- Expert utilization is even for lower layers. For deeper layers, some experts are used more than the 
- After around 500 training steps, the training loss and expert utilization have nearly stabilized, but 
routing entropy continues to decrease. This is because only the top-k experts are selected deterministically, 
rather than sampling from the routing distribution. As a result, changes in the router logits have limited 
effect on routing entropy and overall performance. To address this, adding a routing entropy penalty 
to the loss function could encourage more balanced and confident expert selection, potentially improving 
model performance and expert specialization.

### Results of Experiment Grid
| Model           | Test Loss | Top-3 Accuracy | Routing Entropy (all 5 layers)         | Utilization Std Dev (all 5 layers)      |
|-----------------|-----------|----------------|----------------------------------------|-----------------------------------------|
| Top-1           | 2.46      | 66.79%         | Low [1.11 1.08 1.06 0.94 0.78]        | High [0.02 0.11 0.12 0.11 0.09]         |
| Top-1 Random    | 2.53      | 65.57%         | High [1.27 1.27 1.27 1.27 1.27]       | Even [0.01 0.01 0.01 0.01 0.01]         |
| Top-2           | 2.42      | 67.22%         | Low [1.06 1.02 0.95 0.78 0.65]        | High [0.03 0.02 0.07 0.05 0.11]         |
| Top-2 Random    | 2.53      | 65.63%         | High [1.27 1.27 1.27 1.27 1.27]       | Even [0.01 0.01 0.01 0.01 0.01]         |

- **Routing Entropy (per layer):** Lower values mean the router is more confident in its expert selection; higher values indicate more uncertainty.
- **Expert Utilization Std Dev (per layer):** Higher values mean some experts are used much more than others (imbalanced); lower values ("Even") mean experts are used equally.


**Interpretation:**
- Top-k routing outperforms random routing, with lower test loss and higher top-3 accuracy.
- Top-2 routing yields better results than Top-1, improving both loss and accuracy.
- Effective MoE requires the router to specialize experts (moderate entropy), maintain balanced expert 
utilization, and ensure smooth gradient flow for learning.
- Random routing performs only slightly worse than learned routing, suggesting that even a single expert 
can handle the task reasonably well. This indicates that the problem may be too simple for the network 
to fully benefit from expert specialization.
