# moe-experiments
This project implements a simple Mixture of Experts (MoE) Transformer in PyTorch for educational purposes. Inspired by LLaMA-4 architecture, it features top-k expert routing, modular design, and a mini dataset pipeline using Tiny Shakespeare dataset.

This repository is inspired by FareedKhan-dev/train-llama4, which provides a didactic implementation of a Llama-4-based MoE model.

TO DO: 
1. Alice dataset, analyze results with top-1, top-2 routings (Done)
2. Add random routing, analyze it for Alice dataset
3. Interpret all the results
4. Repeat for TinyStories dataset
5. Add feed-forward layer, and compare with MoE (number of parameters, FLOPs)
6. Documentation