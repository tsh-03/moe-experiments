# moe-experiments
This project implements a simple Mixture of Experts (MoE) Transformer in PyTorch for educational purposes. Inspired by LLaMA-4 architecture, it features top-k expert routing, modular design, and a mini dataset pipeline using Tiny Shakespeare dataset.

This repository is inspired by FareedKhan-dev/train-llama4, which provides a didactic implementation of a Llama-4-based MoE model.

TO DO: 
1. Alice dataset, analyze results with top-1, top-2 routings (Done)
2. Add random routing, analyze it for Alice dataset (Done)
3. Interpret all the results
4. Repeat for TinyStories dataset (Done)
5. In training, replace wording 'epochs' with 'steps' (Done)
6. Remove warning for tiny stories when the sample is smaller than context size (Done)
7. Add top-5 accuracy
8. Documentation
9. Run final code on Lambda, and publish the repository