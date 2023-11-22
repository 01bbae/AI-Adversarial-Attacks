# AI-Adversarial-Attacks

This repository contains the implementation of Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) and analysis. It also contains the proof for FGSM using KKT conditions.

The purpose of using the adversarial attacks was to fool LeNet to incorrectly classify MNIST data to an untargeted attack label with small perturbations in the image.

FGSM was taken from PyTorch's documentation here: https://pytorch.org/tutorials/beginner/fgsm_tutorial.html

The PGD algorithm and comparison table were self-implemented.

The hyperparameters for PGD are set as alpha = 10/255 and epsilon = 25/255

References:

- Explaining and Harnessing Adversarial Examples
  - https://arxiv.org/abs/1412.6572
- Adversarial examples in the physical world
  - https://arxiv.org/abs/1607.02533
