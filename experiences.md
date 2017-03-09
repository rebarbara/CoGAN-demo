# CoGAN demo

This document aims to record some experiences while deploying [andrewliao11](https://github.com/andrewliao11/CoGAN-tensorflow)'s implementation of a [CoGAN](https://arxiv.org/abs/1606.07536) (Coupled Generative Adversarial Networks) model. Following along the lines of the code, I attempt to understand the architecture design decisions that were made.


<img src="https://github.com/rebarbara/CoGAN-demo/blob/master/gan_simple.svg" width="500" />
<img src="https://github.com/rebarbara/CoGAN-demo/blob/master/gan_probabilistic.svg" width="500" />


**Generative models**

| Layer   | Domain 1      | Domain 2    | Shared?  |
| ------- |:-------------:|:-----------:|:--------:|
| 1       | Linear, BN, PReLU  | Linear, BN, PReLU | Yes |
| 2       | Linear, BN, PReLU  | Linear, BN, PReLU | Yes |
| 3       | CONV, BN, PReLU | CONV, BN, PReLU | Yes |
| 4       | CONV, Sigmoid | CONV, Sigmoid | No  |


**Discriminative models**

| Layer   | Domain 1      | Domain 2    | Shared?  |
| ------- |:-------------:|:-----------:|:--------:|
| 1       | CONV, PReLU  | CONV, PReLU | No |
| 2       | CONV, BN, PReLU  | CONV, BN, PReLU | No |
| 3       | Linear, BN, PReLU | Linear, BN, PReLU | Yes |
| 4       | Linear | Linear | Yes  |
| 5       | Sigmoid | Sigmoid | Yes  |
