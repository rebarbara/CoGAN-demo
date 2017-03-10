# CoGAN demo

This document aims to record some experiences while deploying [andrewliao11](https://github.com/andrewliao11/CoGAN-tensorflow)'s implementation of a [CoGAN](https://arxiv.org/abs/1606.07536) (Coupled Generative Adversarial Networks) model. Following along the lines of the code, I attempt to understand the architecture design decisions that were made.

<p align="center">
<img src="https://github.com/rebarbara/CoGAN-demo/blob/master/gan_simple.png" width="500" />
<img src="https://github.com/rebarbara/CoGAN-demo/blob/master/gan_prob.png" width="500" />
</p>

```
$ git clone https://github.com/andrewliao11/CoGAN-tensorflow.git
$ python download.py mnist
$ python invert.py 
$ python main.py --is_train True
```



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

<p align="center">
<img src="https://github.com/rebarbara/CoGAN-demo/blob/master/train_00_0099.png" width="150" />
<img src="https://github.com/rebarbara/CoGAN-demo/blob/master/train_01_0053.png" width="150" />
<img src="https://github.com/rebarbara/CoGAN-demo/blob/master/train_04_0115.png" width="150" />
<img src="https://github.com/rebarbara/CoGAN-demo/blob/master/train_24_0495.png" width="150" />
</p>
<p align="center">
<img src="https://github.com/rebarbara/CoGAN-demo/blob/master/train_00_0099 (1).png" width="150" />
<img src="https://github.com/rebarbara/CoGAN-demo/blob/master/train_01_0053 (1).png" width="150" />
<img src="https://github.com/rebarbara/CoGAN-demo/blob/master/train_04_0115 (1).png" width="150" />
<img src="https://github.com/rebarbara/CoGAN-demo/blob/master/train_24_0495 (1).png" width="150" />
</p>

Epochs 1,2,25
