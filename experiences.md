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


```python
def discriminator(self, image, y=None, share_params=False, reuse=False, name='D'):
    ...
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = prelu(conv2d(image, self.c_dim, name='d'+branch+'_h0_conv', reuse=False), 
					name='d'+branch+'_h0_prelu', reuse=False)
        h1 = prelu(d_bn1(conv2d(h0, self.df_dim, name='d'+branch+'_h1_conv', reuse=False), reuse=reuse), 
					name='d'+branch+'_h1_prelu', reuse=False)
        h1 = tf.reshape(h1, [self.batch_size, -1])            

    h2 = prelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin', reuse=share_params),reuse=share_params), 
					name='d_h2_prelu', reuse=share_params)
    h3 = linear(h2, 1, 'd_h3_lin', reuse=share_params)
            
    return tf.nn.sigmoid(h3), h3
```

```python
def generator(self, z, y=None, share_params=False, reuse=False, name='G'):
    ...
    s = self.output_size
    s2, s4 = int(s/2), int(s/4) 
    h0 = prelu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin', reuse=share_params), reuse=share_params), 
						name='g_h0_prelu', reuse=share_params)
    h1 = prelu(self.g_bn1(linear(z, self.gf_dim*2*s4*s4,'g_h1_lin',reuse=share_params),reuse=share_params),
						name='g_h1_prelu', reuse=share_params)
    h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])

    h2 = prelu(self.g_bn2(deconv2d(h1, [self.batch_size,s2,s2,self.gf_dim * 2], 
			name='g_h2', reuse=share_params), reuse=share_params), name='g_h2_prelu', reuse=share_params)

	with tf.variable_scope(name):
	    if reuse:
		    tf.get_variable_scope().reuse_variables()
 	    output = tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g'+branch+'_h3', reuse=False))

    return output
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
