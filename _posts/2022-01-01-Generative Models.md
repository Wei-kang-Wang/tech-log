---
layout: post
comments: false
title: "Generative Models"
date: 2020-01-01 01:09:00
tags: paper-reading
---

> This post is a summary of synthesis related papers.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

## GAN

### [Generative Adversarial Nets](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)
*Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio*

*NIPS 2014*

这篇是GAN的开山之作。

GAN在很多领域都很火，包括image, NLP, music, audio等。GAN的核心思想来源于博弈论（game theory），其包括两个互相博弈/对抗的网络：generator和discriminator，它两互相博弈，互相被训练，互相成长，共同学习。现如今虽然已经是Diffusion model的世界了，但GAN模型很优雅，而且GAN的变种对如今的Diffusion的变种仍有启发意义。GAN已经可以被认为是生成模型的基础模型之一了。

GAN虽然很强，但也有着不少缺点，最重要的就是其难以训练（难以收敛、loss抖动）以及需要大量的训练数据。

**1. Kullback-Leiber Divergence（KL散度）和Jesen-Shannon Divergence（JS散度）**

KL散度和JS散度是用来定量描述两个概率分布之间距离的度量。

**(1). KL散度**

两个概率分布 $p(x)$ 和 $q(x)$之间的KL散度定义为：

$$D_{KL}(p \Vert q) = \int_{x} p(x) log \frac{p(x)}{q(x)} dx$$

$$D_{KL}(p \Vert q)$$有如下性质：

* 由于 $$-\int_{x} p(x) log \frac{p(x)}{q(x)} dx = \int_{x} p(x) log \frac{q(x)}{p(x)} dx = E_{p(x)} \left[ log \frac{q(x)}{p(x)} \right] \leq log E_{p(x)} \left[ \frac{q(x)}{p(x)} \right] = 0$$（最后一个不等号由Jesen不等式而来），从而$$D_{KL}(p \Vert q) \geq 0$$，其中等号在$$p(x)$$与$$q(x)$$处处相等时成立。

* $$D_{KL}(p \Vert q)$$并不是关于$$p(x)$$和$$q(x)$$对称的，即$$D_{KL}(p \Vert q)$$与$$D_{KL}(q \Vert p)$$并不一定相等。因此DL散度并不是一个数学上的metric（不满足交换律）。


**(2). JS散度**

JS散度也是描述两个概率分布之间的距离的度量，但JS散度是对称的，在这一点上比KL散度强。

JS散度的定义为：

$$D_{JS}(p \Vert q) = \frac{1}{2} D_{KL}(p \Vert \frac{p+q}{2}) + \frac{1}{2} D_{KL}(q \Vert \frac{p+q}{2})$$

JS散度满足如下性质：

* $$0 \leq D_{JS}(p \Vert q) = D_{JS}(q \Vert p) \leq 1$$

有理论认为GAN之所以成功就是因为理论上，GAN衡量真实数据分布和生成数据分布之间的度量用的是对称的JS散度，而并非那些之前的maximum likelihood方法所用的不对称的KL散度。

**2. GAN的结构**

GAN包括两个部分：
* 一个discriminator $$D$$，用来估计输入来自于真实数据集的概率。其优化目标是能够将真实数据与生成数据（即虚假数据）分隔开；
* 一个generator $$G$$，用来生成数据。其输入是一个确定的概率分布所采样得到的随机变量$$z$$，$$z \sim p(z)$$，其中$$p(z)$$是已知的，比如说某个高斯分布。$$z$$的作用是给$$G$$的输入带来随机性（因为$$z$$就是个随机变量），从而带来diversity。

在GAN的训练过程中，$$G$$和$$D$$会相互竞争：$$G$$要尽可能的想办法骗过$$D$$，而$$D$$要尽可能的区分真假数据。这是一个零和博弈过程，这个过程会同时训练$$D$$和$$G$$。

用$$p_z$$来表示输入噪声$$z$$的分布（通常已知），用$$p_g$$表示$$G$$所生成的样本的分布，用$$p_r$$表示真实数据的分布。

一方面，对于$$D$$的训练来说，其对于真实数据，要尽可能的给出大的概率，即$$\max_{D} E_{x \sim p_r(x)} log D(x)$$，对于生成数据，要尽可能的给出小的概率，即$$\max_{D} E_{z \sim p_z(z)} log(1-D(G(z)))$$，从而将这两个合在一起就得到了：

$$\max_{D} E_{x \sim p_r(x)} log D(x) + E_{z \sim p_z(z)} log(1-D(G(z)))$$

而另一方面，对于$$G$$的训练来说，其要尽可能的生成让$$D$$判断为真的生成数据，也就是：

$$\min_{G} E_{z \sim p_z(z)} log(1-D(G(z)))$$

将$$D$$的训练目标和$$G$$的训练目标合在一起，就得到了：

$$\min_{G} \max_{D} E_{x \sim p_r(x)} log D(x) + E_{z \sim p_z(z)} log(1-D(G(z)))$$

记 $$L(G,D) = E_{x \sim p_r(x)} log D(x) + E_{z \sim p_z(z)} log(1-D(G(z)))$$，其还可以写成：$$L(G,D) = E_{x \sim p_r(x)} log D(x) + E_{x \sim p_g(x)} log(1-D(x))$$。

我们考虑如下两个问题：

**(1). 对于$$\min_G \max_D L(G,D)$$来说，$$D$$的optimal是什么？**




**3. Related work**

1. There are two kinds of former works on deep generative models. The first ones concentrated on building a probability distribution function, and these models are trained on maximizing the log likelihood, such as the deep Boltzmann machine. The biggest difficulty is that the calculation of sampling this distribution is hard, especially when the dimension is high. The other ones are called generative machines. They do not explicity construct the distribution, and they learn a model to approximate this distribution. There are intrinsic differences between these two kinds of methods. The first ones acutally learn the distribution, though using some kind of approximation method to make the calculation feasible, but in the end, you actually get a distribution, you can calculate the mean, variance and all kind of properties of this distirbution. But the other ones do not construct the distribution, and only learn a model to approximate this distribution. So in the end, we do not know what this distribution looks like. 

2. Variational Autoencoder (VAE) actually has similar ideas to this paper. And using a distriminator model to assist the generative model is also not novel, such as Noise-contrastive Estimation (NCE). 


**4. Adversarial nets**

1. Generator wants to learn the distribution of the input training data $$x$$, $$p_g$$. We give an example of GAN. Suppose there is a video game and it can generate images of the game, and now we want to learn a generator to generate the images of the game. Suppose that our display resolution is 4K, then each time we need to generate an image vector of length 4k. Each pixel could be considered as a random variable, thus this vector can be considered as a multi-dimensional ramdon variable of length 4k. We know that this vector is controled by the underlying game code, and this code is actually the underlying $$p_g$$ for this vector. Now how to let the generator to generate data? We define a prior on the input noise variable $$p_z(z)$$, this $$z$$ could be a 100 dimensional Guassion distribution with mean 0 and variable matrix I. The generator aims to map $$z$$ onto $$x$$, the generator model can be formed as $$G(z, \theta_g)$$. Return to our game example. In order to generate game images, one way is that we can conversly compile the game code, and know the underlying game code. In this way, we can acutally know how the game images are generated. This method can be considered similar to the methods described in the related work that aim to construct the underlying distribution. Another way is that we neglect the underlying code, we guess that this game is not very complicated, thus maybe there are only 100 variables underlying are used to control the generation of images. Thus we contructed a known distribution of dimension 100 $$z$$, and due to the fact that MLP is able to approximate any functions, we let this MLP to map the input $$z$$ into the image space $$x$$. 

2. The discriminator $$D(x, \theta_d)$$ is also an MLP, and its output is a scalar value, for distinguishing between the true data and generated data. Thus actually $$D$$ is a two-label classifier. We know where our input data is from (true or generated), thus we can give them labels. 

3. The training process involves training D and G simultaneously:

$$\min_{G}\max_{D} V(D,G) = E_{x \sim p_{data}}\left[log D(x)\right] + E_{z \sim p_z(z)}\left[log(1-D(G(z)))\right]$$

This is a two-player minimax game. When the G and D reach a stable state, they are actually arrive at a Nash equilibrium.

4. Look at figure1. This example is simple. The input noise distribution of $$z$$ is a uniform distribution (the lower line of $$z$$ has equal intervals), and our $$x$$ is a Guassian distribution (the black dotted line). The arrows between the line of $$x$$ and $$z$$ is the mapping, i.e., generator. From plot (a) in figure1, we see that, at first, the mapping maps $$z$$ to the behind part of $$x$$, so the output distribution of this mapping is the green line in the plot, and the blue line is the discriminator. The next step, we update the discriminator, we can see that, the discriminator choose the margin in between of the mean of the true distribution and the generator output distribution, as shown in plot (b). Then we update the generator, we can see that the generator output distribution will move closer to the true distribution, in plot(c). And then we update discriminator, update generator, ..., and finally ,we get plot(d), in that the generaotr output will be the same to the true distribution and the discriminator will show a horizontal line indicating that it can not distinguish between true and generated data. 

![GAN]({{ '/assets/images/GAN-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 1. An example of training process of a GAN.*

5. Each training iteration of the training algorithm of GAN involves two steps. In the first step, there is a for loop, loops over $$m$$ times. And in each loop, we get $$m$$ true data and $$m$$ generated data from the generator, and then calculate the gradient of the minimax loss defined above to update the discriminator. In the second step, we sample another $$m$$ samples from the generator to calculate the gradient of the minimax loss with respect to the generator for updating it. The for loop iteration time $$k$$ in the first step, is a hyperparameter of this algorithm. In each training iteration, we need the generator and distriminator be at the same levle, i.e., the performance of one should be be much better than the other. Only in this case, we can make the training trackable. The decision of whether a GAN is trained well is also a difficult question, i.e., the iteration time of the training algorithm. This still remains a hot area and unsolved.

6. One training tip is that, since when the discriminator trains well, the $$log(1-D(G(z)))$$ will be close to 0, thus the gradient will not be applicable, instead of minimizing $$log(1-D(G(z)))$$, we maxmizing $$log(D(G(z)))$$.


**4. Theoretical Results**

1. There is a global optimum for the generator, $$p_g = p_{data}$$. Firstly, we see a lemma firstly. **Lemma**: if the generator is fixed, then the optimal discriminator will be

$$ D_{G}^{\*}(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)} $$

i.e., the error probability (the training criterion of discriminator) of the distriminator will be the smallest. **Theorem**: Setting $$ D_{G}^{\*}(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)} $$ as in Lemma in the equation $$\min_{G}\max_{D} V(D,G) = E_{x \sim p_{data}}\left[log D(x)\right] + E_{z \sim p_z(z)}\left[log(1-D(G(z)))\right]$$, we can get $$C(G) = E_{x \sim p_{data}}\left[log \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}\right] + E_{x \sim p_g}\left[log \frac{p_g(x)}{p_{data}(x) + p_g(x)} \right] $$. Then $$C(G)$$ get its minimum when $$p_g = p_{data}$$.

2. The algorithm described above is able to train the discriminator and the generator, i.e., the training algorithm is convergent. 


**5. Experiments**

The experiments in this paper is not good enough and quite simple.


**6. Conclusion**

GAN is actually an unsupervised learning setting, but it leverages supervised learning framework by using the label of true or generated data for training. This idea insights the future self-supervised learning frameworks.


**The conclusion of writing style of this paper**

This paper proposes a very novel idea and model, thus it elaborates the details of design and ideas behind GAN very clearly. The authors are very ambitious and are confident that this work will be recorded in this area. So the whole writing style is kind of like Wikipedia, i.e., the very detailed description of the proposed model, with a little mention of existing works and comparison with other works. Also, experiments are few.

If you are confident that your work is novel and very important, you can use this kind of writing style. Otherwise, you need to describe clearly the existing works and your contribution to this problem.


### 2. [Auto-Encoding Variational Bayes](https://openreview.net/forum?id=33X9fd2-9FyZd)

*ICLR 2014*


### 3. [Unsupervised 3D Shape Completion through GAN Inversion](https://graphics.stanford.edu/courses/cs348n-22-winter/PapersReferenced/Zhang%20et%20al.%20-%202021%20-%20Unsupervised%203D%20Shape%20Completion%20through%20GAN%20Inversion.pdf)

*CVPR 2021*


### 4. [EG3D: Efficient Geometry-aware 3D Generative Adversarial Networks](https://nvlabs.github.io/eg3d/)

[POST](https://nvlabs.github.io/eg3d/)

*CVPR 2022*


### 5. [CariGANs: Unpaired Photo-to-Caricature Translation](https://ai.stanford.edu/~kaidicao/carigan.pdf)

[POST](https://cari-gan.github.io/)

*SIGGRAPH Asia 2018*


### 6. [3DN: 3D Deformation Network](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_3DN_3D_Deformation_Network_CVPR_2019_paper.pdf)

[CODE](https://github.com/laughtervv/3DN)

*CVPR 2019*


### 7. [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_High-Resolution_Image_Synthesis_CVPR_2018_paper.pdf)

[POST](https://tcwang0509.github.io/pix2pixHD/)

*CVPR 2018*


### 8. [Polymorphic-GAN: Generating Aligned Samples Across Multiple Domains With Learned Morph Maps](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_Polymorphic-GAN_Generating_Aligned_Samples_Across_Multiple_Domains_With_Learned_Morph_CVPR_2022_paper.html)

*CVPR 2022*


### 9. [Self-Supervised Object Detection via Generative Image Synthesis](https://openaccess.thecvf.com/content/ICCV2021/html/Mustikovela_Self-Supervised_Object_Detection_via_Generative_Image_Synthesis_ICCV_2021_paper.html)

*ICCV 2021*


### 10. [Roof-GAN: Learning To Generate Roof Geometry and Relations for Residential Houses](https://openaccess.thecvf.com/content/CVPR2021/html/Qian_Roof-GAN_Learning_To_Generate_Roof_Geometry_and_Relations_for_Residential_CVPR_2021_paper.html)

*CVPR 2021*


### 11. [CLIP: Learning Transferable Visual Models From Natural Language Supervision](http://proceedings.mlr.press/v139/radford21a/radford21a.pdf)

*ICML 2021*


### 12. [MA-CLIP: Towards Modality-Agnostic Contrastive Language-Image Pre-training](https://openreview.net/forum?id=ROteIE-4A6W&referrer=%5Bthe%20profile%20of%20Yu%20Cheng%5D(%2Fprofile%3Fid%3D~Yu_Cheng1))

*Arxiv 2022*



## [Layout to image generation论文](https://paperswithcode.com/task/layout-to-image-generation)

这个是从scene graph生成image的benchmarks还有相关的论文的网页。

### 1. [Image Generation from Scene Graphs](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0764.pdf)

*CVPR 2018*






---
