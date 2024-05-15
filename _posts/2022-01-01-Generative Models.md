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

我们考虑如下三个问题：

**(1). 对于$$\min_G \max_D L(G,D)$$来说，$$D$$的optimal是什么？**

$$L(G,D) = E_{x \sim p_r(x)} log D(x) + E_{x \sim p_g(x)} log (1-D(x)) = \int_x (p_r(x) log D(x) + p_g(x) log (1-D(x)) dx$$

假设$$G$$给定，那么$$L(G,D)$$是关于函数$$D$$的一个泛函，记$$\tilde{x} = D(x), A=p_r(x), B=p_g(x)$$，那么上述积分号内部是关于$$\tilde{x}$$的一个泛函，记为$$f(\tilde{x}) = A log \tilde{x} + B log (1 - \tilde{x})$$。

让$$f(\tilde{x})$$对$$\tilde{x}$$求导，并令导数等于0，可得$$D^{\ast}(x) = \tilde{x}^{\ast} = \frac{A}{A+B} = \frac{p_r(x)}{p_r(x)+p_g(x)}$$，这也就是在$$G$$固定的时候，最优的$$D$$。当$$G$$训练的很好的时候，即$$p_r(x)$$和$$p_g(x)$$很接近，此时最优的判别器$$D(x)=\frac{1}{2}$$。

**(2). 之前定义的损失函数$$L(G,D)$$的全局最优是多少?**

当$$G$$最优时，$$p_r(x)=p_g(x)$$，而此时可知最优的$$D$$为$$D(x)=\frac{1}{2}$$，从而此时的loss值为：$$L(G^{\ast}, D^{\ast}) = -2log2$$。

**(3). 对损失函数$$L(G,D)$$的进一步分析**

$$D_{JS}(p_r \Vert p_g) = \frac{1}{2}(log4 + L(G, D^{\ast}))$$，即$$L(G, D^{\ast}) = 2D_{JS}(p_r \Vert p_g) - 2log2$$，也就是说，在判别器$$D$$是最优的情况下，损失函数实际上就是真实数据的分布与生成数据的分布之间的JS散度，从而损失函数的最小值为-2log2，在这两个分布相同的时候取到。

**3. GAN的不足之处**

* 难以训练，这个又可以分为以下几点：
    - 训练GAN很难达到Nash均衡。如果使用通常的SGD方法来训练$$G$$和$$D$$，那么其训练是独立的，从而同时训练它两的话，不一定能到到Nash均衡。
    - $$p_r(x)$$大部分有效值都定义在低维流形上，这是高维数据经常出现的问题。而$$p_g(x)$$由$$p_z(z)$$而来，$$p_z(z)$$是低维的，所以$$p_g(x)$$本质上由一个低维的向量$$z$$来控制，所以也在一个低维流形上，从而这两个流形很可能没有多少交集，这种情况下，$$D$$很容易区分这两个分布产生的数据。
    - 梯度消失。在$$D$$很好的时候，对于真实数据，$$D(x)$$约为1，对于生成数据，$$D(x)$$约为0，从而此时损失函数约为0，且损失函数的梯度也很小（尤其是对$$G$$的）。
  以上三点是GAN训练的困难，而且还构成了一个dilemma，如果$$D$$表现得很差，那么$$G$$就没有正确的反馈，从而难以正确的被训练，但如果$$D$$表现得很好，那么就会梯度消失，同样难以训练。
* Mode collapse。$$G$$可能会陷入一种情况，其一直输出差不多的样本，不论输入$$z \sim p_z(z)$$如何改变。这样的现象叫做mode collapse。尽管$$G$$在这种情况下可能可以骗过$$D$$，但此时$$G$$被困在了一个variety很小的空间里，无法表达复杂的数据。
* 缺乏好的evaluation metric。GAN的设计本身就不包括一个好的objective function来表示训练过程的好坏，即不知道何时该停止训练，以及如何比较各个GAN之间的好坏。

**4. 对GAN训练的改进**

基于以上的训练困难，前人们经过了很多经验总结，得出了以下几条经验，其中前五条用于使得GAN的训练能够更加快速的收敛，而后两条用来解决$$p_r(x)$$与$$p_g(x)$$可能位于低维流形上的问题。

* Feature matching。feature matching的意思是是训练$$D$$的时候，不仅要辨别$$p_r(x)$$和$$p_g(x)$$，还要辨别这两个分布的统计量，即在损失函数中加上一项$$\Vert E_{x \sim p_r(x)} f(x) - E_{x \sim p_g(x)} f(x) \Vert_2$$，其中$$f(x)$$可以是任意的统计量，比如median，mean等。 
* Minibatch Discrimination。对于$$D$$的训练来说，使用一个minibatch计算的loss来训练，而并非单个数据。
* Historical Averaging。对于$$G$$和$$D$$的参数，加上一项$$\Vert \theta_{t+1} - \frac{1}{t} \sum_{i=1}^t \theta_i \Vert_2$$在损失函数之中，其中$$\theta_t$$为当前step的某个参数，$$\theta_{i}$$为过去的时刻的参数，这样可以让参数的变化更平缓。
* One-sided Label Smoothing。对于真实数据和生成数据的标签，不要给成1.0和0.0，而是给成0.9和0.1，给一定的余量，可以增强网络的鲁棒性。
* Virtual Batch Normalization (VBN)。对于每个batch的输入，对其做batch normalization，但所用的mean和variance并非由该batch计算得来，而是由在训练之初就指定好的一组数据计算得来（也就是说mean和variance是不变的）。
* Adding noise。对于真实数据和生成数据，在其喂给$$D$$之前，加上已知的全局弥漫的噪声，比如高斯，这样使得两个数据分布不再位于低维流形中（这个方法也被用在diffusion model里面）。
* Use better metric of distribution similarity。由之前可知，在判别器$$D$$最优的时候，GAN的损失函数等价于$$p_r(x)$$和$$p_g(x)$$之间的JS距离，但在$$p_r(x)$$与$$p_g(x)$$的support交集很小的时候，JS散度没有意义。可以用之前的方法加噪声让它们的support的交集变大，还可以直接更换掉JS散度，用新的度量来设计损失函数，比如说Wasserstein distance。


### [Wasserstein GAN](https://arxiv.org/pdf/1701.07875)
*Martin Arjovsky, Soumith Chintala, Leon Bottou*

*ICML 2017*

Wasserstein GAN在推出的当时就得到了很多追捧，甚至Goodfellow也跟帖讨论，Wasserstein GAN（WGAN）成功的做到了以下几点：

* 彻底解决了GAN训练不稳定的问题，不再需要小心平衡生成器和判别器的训练程度了。
* 基本解决了Mode collapse的问题，确保了生成样本的多样性。
* 训练过程终于有了一个像交叉熵、准确率这样的数值来指示训练的进程，这个数值越小代表GAN训练的越好，代表生成器产生的图像质量越高。
* 并不需要精心设计的网络架构，普通的MLP就可以。

作者花了两篇论文来从头到尾来从GAN到WGAN，第一篇论文[Towards Principled Methods for Training Generative Adversarial Networks](https://arxiv.org/pdf/1701.04862)里面推导了理论结果，分析了GAN存在的问题，也针对性的给出了理论改进。第二篇论文[Wasserstein GAN](https://arxiv.org/pdf/1701.07875)，从理论改进推导出了更多的定理，从而给出了最终的改进算法。而且实际上最终的改进算法相对于原始的GAN只有以下四点改进：

* 判别器的最后一层去掉sigmoid函数。
* 生成器和判别器的loss不再取log了。
* 每次更新判别器参数的时候，将参数的绝对值截断到一个固定常数$$c$$以内。
* 不要用基于动量的优化算法来更新参数（比如momentum、Adam等），推荐RMSProp，SGD等。

WGAN算法如下：

![WGAN_algorithm](/assets/img/philly-magic-garden.jpg "WGAN Algorithm")

实际上这几个改动都很简单，但效果确实是非常好的，这是一个优秀的工作所应该有的：扎实的理论分析，细小的改动，巨大的结果提升。

以下从五个部分来分析WGAN：

* 原始的GAN的问题
* WGAN之前的一个过渡解决方案
* Wasserstein distance的优越性
* 从Wasserstein distance到WGAN
* 总结

WGAN源码实现：[martinarjovsky/WassersteinGAN](https://github.com/martinarjovsky/WassersteinGAN)






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
