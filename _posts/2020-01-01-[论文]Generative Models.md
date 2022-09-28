---
layout: post
comments: false
title: "[论文]Generative Models"
date: 2020-01-01 01:09:00
tags: paper-reading
---

> This post is a summary of synthesis related papers.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

## Generative Models

### 1. [Generative Adversarial Nets](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)
*Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio*

*NIPS 2014*

**1. Abstract**

1. The abstract of this paper is very standard and unique. This abstract generally introduces the main concept of this paper, i.e., GAN, rather than introducing the problem and existing methods. This abstract is kind of like the Wikipedia introduction of some concept. If you are very confident that your paper is very novel and proposes a very useful idea, model or concept that can be recorded in this area, you can use this kind of writing style of abstract. This is very clear to the community, but not very clear to the people that do not know any about this area.

**2. Introduction**

2. This paper thinks that besides deep learning architectures, deep learning will learn the representations of data probability distributions over all kinds of data, including natural images, audio waveforms containing speech, symbols in natural language processing, etc. This idea is always held by Yoshua Bengio and his group.

3. Discriminative models in deep learning have already invovled into various kinds and solved many kinds of problems. But generative models, instead, still have lots of part remained mysterious. This due to the fact that in order to learn data distributions underlying generative models, you need many approximation methods to approximate the distribution in order to make calculation work. But this process will make the distribution unaccurate and even not working. So in this paper, they do not try to approximate the distributions, they use deep learning models to do this job.

4. Generative adversarial models will contain two parts: the generative model and the discriminator model. The generative model aims to generate data that the discriminator model can not distinguish apart from true training data, and the discriminator model aims to distinguish between them. The final goal is to let the generative model to generative plausible data that the distriminator model can not distinguish.

5. In this paper, they use MLP as models for the generative and discriminator models, and in this case, this generative adversarial model is called generative adversarial net (GAN). The input of the generative model is random noise (usually a Guassion noise), and they want the generative model to map this random noise distribution to any distribution we want to fitting, i.e., the training data distribution.


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
