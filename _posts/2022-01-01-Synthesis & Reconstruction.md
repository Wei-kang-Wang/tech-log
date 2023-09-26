---
layout: post
comments: false
title: "Synthesis & Reconstruction"
date: 2020-01-01 01:09:00
tags: paper-reading
---

> This post is a summary of synthesis related papers.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

## Image-based synthesis and reconstruction

### 各种从3D representations渲染2D图片的办法

* 在NeRF里，所有的相机都是置于某一个半球球面上，而场景假设位于半球的球心位置，也就是说，对于每一张view来说，相机focal length，以及scene大致距离相机的距离都是一样的，从而就没有了距离ambiguity的问题。而且对于每一张2D图片，长宽分别为$$W,H$$，从而2D图片上每一个像素点的坐标值就可以确定了，从而从该像素点到相机光心的射线就确定了，这样就可以沿着这条射线对neural radiance field function的每个空间坐标值得到的color和density进行积分，从而得到该2D图片像素点位置的预测值。

### 1. [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460392.pdf)

*Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng*

*ECCV*

![1]({{ '/assets/images/NERF-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

*fig1. 这篇文章提出一个方法从一系列图片来优化一个continuous的5D neural radiance field representation（在任何位置上的volume density和view-dependent color）去表示一个scene。本文使用volume rendering里的方法，从相机光心出发到空间中一点的一条射线来对这条射线上的点的色彩值进行积分，从而得到这个点从相机的这个角度project到2D图片上的RGB值，因而可以从任意相机角度来得到这个scene的2D彩色渲染图。上面的图可视化了100个Drums这个场景的views，其是被环绕的一个半球的区域内随机采样出来的。右侧展示了从已经学习好了的这个scene的NeRF representation里两个新角度渲染得到的照片。*

* 本文提出一个方法，使用同一个场景的一系列不同相机角度的input views作为输入，通过优化一个潜在的continuous volumetric scene函数来对复杂场景生成新的角度的views，乘坐NeRF。NeRF算法使用一个MLP（而不是CNN）来表示一个scene，其输入是一个连续的5D coordinate（空间位置$$(x,y,z)$$以及view的方向$$(\theta, \phi)$$（也就是相机角度）），输出是volume density和那个位置$$(x,y,z)$$的view dependent emitted radiance（也就是RGB颜色）。对于从相机光心射出的每一条射线，NeRF沿着这条射线对射线上每一个坐标点对应的color和density使用经典的volume rendering进行积分，从而得到这个相机角度的一张照片。因为volume rendering本身就是differentiable的，所以NeRF唯一的输入为一系列已知相机姿态的不同角度的照片，也就是一个场景不同角度的views。
* 这篇文章用一种全新的方法来解决view synthesis这个被研究了很久的问题，方法是，通过直接优化一个连续的5D scene representation的参数来最小化渲染images的error。可以看看related work来了解传统的view synthesis方法（2020年之前）
* 本文将一个静态的场景表示为一个连续的5D函数，输出空间中每个点$$(x,y,z)$$在每个角度$$(\theta, \phi)$$情况下的emitted radiance，以及一个density（这个emitted radiance就可以理解为这个点在这个角度下的RGB值，而density就是这个点在空间中的透明度，也就是在物体表面还是内部）。这个density作为一个可微分的opacity存在，其控制着一条射线穿过$$(x,y,z)$$这个点时，需要有多少的radiance被考虑进去。本文的方法优化一个MLP（没有任何卷积）来表示上述这个5D的function，其输入为一个5D coordinate $$(x,y,z,\theta,\phi)$$，输出为一个volume density和view-dependent RGB color，所以是一个regression问题。

![2]({{ '/assets/images/NERF-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*fig2. neural radiance field scene representation和differentiable rendering procedure的一个overview。我们通过沿着camera ray采样5D coordinates（location和viewing direction）来生成images。(a)将这些locations喂给一个MLP来输出color和volume density；(b) 使用volume rendering方法来将这些values composite到一个image里；(c) 这个rendering function是可微分的，所以我们可以通过最小化生成的images和ground truth images之间的差异来优化这个scene representation。*
* 为了从一个特定的视角来渲染这个neural radiance field（NeRF），我们需要：1)从相机发射rays穿过场景来采样一系列3D的点；2)使用这些点和对应的2D的角度输入作为5D输入给网络来输出每个点的color和density；3)使用经典的volume rendering方法来收集这些colors和densities，从而生成一张2D的照片。因为上述过程是可以微分的，所以我们可以使用梯度下降的方法通过最小化输入图片和网络生成图片两者之间的差异来优化这个模型。对于多个角度拍摄的这个场景的图片都进行上述最小化的操作，我们所学习到的这个网络就能够对于场景中真实存在物体的位置给出很高的volume density和准确的colors。上图给出了这整个过程的pipeline。

* 本文发现对于一个复杂的scene，用基础的方法来优化上述这个neural radiance field representation是不能让其收敛到很好的情况的，而且对于每个camera ray都需要很多的采样点。我们通过给原始的5D输入加上positional encoding，从而使得这个MLP能够表示更高频率的函数，来解决这个问题。我们还提出了一个hierarchical的采样方法来减少所需要的采样点的个数。

* 我们的方法继承了volumetric representation的优势：可以表示复杂的真实世界的geometry和appearance，而且很适合利用projected images来进行基于梯度的优化。而且，我们的方法克服了表示高分辨率的场景的时候所需要存储大量离散的voxel grids的问题。总而言之，这篇文章的贡献包括：（1）提出了一个将具有复杂geometry和appearance的连续的scene表示为5D neural radiance fields的方法，将其用MLP来表示； （2）基于经典的volume rendering方法的一个可微分的rendering方法，可以让我们通过RGB图片来优化上面提到的表示neural radiance field的MLP。而且还提出了一种hierarchical的采样方法；（3）提出一个positional encoding将每个5D输入map到一个更高维度的空间，从而使得优化能够表示更高频率的scene的neural radiance fields成为现实。

* 本文定性定量的表明我们的方法要比现在state-of-the-art的view synthesis方法要好，包括那些将3D representations fit到scene的工作，也包括那些训练CNN来预测采样的volumetric representations的工作。这篇论文提出了第一个能够从真实世界的scene或者objects的众多角度的RGB图片中学习到连续的neural scene representation，并且渲染出新的角度的高分辨率的photorealistic的图片的模型。

* 这篇文章的introduction写的非常的好，值得学习。


* NeRF这篇文章的related works

近期CV领域一个有前途的方向是将objects和scenes encode到一个MLP的参数里，也就是直接从一个3D的location映射到一个shape的隐式的representation，比如说在某个位置的signed distance：[A Volumetric Method for Building Complex Models from Range Images](https://dl.acm.org/doi/pdf/10.1145/237170.237269)。但是到目前为止，这些方法（利用MLP来表示representations）都无法做到基于discrete representations（比如triangle meshes，voxel grids等）的方法那样，生成十分逼真的图片。在这一节里，我们回顾一下两条line of research，并将它们与我们的方法进行对比，突出neural scene representations表达复杂场景并能渲染出逼真照片的能力。

一些相似的方法也是使用MLP来从低维的coordinates映射到colors，但这些方法用来表示的是其它的graphics functions，比如说images（[Compositional pattern producing networks: A novel abstraction of development]()），textured materials（[Learning a neural 3d texture space from 2d exexmplars]()，[Unified neural encoding of BTFs]()，[Neural BTF compression and interpolation]()）以及indirect illumination values（[Global illumination with radiance regression functions]()）。

*Neural 3D shape representations*

最近的工作通过优化一个将xyz坐标映射到signed distance functions（[Local implicit grid representations for 3d scenes]，[DeepSDF: Learning continuous signed distance functions for shape representation]）或者occupancy fields（[Local deep implicit functions for 3d shapes]，[Occupancy networks: Learning a 3D reconstruction in function space]）的神经网络，将连续的3D shapes当作level sets来研究其隐式表示。然而，这些方法都需要ground truth的3D geometry，一般都是从某些synthesis的3D shape dataset里获得的，比如说ShapeNet（[Shapenet: An introduction-rich 3d model repository]）。后续的工作放宽了需要3D shapes的这个要求，他们通过构造可微分的rendering functions来实现，而这些rendering functions允许只利用2D images来优化neural implicit shape representations。[Differentiable volumetric rendering: Learning implicit 3D representations without 3D supervision]()将surfaces表示为3D occupancy fields，并且使用了某种数值方法来找到每个ray和surface的相交点，然后利用隐式微分来计算该点的导数。每个ray和surface的焦点都是网络的输入，输出是该点的diffuse color。[Continuous 3D-structure-aware neural scene representations]()使用一个并不是那么直接的3D representation，输出是每个3D coordinate的一个feature向量和RGB color，并且利用一个由RNN组成的可微分的rendring function来沿着每条ray计算，从而判断surface位于什么为止。

尽管这些方法潜在的可以表示复杂的、高分辨率的geometry，但是它们现在只被用于有着低geometry复杂度的很简答的shape上，从而导致过于平滑的渲染效果。我们这篇文章表明使用5D输入（3D的位置，2D的相机角度），能够让网络表示高分辨率的geometry，并且能对复杂场景的新的view也生成photorealistic的照片。

*View synthesis and image-based rendering*

给定views的dense采样，很逼真的新的views可以直接被简单的light field sample interpolation方法构造出来，[The lumigraph]()，[Unstructured light fields]()和[Light field rendering]()。而对于基于稀疏的views的新的view synthesis，CV和graphics研究团体通过从images里预测传统的geometry和appearance representations做出了很大进步。一类很流行的方法使用scenes的mesh representations，而这些scenes有着diffuse或者view-dependent的appearance。可微分的rasterizers或者pathtracers可以直接优化这些mesh representations从而重构输入图片。然而基于image reprojection的对mesh的优化往往效果不好。而且，这个方法需要一个有着固定拓扑结构的template mesh，这对于真实世界的场景往往是获取不到的。

另一类比较流行的方法使用volumetric representations来解决从一系列RGB图片中学习到高质量的逼真的view synthesis问题。volumetric方法能够很逼真的表示复杂的shapes和materials，而且很适合用来做基于梯度下降的优化。早期的volumetric方法使用图像直接给voxel grids染色。对于更近期的一些方法（[DeepView: view synthesis with learned gradient descent]()，[Single-image tomography: 3d volumes from 2d cranial x-rays]()，[Learning a multi-view stereo machine]()，[Local light field fusion: Practical view synthesis with prescriptive sampling guidlines]()，[Soft 3D reconstruction for view synthesis]()，[Pushing the boundaries of view extrapolation with multiplane images]()，[Multi-view supervision for single-view reconstruction via differentiable ray consistency]()，[Stereo magnification: Learning view synthesis using multiplane images]()），他们使用多个场景的大的数据集来训练网络能够从输入的图像里预测volumetric representation，之后在测试阶段再利用alpha-compositing或者可学习的compositing，沿着rays来渲染出新角度的views。另一些工作对于每个scene联合优化一个CNN和sampled voxel grids，使得这个CNN可以对于低分辨率voxel grids中阐述的discretization artifacts进行补偿。

虽然上述这些volumetric方法对于novel view synthesis获得了很大的成功，但是他们对于更高分辨率的图片的生成受限于时间和空间复杂度——因为他们需要discrete sampling——渲染高分辨率的图片需要对3D空间进行更精细的采样。我们通过利用MLP来encode一个volume来解决这个问题，其不仅能生成更加逼真的图像，而且和这些sampled volumetric representations的方法相比，所需要的存储空间要小得多。


* Neural Radiance Field Scene Representation

我们将一个continous scene表示为一个5D vector-valued function，其输入是一个3D location，$$\pmb{x} = (x,y,z)$$和一个2D的view direction，$$(\theta, \phi)$$，输出是一个emitted color，$$\pmb{c} = (r,g,b)$$以及一个volume density $$\rho$$。在实践中，我们将方向表示为一个3D Cartesian单位向量$$\pmb{d}$$。我们用一个MLP来近似这个连续的5D scene representation：$$F_{\Theta}:(\pmb{x}, \pmb{d}) \longrightarrow (\pmb{c}, \rho)$$，并且优化这个MLP的weights，$$\Theta$$。

我们通过将volume density $$\rho$$仅仅表示为输入location $$\pmb{x}$$的函数，而输出的color $$\pmb{c}$$表示为输入location $$\pmb{x}$$和direction $$\pmb{d}$$的函数，来鼓励representation是multiview consistent的。为了实现这点，MLP $$F_{\Theta}$$首先利用8层MLP来处理输入的3D坐标$$\pmb{x}$$（每层256个通道，使用ReLU作为激活函数），然后输出$$\rho$$和一个256维的feature向量。这个feature向量再和相机角度$$\pmb{d}$$链接，喂给另一个MLP（每层128通道，使用ReLU作为激活函数）输出view-dependent RGB color。

fig4 表示，如果没有相机角度这个输入，仅仅利用$$\pmb{x}$$作为输入，那么表示高光反射的位置就会出现问题。


* Volume Rendering with Radiance Fields

我们的5D neural radiance field将一个scene表示为空间中任意一个点的volume density和directional emitted radiance。对于任意穿过这个scene的ray的渲染color，使用的是经典的volume rendering方法（[Ray tracing volume densities]()）。volume density $$\rho(\pmb{x})$$可以被理解为从相机出发的ray在点$$\pmb{x}$$处被拦下来的概率。相机ray $$\pmb{r}(t) = \pmb{o} + t \pmb{d}$$的color，$$C(\pmb{r})$$是：

$$C(\pmb{r}) = \int_{t_n}^{t_f} T(t)\rho(\pmb{r}(t))\pmb{c}(\pmb{r}(t), \pmb{d})dt$$

其中$$T(t) = exp(-\int_{t_n}^t \rho(\pmb{r}(s))ds)$$，而$$t_n$$和$$t_f$$是相机ray的起始和终止点。


>这个$$C(\pmb{r})$$表示的就是从某个相机角度出发的ray的平均color，积分内的值，可以理解为，$$T(t)$$表示从出发点到$$t$$位置为止积累的透明度，如果在碰到$$t$$以前就碰到了其它的透明度很低的点了（也就是真实的scene的点，$$\rho$$很大），那这个点之后的$$T$$都很小了，所以即使以后再遇到真实的scene的点，$$\rho$$很大，也不会将这个新遇到的点的color记进去的，这也符合事实，因为我们从一个角度来看scene的话，前面的点会挡住后面的点，那看到的只能是前面的点的颜色。

$$T(t)$$这个函数表示的是沿着ray从$$t_n$$到$$t$$积攒的透明度，也就是ray从$$t_n$$到$$t$$没有碰到任何particle的概率。从continuous neural radiance field渲染一个view，那么对于每个相机角度，假设有个虚拟的相机放在该处，则对于该相机的每个像素点都需要计算通过该像素点的ray的$$C(\pmb{r})$$的值。

我们使用quadrature来近似上述这个integral的值。deterministic quadrature，经常被用来渲染discretized voxel grids，会限制我们的representation的分辨率，因为这样color的值只会在一些固定的离散区域被考虑，这样MLP就只会拟合这些特定离散位置点的color，从而限制了更精确、分辨率更高的渲染。所以，我们使用了一种stratified sampling方法，将$$\left[t_n,t_f\right]$$分为$$N$$等份个bins，然后在每个bin里随机取一个点：

$$t_i \sim \mathcal{U} \left[t_n + \frac{i=1}{N}(t_f - t_n), t_n + \frac{i}{N}(t_f - t_n)\right]$$

尽管我们使用的是一个采样的离散集合来估计这个integral，stratified sampling使得我们可以表示一个continuous scene representation，因为加入了随机性使得MLP是在连续的位置上被优化的。我们使用这些采样点来估计$$C(\pmb{r})$$的值：

$$\hat C(\pmb{r}) = \Sigma_{i=1}^N T_i(1 - exp(-\sigma_i \delta_i)) \pmb c_i$$

其中$$T_i = exp(-\Sigma_{j=1}^{i-1} \sigma_j \delta_j)$$，$$\delta_i = t_{i+1} - t_i$$是两个相邻采样点之间的距离。上述这个从一个集合的$$(\pmb c_i, \sigma_i)$$来d计算$$\hat C(\pmb{r})$$的函数是可微的。


* Optimizing a Neural Radiance Field

在前面的章节里，我们描述了将一个scene建模为一个neural radiance field的核心内容，并介绍了如何从这个representation里渲染新的views。然而，这些对于生成state-of-the-art效果的图片来说并不够。我们介绍两个improvements来使得表示高分辨率的复杂scenes成为现实。第一个是对于输入coordinates的positional encoding，其可以帮助MLP拟合更高频率的函数（高频一般意味着更多的细节），第二个是使用了一种hierarchical的采样方法使得我们可以高效的采样这个高频的representation。


（1）Positional encoding

尽管神经网络是universal function approximators，我们发现将网络$$F_{\Theta}$$直接作用在$$xyz\theta \phi$$上会导致渲染的结果对于color和geometry的高频变化效果不好（也就是变化很频繁的区域，而这往往是细节区域）。这和[On the spectral bias of neural networks]()$$里的关于神经网络倾向于学习low frequency function的观点一致。他们还说在输入网络之前，将输入使用high frequency function映射到更高维的空间，会使得网络能学习到含有高频变化的部分。

我们将这个想法用在neural scene representations的设计里，我们将$$F_{\Theta}$$重写为两个functions的复合函数：$$F_{\Theta} = F_{\Theta}^{'} \circ \gamma$$，$$F_{\Theta}^{'}$$是需要被学习的，而$$\gamma$$不需要。这样操作会显著提升效果，由fig 4可以看出。这里$$\gamma$$是一个从$$\mathbb R$$到更高维空间$$\mathbb R^{2L}$$的映射，而
$$F_{\Theta}^{'}$$还是个MLP。我们使用的encoding function是：

$$\gamma(p) = (sin(2^0 \pi p), cos(2^0 \pi p), \cdots, sin(2^{L-1}\pi p), cos(2^{L-1}\pi p))$$

上述的$$\gamma$$对于输入的3D坐标$$\pmb{x}$$里的每个坐标分别使用（这些坐标在被$$\gamma$$操作之前就被归一化到了$$\left[-1,1\right]$$之间），对于表示相机角度的Cartesian viewing direction unit vecotr $$\pmb{d}$$的三个坐标也分别使用（这个unit vector的三个坐标本身就在$$\left[-1,1\right]$$之间了）。在这篇文章里，对于$$\pmb{x}$$，$$L=10$$，对于$$\pmb{d}$$，$$L=4$$。

在Transformer里也用到了相似的encoding的方式，在那里叫做positional encoding。但是Transformer用它的原因和我们这里的截然不同（它是为了给无法识别顺序的网络框架输入的sequence的顺序信息）。而我们这里用到的原因是通过将输入映射到高维空间里，使得MLP能够更加轻易的拟合高频函数。

![3]({{ '/assets/images/NERF-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 3 view-dependent emitted radiance的一个可视化。我们的neural radiance field representation输入为3D locations和2D viewing direction，输出为RGB color。这里，我们对于Ship scene，在neural representation里可视化两个位置的directional color distributions。在(a)和(b)里，我们展示了两个3D points在不同camera角度下的appearance：一个3D points是船舷，另一个是水面。我们的方法预测了这两个点在不同相机角度下的appearance，在(c)里我们显示了在viewing directions构成的半球上，每个3D points的appearance是怎么连续变化的。*

![4]({{ '/assets/images/NERF-4.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 4 这里我们展示了我们的方法是如何从view-dependent emitted radiance以及positional encoding上受益的。将view-dependent移除，那我们的模型就无法预测那些镜面反射的点。将positional encoding移除，我们的模型对于那些高频的geometry和texture的预测就会很差，从而导致一个过于光滑的appearance。*

（2）Hierarchical volume sampling

我们的渲染策略是沿着每个camera ray稠密的采样$$N$$个query points，来evaluate neural radiance field network。这样的方式是不高效的：没有物体的free space以及被遮住的occluded regions，它们对于渲染的图片没有贡献，但却被重复的采样了很多次。我们从之前的volume rendering的论文里获取了灵感，提出了一个通过根据正比于点在最终渲染的图片上的贡献来采样点的方式提高渲染效率的策略。

和仅仅用一个网络来表示scene不一样，我们同时使用两个网络，一个coarse，一个fine。我们先利用stratified sampling的方式采样$$N_c$$个点，然后用第四节里提到的方法来evaluate这个coarse网络。我们将coarse网络里活得的alpha composited color $$\hat C_c (\pmb{r})$$表示为沿着这个ray的所有采样点的colors $$c_i$$的加权和：

$$\hat C_c (\pmb{r}) = \Sigma_{i=1}^{N_c} w_i c_i$$

其中$$w_i = T_i ( 1 - exp(-\sigma_i \delta_i))$$。

将这些weights归一化为：$$\hat w_i = w_i / \Sigma_{j=1}^{N_c} w_j$$，就产生了一个沿着ray的piecewise-constant PDF。之后再从这个distribution上采样$$N_f$$个点，最后利用第四届里的方法使用所有的$$N_f + N_c$$个采样点来计算$$\hat C_f (\pmb{r})$$。这个方法为我们认为会有更多有用信息的位置增加了更多的点。


* Implementation details

我们为每个scene优化一个单独的neural continuous volume representation network。我们仅仅需要一系列该scene的RGB图片、每张图片对应的相机角度和intrinsic parameters，以及scene bounds（对于生成数据，我们有相机角度、intrinsic parameters和scene bounds的ground truth，而对于真实数据，我们使用COLMAP structure-from-motion package（[Structure-from-motion revisited]()）来估计这些参数）。在每个optimization iteration，我们从数据集里所有的pixels里随机采样一个batch的camera rays，然后利用第五节里说的方法query $$N_c$$个点给coarse network，query $$N_c + N_f$$个点给fine network。我们之后再用第四节里说的volume rendering procedure来为这两个点的集合渲染出这条ray的color。我们的loss仅仅是渲染出来的颜色和真实的pixel颜色之间的$$L_2$$距离：

$$\mathcal L = \Sigma_{\pmb r in \mathcal R} \left[ \lVert \hat C_c (\pmb{r}) - C(\pmb r) \rVert_2^2 + \lVert \hat C_f (\pmb{r}) - C(\pmb r) \rVert_2^2 \right]$$

其中$$\mathcal R$$是每个batch里的rays，$$C(\pmb r)$$，$$C_c (\pmb r)$$和$$C_f(\pmb r)$$分别是ray $$\pmb r$$的ground truth，coarse volume predicted和fine volume predicted RGB颜色。我们同样在loss里加入了coarse network，因为这样才能使得采样的distribution是有效的。

在真实的实验里，我们的batch size是4096 rays，$$N_c = 64$$，$$N_f = 128$$。我们使用了Adam优化器，learning rate开始设置为$$5 \times 10^{-4}$$，decay是 exponentially设置为$$5 \times 10^{-5}$$。其余的Adam参数为默认值。对于一个scene，使用单张NVIDIA V100 GPU需要循环100-300K个循环，需要1-2天的时间。


* 尽管我们提出了一个hierarchical sampling的方法使得渲染更加的高效，但在高效优化neural radiance field和采样的方向还有很多工作要做。另一个将来的方向是interpretability：sampled representations比如说voxel grids和meshes允许推理和思考所渲染的图片的效果以及失败的情况，但是将scenes表示为MLP的参数之后，我们就无法分析这些了。这篇文章推动了基于现实世界的graphics pipeline的发展，因为真实的objects和scenes现在可以被表示为neural radiance fields了，而多个objects或者scenes可以组合称为更复杂的scenes。



### [GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis](https://proceedings.neurips.cc/paper/2020/file/e92e1b476bb5262d793fd40931e0ed53-Paper.pdf)

*Nerf $$\rightarrow$$ GRAF*

*NeurIPS 2020*

[CODE](https://github.com/autonomousvision/graf)

*Katja Schwarz, Yiyi Liao, Michael Niemeyer, Andreas Geiger*

**Abstract**

尽管2D的generative adversarial networks已经可以进行高分辨率的image synthesis了，但是其并不具备对三维世界的理解以及从三维场景生成二维图片生成的能力。因此，它们并没有明确控制相机角度或者物体的姿态。为了解决这个问题，最近有一些方法使用了将基于voxel的representations和differentiable rendering结合起来的方式。然而，现存的方法要么就是只能生成低分辨率的图片，要么就是无法将相机角度和场景特性相分开（比如说场景内的物体）。这篇文章为radiance fields提出了一个generative model。radiance fields方法最近对于单个场景的novel view synthesis效果很成功（也就是Nerf）。和基于voxel representations的方法相比，radiance fields并不仅仅是3D空间的一个粗糙的离散化（discretization），还具有能够分离相机姿态和场景特性的能力。通过引入一个multi-scale的基于patch的discriminator，作者可以仅仅通过训练那些没有相机参数的2D图片来生成新的角度的这个场景的高分辨率的图片。作者在几个具有挑战性的synthetic和real-world数据集上系统性的验证了所提出的方法。大量的实验证明了radiance fields对于generative image synthesis来说是一个很有力的representation，从而可以生成3D场景的高保真的2D图片。


**1. Introduction**

convolutional generative adverarial networks（也就是GAN）已经证明了它从无结构化信息的图片中生成高质量图片方面的能力。然而，尽管GAN很成功，如果输入信息里有3D shapes信息和相机参数信息，它仍然无法将它们解耦开。这和人类能够推理出场景的3D结构以及能够想象出场景的新的角度的照片的能力形成了对比。

因为推理3D结构对于robotics，visual reality等领域都有着很重要的影响，最近一些工作开始考虑3D-aware image synthesis，其目标是通过对相机角度明确的控制，获得十分真实的图片生成。和2D GAN相反的是，3D-aware image synthesis学习到一个3D的scene representation之后，使用differentiable rendering的方式渲染出2D图片，因此这个方法对于场景内容和相机参数都有控制。因为3D的监督信息以及2D图片的相机角度信息比较难以获得，所以最近有一些工作尝试仅仅通过2D图片作为监督信息来进行3D场景的学习以及新角度的图片生成。为了实现这个目标，现有的方法使用了离散化的3D representations，也就是基于voxel的representations，既包括3D objects也包括3D features，如fig1所示。尽管这种方式可以实现后续的differentiable rendering，但是基于voxel的representations所需要的存储空间是随着场景尺寸三倍成长的，所以一般只能生成低分辨率的图片。基于voxel的3D feature representations在这方面会好一点，可以允许更高分辨率的图片生成，但是这需要学习一个从3D到2D的从features到RGB值得映射，这会导致模型学习到了一些在不同views之间不连续的特征。

这篇文章认为上述所说的低分辨率的生成图片和entangled的特征可以被使用conditional radiance field所解决，也就是Nerf的一个conditional版本。更准确地说，作者做出了如下的贡献：（1）提出了GRAF，一个generative model来为radiance field从没有相机角度的各个角度的某个场景的图片集合生成高分辨率的3D-aware的新的角度的图片。除了能够生成新的角度的图片，这种方法还可以让我们对场景内物体的shape和appearance做出个性化的editing。（2）作者提出了一个基于patch的discriminator，其会采样同一张图片不同尺度的图片，这也就是作者提出的方法能够很高效的学到高分辨率generative radiance field的诀窍。（3）作者在生成和真实的数据集上都验证了所提出的方法的效果。结果和目前sota的方法都是差不多的。


**3. Method**

作者考虑3D-aware image synthesis这个任务，也就是在提供相机角度的情况下生成场景高质量的图片。





### 2. [KeypointNeRF: Generalizing Image-based Volumetric Avatars using Relative Spatial Encoding of Keypoints](https://arxiv.org/pdf/2205.04992.pdf)

[POST](https://markomih.github.io/KeypointNeRF/)

*Marko Mihajlovic, Aayush Bansal, Michael Zollhoefer, Siyu Tang, Shunsuke Saito*

*ECCV 2022*


**Abstract**

使用像素对齐（pixel-aligned）的features来做基于图片的体素avatars（image-base volumetric avatars）能保证结果可以泛化到没有见过的实体和姿态上去。先前的工作全局空间编码（global spatial encoding）和多角度的geometric consistency（multi-view geometric consistency）来减少空间不确定性（spatial ambiguity）。然而，global encodings经常会对训练数据overfitting，而从稀疏的几个views之间也很很难学习到multi-view consistency。在这篇文章里，我们研究了现有的spatial encoding方法的问题，并提出了一个简单但是很有效的方法来从稀疏的views（也就是只有几张不同角度的照片）来构造高保真的（high-fidelity）volumetric avatars。一个关键的想法就是使用稀疏的3D Keypoints来编码相对的空间3D信息。这个方法对于角度的稀疏性以及跨数据集之间的差异性是鲁棒的。我们的方法在head reconstruction领域超过了sota的方法的效果。对于未知身份的human body reconstruction，我们也达到了和之前工作差不多的效果，而之前的工作是通过使用参数化的human body模型以及使用时序feature aggregation实现的。我们的实验证明之前的那些方法之所以会效果不好，很大程度上是因为spatial encoding的方式选的不好，因此我们提出了一个构造基于图片的高保真的avatars的新方向。


**1. Introduction**

可渲染的3D human representations对于social telepresence，mixed reality，entertainment platforms等来说都是很重要的一个部分。经典的基于mesh的方法需要空间稠密的多个角度的立体数据（dense multi-view stereo）：[]()，[]()，[]()；或者需要融合深度信息：[]()。这些方法的结果的保真性不高，因为获取准确的geometry reconstruction的难度很大。最近，neural volumetric representations（[Neural Volumes: Learning Dynamic Renderable Volumes from Images](https://stephenlombardi.github.io/projects/neuralvolumes/)，[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460392.pdf)）的提出使得高保真的avatar reconstruction成为了可能，特别是当准确的几何特性无法获得的时候（比如说头发的几何特性）。通过加入human-specific parametric shape models，获取大批的不同角度的数据只需要通过简单的设置相机参数就可以通过模型来获得了。然而，这些基于学习的方法是subject-specific的（也就是说每个模型只能针对某个具体的场景，因为实际上模型并没有学习到场景内到底有什么物体以及它的特性等这种high-level的semantic information，Nerf学习到的是一个不具有semantic information的整个空间的基于角度的颜色感知），而且需要对于每个场景训练好几天的时间，这样就大大限制了模型的可扩展性（scalability）。如果想要volumetric avatars迅速普及，需要能够做到，用户拿着手机给他自己从几个角度来几张自拍，传入网络里，然后很快就能构建出一个avatar出来。为了达到这个目标，我们需要能够从两三个views里学到准确的基于图像的volumetric avatars。

fully convolutiona pixel-aligned features（也就是说输出的feature map和输入的图片的尺寸是一样的，从而feature map每个位置的各个通道组成的feature就代表输入图片这个像素点位置的feature）使用了多个尺度的信息，从而让很多2D的CV任务效果都变好了（包括[PixelNet: Representation of the pixels, by the pixels, and for the pixels](https://arxiv.org/pdf/1702.06506.pdf)，[Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://openaccess.thecvf.com/content_cvpr_2017/papers/Cao_Realtime_Multi-Person_2D_CVPR_2017_paper.pdf)，[Fully Convolutional Networks for Semantic Segmentation
](https://openaccess.thecvf.com/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)，[Stacked Hourglass Networks for
Human Pose Estimation](https://islab.ulsan.ac.kr/files/announcement/614/Stacked%20Hourglass%20Networks%20for%20Human%20Pose%20Estimation.pdf)）。应用在occupancy和texture领域（[Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization]()）以及neural radiance fields领域（[Neural human performer: Learning generalizable radiance fields for human performance rendering]()，[Pva: Pixel-aligned volumetric avatars]()，[pixelnerf: Neural radiance fields from one or few images]()）的pixel-aligned representations使得这些方法能够泛化到没有见过的主体上去。pixel-aligned neural fields通过一个像素位置（pixel location）和空间编码函数（spatial encoding function）来推断field quantities，加入spatial encoding function是为了避免ray-depth不确定性。在这篇文章里，各种不同的spatial encoding functions都被研究了：[Portrait neural radiance fields from a single image]()，[Arch++: Animation-ready clothed human reconstruction revisited]()，[Arch: Animatable reconstruction of clothed humans]()，[Pva: Pixel-aligned volumetric avatars]()。然而，spatial encoding的效用并没有被完全理解。在这篇文章里，我们详细分析了使用pixel-aligned neural radiance fields方法为human faces建模的spatial encodings方法。我们的实验证明spatial encoding的选择影响了reconstruction的效果，也影响了模型泛化到未见过的人或者未见过的角度上的效果。那些使用了深度信息的模型会容易过拟合，过分拟合训练数据的分布，而使用multi-view stereo约束的方法对于稀疏的views的情况的效果就很不好了。

我们提出一个简单但是有效的方法利用了稀疏的3D keypoints来解决上述这些方法里存在的问题。3D keypoints可以很简单的用一个训练好的2D keypoints detector，比如说[Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://openaccess.thecvf.com/content_cvpr_2017/papers/Cao_Realtime_Multi-Person_2D_CVPR_2017_paper.pdf)，和简单的multi-view triangulation来计算而得，[Nultiple view geometry in computer vision](http://www.r-5.org/files/books/computers/algo-list/image-processing/vision/Richard_Hartley_Andrew_Zisserman-Multiple_View_Geometry_in_Computer_Vision-EN.pdf)。我们将3D keypoints看作spatial anchors，然后将其它部分的相对3D空间信息（relative 3D spatial information）都编码到这些keypoints上去。和global spatial encoding（[Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization]()，[Pifuhd: Multi-level pixel-aligned implicit function for high-resolution 3d human digitization]()，[pixelnerf: Neural radiance fields from one or few images]）不同，relative spatial encoding并不需要知道相机参数。这个特性使得基于relative spatial encoding的方法对于姿态（pose）来说很鲁棒。3D keypoints还让我们可以使用同样的方法来处理faces和bodies。我们的方法使得我们可以从两三个views上生成volumetric facial avatars，而且效果是sota的，而且提供更多的views会让效果更好。我们的方法同样在human body reconstruction这个任务上和Neural Human Performer(NHP)（[Neural human performer: Learning generalizable radiance fields for human performance rendering]()）达到了差不多的效果。NHP依赖于准确的参数化的human body模型，以及时序的feature aggregation，而我们的方法仅仅需要3D keypoints。我们的方法同时也不过过拟合训练数据的分布。我们可以将训练好的模型（不需要任何微调）对于从未见过的人的照片，生成其的avatars。我们认为这种能够对没见过的数据（人脸）生成volumetric avatars的能力是源于我们对于spatial encoding的选择。我们的主要贡献如下：

* 一个简单的利用3D keypoints的框架，可以从两三个不同角度的输入图片生成volumetric avatars，而且还可以泛化到从未见过的人。
* 对于已有的spatial encodings进行了一个细致的分析。
* 我们的模型在一个数据集上训练好之后，对于任何其他的没有见过的人的几个角度的照片，仍然可以生成volumetric avatars，这样好的泛化效果是之前的工作所没有的。


**2. Related work**

我们的目标是从很少的views（甚至两张）来构造高保真的volumetric avatars。

*Classical Template-based Techniques* 

早期的avatar reconstruction需要大量的图片，以及将一个template mesh匹配到3D point cloud等操作。[Real-time facial animation with image-based dynamic avatars]()提出使用一个粗造的geometry来为face建模，并使用一个可变性的hair model。[Avatar digitization from a single image for real-time rendering]()从一个数据集里获取hair model的信息，并组合facial和hair的细节信息。[Video based reconstruction of 3d people models]()使用基于轮廓的建模方法基于一个单目的视频来获取一个body avatar。对于geometry和meshes的以来限制了这些方法的发挥，因为某些区域比如hair，mouth，teeth等的geometry非常难以获得。


*Neural Rendering*

neural rendering（[State of the Art on Neural Rendering](https://arxiv.org/pdf/2004.03805.pdf)，[Advanced in neural rendering](https://arxiv.org/pdf/2111.05849.pdf)）通过直接从原始的sensor measurements里学习image formation的组成部分来解决那些template-based的方法解决不了的问题。2D neural rendering方法，比如DVP（[Deep Video Portraits](https://arxiv.org/pdf/1805.11714.pdf)），ANR（[Anr: Articulated neural rendering for virtual avatars](https://openaccess.thecvf.com/content/CVPR2021/papers/Raj_ANR_Articulated_Neural_Rendering_for_Virtual_Avatars_CVPR_2021_paper.pdf)），SMPLpix（[Smplpix: Neural avatars from 3d human models](https://arxiv.org/pdf/2008.06872.pdf)）使用了surface rendering和U-Net结构来减小生成的图片和真实图片之间的差距。这些2D方法的一个缺点是它们无法以一种时序紧密联系的方式（temporally coherent manner）来生成图片里物体新的角度的图片。Deep Appearance Models（[Deep appearance models for face rendering]()）联合使用了一个粗造的3D mesh和基于角度的texture mapping来从稠密的各个角度的images以监督学习的方式来学习face avatars。使用一个3D的mesh极大地帮助了新视角生成，但是这个方法在某些区域就很难生成真实的效果好的图片，比如说hair或者mouth里面，因为这些地方的3D mesh本身就缺乏对应的几何信息，因为根本就很难获取。现在的sota方法比如说NeuralVolumes（[Neural volumes: learning dynamic renderable volumes from images]()）和Nerf（[Nerf: Representing scenes as neural radiance fields for view synthesis]()）使用了differentiable volumetric rendering而不是meshes。这些方法能够在那些很难估计3D geometry的区域仍然有好的效果。之后还有更多的改进版本进一步改善了效果：[Flame-in-nerf: Neural control of radiance fields for free view face animation]()，[Mixture of volumetric primitives for efficient neural rendering]()，[Learning compositional radiance fields of dynamic human heads]()。但这些方法需要稠密的multi-view的图片，并且一个模型只针对一个场景，需要训练好几天才能有好的效果。


*Sparse View Reconstruction*

大规模的模型部署需要用户提供两到三张不同角度的照片，就可以生成avatars。通过使用pixel-aligned features（[Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization]()，[Pifuhd: Multi-level pixel-aligned implicit function for high-resolution 3d human digitization]()），我们就可以训练这样的网络了：从几个views来生成avatars的网络。各种不同的方法都将multi-view约束、pixel-aligned features与NeRF结合起来来学习可泛化的view-synthesis：[Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo]()，[Stereo radiance fields(srf): Learning view synthesis from sparse views of novel scenes]()，[Ibrnet: Learning multi-view image-based rendering]()，[pixelnerf: Neural radiance fields from one or few images]()。在这篇文章里，我们会发现这些方法对于没见过的human face和bodies很难只从几张views就生成细节很好的结果。


*Learning Face and Body Reconstruction*


### 3. [3D Moments from Near-Duplicate Photos](https://3d-moments.github.io/static/pdfs/3d_moments.pdf)

[POST](https://3d-moments.github.io/)

*CVPR 2022*

### 4. [NeuMesh: Learning Disentangled Neural Mesh-based Implicit Field for Geometry and Texture Editing](https://arxiv.org/pdf/2207.11911.pdf)

[POST](https://zju3dv.github.io/neumesh/)

*ECCV 2022 Oral*

### 5. [Eikonal Fields for Refractive Novel-View Synthesis](https://eikonalfield.mpi-inf.mpg.de/paper/eikonalfield_2022.pdf)

[POST](https://eikonalfield.mpi-inf.mpg.de/)

*SIGGRAPH 2022*


### 6. [Block-NeRF: Scalable Large Scene Neural View Synthesis](https://waymo.com/intl/zh-cn/research/block-nerf/)

[POST](https://waymo.com/intl/zh-cn/research/block-nerf/)

*CVPR 2022*

### 7. [AutoRF: Learning 3D Object Radiance Fields from Single View Observations](https://sirwyver.github.io/AutoRF/)

[POST](https://sirwyver.github.io/AutoRF/)

*CVPR 2022*


### 8. [GRAM: Generative Radiance Manifolds for 3D-Aware Image Generation](https://openaccess.thecvf.com/content/CVPR2022/papers/Deng_GRAM_Generative_Radiance_Manifolds_for_3D-Aware_Image_Generation_CVPR_2022_paper.pdf)

*CVPR 2022*


### 9. [NeRFRen: Neural Radiance Fields with Reflections](https://bennyguo.github.io/nerfren/)

[POST](https://bennyguo.github.io/nerfren/)

*CVPR 2022*


### 10. [Learning Object-Compositional Neural Radiance Field for Editable Scene Rendering](https://zju3dv.github.io/object_nerf/)

[POST](https://zju3dv.github.io/object_nerf/)

*ICCV 2021*


### 11. [ARF: Artistic Radiance Fields](https://www.cs.cornell.edu/projects/arf/)

[POST](https://www.cs.cornell.edu/projects/arf/)

*ECCV 2022*


### 12. [StylizedNeRF (Jittor): Consistent 3D Scene Stylization as Stylized NeRF via 2D-3D mutual learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_StylizedNeRF_Consistent_3D_Scene_Stylization_As_Stylized_NeRF_via_2D-3D_CVPR_2022_paper.pdf)

[CODE](https://github.com/IGLICT/StylizedNeRF)

*CVPR 2022*


### 13. [Unsupervised Novel View Synthesis from a Single Image](https://arxiv.org/pdf/2102.03285.pdf)

*Arxiv 2021*


### 14. [TensoRF: Tensorial Radiance Fields](https://arxiv.org/pdf/2203.09517.pdf)

[POST](https://apchenstu.github.io/TensoRF/)

*ECCV 2022*


### 15. [NeRF in the wild: Neural Radiance Fields for Unconstrained Photo Collections](https://arxiv.org/pdf/2008.02268.pdf)

[POST](https://nerf-w.github.io/)


### 16. [Single-view View Synthesis with Multiple Images](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tucker_Single-View_View_Synthesis_With_Multiplane_Images_CVPR_2020_paper.pdf)

[POST](https://single-view-mpi.github.io/)


### 17. [SynSin: End-to-end View Synthesis from a Single Image](https://arxiv.org/pdf/1912.08804.pdf)

[POST](https://www.robots.ox.ac.uk/~ow/synsin.html)

*CVPR 2020 Oral*


### 18. [NeX: Real-time View Synthesis with Neural Basis Expansion](https://arxiv.org/pdf/2103.05606.pdf)

[POST](https://nex-mpi.github.io/)

*CVPR 2021 Oral*


### 19. [pixelNeRF: Neural Radiance Fields from One or Few Images](https://arxiv.org/pdf/2012.02190.pdf)

[POST](https://alexyu.net/pixelnerf/)

*CVPR 2021*


### 20. [GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields](http://www.cvlibs.net/publications/Niemeyer2021CVPR.pdf)

[POST](https://m-niemeyer.github.io/project-pages/giraffe/index.html)

*CVPR 2021 Best Paper*


### 21. [NeRFReN: Neural Radiance Fields with Reflections](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_NeRFReN_Neural_Radiance_Fields_With_Reflections_CVPR_2022_paper.html)

[POST](https://bennyguo.github.io/nerfren/)

*CVPR 2022*


### 22. [Pix2NeRF: Unsupervised Conditional $$\pi$$-GAN for Single Image to Neural Radiance Fields Translation
](https://openaccess.thecvf.com/content/CVPR2022/papers/Cai_Pix2NeRF_Unsupervised_Conditional_p-GAN_for_Single_Image_to_Neural_Radiance_CVPR_2022_paper.pdf)

*CVPR 2022*


### 23. [IBRNet: Learning Multi-View Image-Based Rendering](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_IBRNet_Learning_Multi-View_Image-Based_Rendering_CVPR_2021_paper.pdf)

[POST](https://ibrnet.github.io/)

*CVPR 2021*


### 24. [Infinite Nature: Perpetual View Generation of Natural Scenes from a Single Image](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Infinite_Nature_Perpetual_View_Generation_of_Natural_Scenes_From_a_ICCV_2021_paper.pdf)

[POST](https://infinite-nature.github.io/)

*ICCV 2021 Oral*


### 25. [MINE: Towards Continuous Depth MPI with NeRF for Novel View Synthesis](https://arxiv.org/pdf/2103.14910.pdf)

[POST](https://vincentfung13.github.io/projects/mine/)

*ICCV 2021*


### 26. [Structure-aware Editable Morphable Model for 3D Facial Detail Animation and Manipulation](https://arxiv.org/pdf/2207.09019.pdf)

*ECCV 2022*


### 27. [Pose2Room: Understanding 3D Scenes from Human Activities](https://arxiv.org/pdf/2112.03030.pdf)

[POST](https://yinyunie.github.io/pose2room-page/)

*ECCV 2022*


### 28. [One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_One-Shot_Free-View_Neural_Talking-Head_Synthesis_for_Video_Conferencing_CVPR_2021_paper.html)

*CVPR 2021*


### 29. [Transferring Dense Pose to Proximal Animal Classes](https://openaccess.thecvf.com/content_CVPR_2020/html/Sanakoyeu_Transferring_Dense_Pose_to_Proximal_Animal_Classes_CVPR_2020_paper.html)

*CVPR 2020*


### 30. [DeepI2I: Enabling Deep Hierarchical Image-to-Image Translation by Transferring from GANs](https://proceedings.neurips.cc/paper/2020/hash/88855547570f7ff053fff7c54e5148cc-Abstract.html)

*NeurIPS 2020*


### 31. [LatentKeypointGAN: Controlling GANs via Latent Keypoints](https://arxiv.org/pdf/2103.15812.pdf)

[POST](https://xingzhehe.github.io/LatentKeypointGAN/)

*Arxiv 2022*


### 32. [TransGaGa: Geometry-Aware Unsupervised Image-To-Image Translation](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_TransGaGa_Geometry-Aware_Unsupervised_Image-To-Image_Translation_CVPR_2019_paper.html)

*CVPR 2019*

[POST](https://wywu.github.io/projects/TGaGa/TGaGa.html)

**Abstract**

无监督的image-to-image translation的目标在于在两个视觉domains之间学习到一个映射。然而，因为这两类图片之间具有很大的geometry variations，这样的框架的学习往往都是失败的。在这篇工作里，作者介绍了一个新的disentangle-and-translate框架来解决这个复杂的image-to-image translation任务。这个方法并不会直接学习image spaces之间的映射，而是将image space解耦为appearance和geometry latent spaces的笛卡尔积。更严格地说，作者先介绍了一个geometry prior loss和一个conditional VAE loss来让网络能学习到相互独立且相互补充的representations。然后，在appearance和geometry space上分别独立进行image-to-image的translation任务。大量的实验证明了本文提出的方法达到了sota的效果，尤其是对于那些non-rigid的object的translation任务。而且如果用不同的数据作为appearance space，本文的方法还可以实现多模态之间的translation。


**1. Introduction**

无监督的image-to-image translation的目标是在没有任何监督信号的情况下，学习两个不同的image domains之间的translation。image translation这个概念在colorization，super-resolution以及style transfer等领域被用的很多。

早期很多工作阐明了deep neural networks有能力可以进行改变局部纹理，改变图片的季节风格以及改变图片的作画风格。然而，研究者们最近发现了其在更加复杂的情况下就力不从心了，比如如果要进行translation的两个image domains之间有很大的geometry variation。为了解决这些更复杂的情况，我们需要在更高语义层面上设计这样的translation。比如说，基于对一匹马的脖子、身体和腿的理解，我们可以想象出一只具有相同姿态的长颈鹿。然而，如果仅仅是进行局部纹理的translation，这样的任务是无法完成的，因为这两类图片里的物体之间有着很大的geometry variantions。

在更高的语义层面上进行translation并不是trivial的。几何信息往往扮演了很重要的角色，然而对于有些image-to-image translation来说，两个image domain之间的几何结构差别很大，比如说cat和human的faces，horse和giraffe。尽管两个image domain的物体都有着相似的部件，但这些部件在空间中的分布是非常不一样的。

在这篇文章里，作者提出了一个新颖的geometry-aware的框架，来解决无监督的image-to-image translation。这个方法并不会直接在image space上进行translation，而是先将image映射到由geometry space和appearance space的笛卡尔积构成的空间里，然后再分别在这两个spaces上进行translation。为了鼓励网络能够更好的对这两个space进行解耦，作者提出了一个无监督的conditional的variational autoencoder (VAE)框架，其使用了KL散度和skip-connection的结构设计来使得网络能够得到互补且独立的geometry和appearance representations。接着作者再在这两个space上进行translation操作。大量的实验表明不管是在真实数据还是生成的数据上，这篇文章提出的方法效果都很好。

本文的贡献总结如下：
* 作者提出了一个新的框架来进行无监督的image-to-image translation。这个方法并不会直接在image space上进行translation，而是在它们被解耦知乎的appearance和geometry空间上进行。这篇文章的方法可以认为是CycleGAN的一个推广，其可以处理更复杂的objects，比如animals。
* 对于appearance空间和geometry空间的很好的解耦可以使得文章的框架能够处理多模态的translation任务，也就是说只需要提供不同的appearance模板就行。


**2. Related Work**

*Image-to-image Translation*
image-to-image的translation的目标是在source image domain和target image domain之间学习一个映射。Pix2Pix基于conditional GAN第一次提出了一个image-to-image的translation框架。之后有一些工作扩展了Pix2Pix使得其可以处理高分辨率或者video synthesis的任务。尽管这些工作已经取得了很好的效果，其需要成对的图片作为训练数据。对于无监督的image-to-image translation任务，其是没有成对的数据的，有的只是两个image domain的所有的图片的collections。Cycle-GAN，DiscoGAN，DualGAN以及UNIT都是基于cycle-consistency思想提出来的框架。GAN-imorph提出了一个带有膨胀卷积的discriminator用来获取一个信息更丰富的generator。然而，因为没有成对的图片作为训练数据，这样的translation任务本质上就是ill-posed的，因为基于训练数据，两个image domain之间可以存在无数个有效的映射。最近有一些工作已经开始尝试为多模态生成任务解决这个问题。CIIT，MUNIT，DRIT和EG-UNIT将图片space分解为一个domain-invariant内容空间和一个domain-specific style空间。然而，一旦做translation的两个image domain之间的几何结构信息差别太大，他们所假设的domain-invariant内容空间就会被违背了。尽管直觉上，不同的image domain是可以共享同一个content space的，但是使用同一个分布来表示不同的image domains里content分布的信息是很难的。所有的现存的方法在image domains之间有着很大的geometry variations的情况下，效果都不好。

*Structural Representation Learning*
为了建模图片里的内容，很多无监督的方法被提了出来，包括VAE，GAN，ARN等。最近，很多文章聚焦于无监督的landmark discovery来进行结构表征学习。因为landmark是物体结构的一个显式地表示，其相对于其它的特征能够更好的表示物体的几何结构信息。受到最近的landmark discovery工作的启发，本篇文章也使用了以堆叠的heatmap表示landmarks的方法来显式表示物体的结构信息。

*Disentanglement of Representation*
我们需要有一种很好的解耦方法来将appearnce和geometry特征分开。现在已经有很多的工作在研究face和person的图片生成任务。尽管他们的效果不错，但都需要有标注的信息进行监督学习。也有一些无监督解耦方法被提了出来，比如InfoGAN，$$\beta$$-VAE。然而，这些方法都缺乏可解释性，而且所学习到的解耦之后的特征都无法进行控制。本篇文章所使用的方法是可以以一种完全无监督的方法来获取appearance和geometry特征的解耦的。


**3. Methodology**

给定两个image domain，$$X$$和$$Y$$。要解决的任务是学习一对mapping，$$\Phi_{X \rightarrow Y}$$和$$\Phi_{Y \rightarrow X}$$，其可以将输入$$x \in X$$映射到$$y = \Phi_{X \rightarrow Y}(x)$$，其中$$y \in Y$$，反之亦然。这样的问题描述是一个典型的unpaired cross-domain image translation任务，其中最大的困难是其需要对两个image domain里的object的姿态，也就是geometry information进行改变。现存的方法致力于使用两个neural networks来拟合这两个mapping，但是训练起来非常的困难。在这篇文章里，作者假设每个image domain可以被解耦为一个structure space $$G$$和一个appearance space $$A$$的笛卡尔积。然后在每个space上再进行两个domain之间的translation，也就是两个geometry transformers $$\Phi_{X \rightarrow Y}^{g}$$，$$\Phi_{Y \rightarrow X}^{g}$$，和两个appearance transformers $$\Phi_{X \rightarrow Y}^{a}$$，$$\Phi_{Y \rightarrow X}^{a}$$。fig2解释了整个框架的结构。

![gaga2]({{ '/assets/images/GAGA-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2 Architecture。作者提出的框架包括四个主要部分：两个autoencoders（X/Y domain）以及两个transformers（geometry/appearance）。Autoencoder：以X domain为例。对于输入x，使用一个encoder $$E_x^g$$来获取其的geometry representations $$g_x$$，其是一个通道数为30的heatmap，分辨率和输入图片$$x$$相同（每个通道都是一个heatmap，表示每个keypoint的位置信息）.然后，$$g_x$$再被编码从而得到geometry code $$c_x$$。同时，$$x$$还被喂给一个appearance encoder $$E_x^a$$来获取appearance code $$a_x$$。最后，$$a_x$$和$$c_x$$被连接在一起使用$$D_x$$来生成$$\hat{x}$$。Transformer：对于cross-domain的translation，geometry（$$g_x \iff g_y$$）和appearance transformation（$$a_x \iff a_y$$）是分别被操作的。*


**3.1 Learning Disentangled Structure and Style Encoders**

和之前那些使用单独一个卷积网络的encoder-decoder来encoder所有信息的结构不同，本文的方法试图分别encode图片里的geometry structure和appearance style。为了达到这个目标，作者对于geometry和appearance space都分别使用了一个conditional VAE。conditional VAE包括：（1）一个无监督的geometry estimator $$E_{\dot}^g(;\pi)$$；（2）一个geometry encoder $$E_{\dot}^c(;\theta)来将heatmap作为输入，输出为geometry code；（3）一个appearance encoder $$E_{\dot}^a(;\phi)$$来将appearance信息encode到space $$A$$当中；（4）一个decoder $$D_{\dot}(;\omega): C_{\dot} \times A_{\dot} \rightarrow X/Y$$，来将特征空间再映射回输入图片空间。

>注意这些网络的下标的点，表示既可以是$$x$$也可以是$$y$$

为了能够以无监督的方式解耦geometry和appearance信息，作者将loss设置为一个conditional VAE loss和一个geometry estimation的prior的loss的结合，也就是：

$$\mathcal L_{disentangle} = \mathcal L_{CVAE} + \mathcal L_{prior}$$

受到之前工作的启发，conditional VAE loss如下所示：

$$\mathcal L_{CVAE}(\pi, \theta, \phi, \omega) = -KL(q_{\phi}(c \vert x, g) \Vert p(a \vert x)) + \lVert x - D(E^c(E^g(x)), E^a(x)) \rVert$$

上述loss里的第一项是两个参数化的高斯分布之间的KL散度，第二项是reconstruction loss。在监督学习的框架下，这个loss是可以保证其可以学习到互补且独立的appearance和geometry信息的。但是本文解决的是无监督的情况，所以在没有对geometry map $$g$$进行监督的情况下，无法保证哪个encoder学习到的是geometry信息。而下面将要介绍的prior loss将会帮助我们对geometry estimator $$g$$进行约束。

**3.2 Prior Loss for Geometry Estimator**

和现有的那些框架试图encode所有的细节信息不同的是，我们的geometry estimator尝试仅仅去提取那些纯粹的geometry结构信息，而这是通过一系列landmark heatmap堆叠来表示的。为了实现这个目标，作者使用了物体的landmarks该如何分布的先验信息来帮助学习structure estimator $$E_x^g$$和$$E_y^g$$。

作者使用的prior是：

$$\mathcal L_{prior} = \sum_{i \neq j} exp(-\frac{\lVert g^i - g^j \rVert}{2 \sigma^2}) + Var(g)$$

上述loss里的第一项是separation loss。如同之前的工作所研究的一样，如果不加这个约束，从随机的初始化开始，网络会学习到所有的landmarks都聚集在一个点上，这不是我们想要的。加了separation loss可以让landmark heatmap去覆盖object的其他区域。上述loss里的第二项是concentration loss，也就是说对于每个landmark的heatmap，我们不希望这个heatmap所代表的分布的方差太大，也就是说希望概率更加地集中于landmark那一个点。

**3.3 Appearance Transformer**

在有了解耦后的geometry和appearance空间之后，我们可以将原来的image-to-image translation问题分解为两个独立的问题。在这一节里，我们首先考虑appearance空间$$A_X$$，$$A_Y$$上的transformation $$\Phi^a$$。如果仅仅是用CycleGAN里的方法来解决，也就是说使用cycle consistency loss和adversarial loss来进行约束，是完全不够的。因为这两个约束只能保证网络学习到两个分布之间的一个translation，而这个translation具体是什么样的，完全是随机的。也就是说，其并不能保证$$a_x$$和$$\Phi_{X \rightarrow Y}(a_x)$$有着什么appearance上的关联。为了解决这个问题，作者提出了一个cross-domain appearance consisntecy loss来约束这个appearance transformer：

$$\mathcal L_{con}^{a} = \lVert \zeta_{x} - \zeta(D_y(\Phi_{X \rightarrow Y}^{g} \ddot E_{x}^{g}(x), \Phi_{X \rightarrow Y}^{a} \ddot E_x^a (x))) \rVert

其中$$\zeta$$是Gram matrix，也就是将输入通过一个预训练好的VGG-16网络得到的feature矩阵，乘以这个feature矩阵的转置得到的矩阵，$$\Phi_{X \rightarrow Y}^g \ddot E_x^g(x)$$是从$$X$$转换到到$$Y$$的geometry code，$$\Phi_{X \rightarrow Y}^a \ddot E_x^a(x)$$是从$$X$$转换到到$$Y$$的appearance code，$$D_y(,)$$是$$Y$$ domain的decoder。这个loss保证了生成图片和原始图片的appearance风格是一样的。在实验中，作者也尝试了只使用CycleGAN里的loss而不使用上述的这个loss，结果是每次训练，得到的结果都是不一样的，而加上了这个loss则变得稳定，而且结果的可解释性增强了。

**3.4 Geometry Transformer**

作者发现直接在$$E_x^g$$的输出的heatmaps之间学习一个geometry transformer是很困难的，因为CNN并不擅长获取图片里的几何结构信息。从而，作者使用了一个re-normalization operator $$R$$从heatmap中获取了每个landmark的位置信息。也就是说，geometry translation是基于landmark coordinate space上的，而并不是heatmap空间上。

具体来说，对于每个landmark heatmap，作者计算了一个加权坐标值。尽管以2D坐标表示的landmark的维度要远小于图片本身，作者还是使用了PCA来对landmark representation进行了降维处理。这背后的原因是作者发现结果对于几何结构的微小改变要比原图片pixel值得微小改变要更加敏感。

值得注意的是作者尝试了基于三种不同的geometry representations得geometry transformer，也就是geometry heatmaps，landmark coordinates和PCA处理后的landmark coordinates。然后作者发现，只有PCA处理后的landmark coordinate对应的训练过程更加地稳定。其为物体的几何形状构建了一个嵌入空间，这个空间里的每个维度都表示一个有意义的几何信息。因此，这个嵌入空间里的每个采样都会保有物体的基本形状信息，就会减少模型训练崩了的风险。


**3.5 Other Constraints**

除了之前提到的针对geometry的geometry prior loss和确保appearance consistency的consistency loss，作者还使用了cycle-consistency和adversarial loss来帮助模型训练。

*Cycle-consistency loss*
作者采用了三种cycle-consistency loss，$$\mathcal L_{cyc}^a$$，$$\mathcal L_{cyc}^g$$和$$\mathcal L_{cyc}^{pix}$$，分别对应于geometry space，appearance space和pixel space的cycle-consistency loss。

*Adversarial loss*
作者同样也使用了三种adversarial loss，也就是$$\mathcal L_{adv}^a$$，$$\mathcal L_{adv}^g$$和$$\mathcal L_{adv}^{pix}$$，分别对应于geometry space，appearance space和pixel space的adversarial loss。



### 33. [MobileNeRF: Exploiting the Polygon Rasterization Pipeline for Efficient Neural Field Rendering on Mobile Architectures](https://arxiv.org/pdf/2208.00277.pdf)

[POST](https://mobile-nerf.github.io/)

*Arxiv 2022*


### 34. [PlenOctrees: For Real-time Rendering of Neural Radiance Fields](https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_PlenOctrees_for_Real-Time_Rendering_of_Neural_Radiance_Fields_ICCV_2021_paper.pdf)

[POST](https://alexyu.net/plenoctrees/)

*ICCV 2021 Oral*


### 35. [LOLNerf: Learn From One Look](https://openaccess.thecvf.com/content/CVPR2022/html/Rebain_LOLNerf_Learn_From_One_Look_CVPR_2022_paper.html)

*CVPR 2022*


### 36. [EG3D: Efficient Geometry-aware 3D Generative Adversarial Networks](https://nvlabs.github.io/eg3d/)

*CVPR 2022*

Work from Stanford and NVidia.

[POST](https://nvlabs.github.io/eg3d/)

**Abstract**

从一系列的2D图片中以无监督的方式来生成高质量的multi-view-consistent的2D图片以及3D的shape一直以来都是个很有挑战性的问题（这里的一系列2D的图片指的是同一个场景的2D图片）。现有的3D GANs要么就是计算量太大，要么就是无法做到3D-consistent，前者限制了生成图片的质量和分辨率，后者影响了multi-view consistency以及3D shape的质量。在这篇文章里，我们将要改进3D GAN的计算效率和图片质量。我们提出了一个expressive hybrid explicit-implicit网络框架，再结合一些其他的techniques，就可以在real time生成高质量高分辨率的multi-view consistent的图片，以及高质量的3D geometry。通过解耦feature generation和neural rendering，我们的框架可以使用sota的2D CNN generators（比如说StyleGAN2），从而可以借用它们的高效性和强表达性等优势。我们使用了FFHQ和AFHQ Cats数据集来展现了所提出的方法的效果。


**1. Introduction**

GAN在被提出之后，有了非常多以及非常广的改进，现在的一些模型已经可以生成高分辨率，photorealistic，真假难辨的图片了（[Alias-free generative adversarial networks]，[A style-based generator architecture for generative adversarial networks]，[Analyzing and improving the image quality of StyleGAN]）。但现有的sota GAN仅仅在2D下作用，而且并没有显式的对潜在的3D场景进行建模。

最近的3D-aware GAN的工作开始研究multi-view consistent的图片生成问题，而且还研究了在没有geometry或者multi-view images作为监督数据的情况下获取3D shapes的问题。然而，现有的3D GAN生成的图片质量和分辨率都大大落后于这些2D GAN。而且，它们的3D reconstruction效果也还有很大的改进空间。有这样的差距的一个主要的原因是这些3D GANs的generators以及neural rendering architectures的计算都是很低效的。

和2D GANs不同，3D GANs由两部分组成：一个含有3D structure aware inductive bias的generator，以及一个neural rendering eigine来提供view-consistent结果。generator里的inductive bias可以使用显式的voxel grids或者隐式的neural implicit representations来提供（有非常多的论文，去看原论文找引用，这里就不赘述）。尽管这些方法在单个场景下是成功的，但是它们对于训练一个3D GAN来说都是不合适的，因为就像NERF一样，训练一个这样的generator需要大量的单一场景的多角度图片，会使得计算非常的慢且低效。训练一个3D GAN需要渲染上千万级别的图片，但是sota的neural volume rendering方法（也就是NERF）如果要生成高质量图片的话，利用上述的那些方法那些representations根本是不可行的。基于CNN的upsampling网络被提出来解决这个问题（[GIRAFFE: Representing scenes as compositional generative neural feature fields]），但这样的方法会牺牲掉view consistency，并且会损害学习到的3D geometry。

我们提出了一个新的方法来从一系列2D图片里无监督的学习3D representations，并且使得rendering的过程更加的高效。我们使用了一种双管齐下的策略来实现这个目标。首先，我们使用了一个hybrid explicit-implicit 3D representations来替代之前的explicit或者implicit的representations来做rendering，从而可以在不牺牲精度的情况下大大提升rendering的效率。其次，尽管我们提出的方法使用了一些近似可能会导致结果偏离ground truth的rendering结果，但我们提出了一种dual-discrimination的策略来保证rendering结果的view consistency。更进一步，我们还在generator里使用了pose-based的condition，其将pose-correlated的特征（比如说人脸表情）从其他特征中分离开，从而更好的对整个数据集里的这样的特征进行建模。

我们提出的方法还有个附加的优势，其可以将feature generation和nerual rendering解耦开，从而我们就可以使用那些sota的2D CNN-based feature generator（比如说StyleGAN2）来在3D场景里生成features，然后再用我们提出的3D multi-view-consistent的neural volume rendering方法来进行渲染。我们的方法不仅在生成3D-aware的图片上qualititave和quantitative都达到了sota的效果，还可以输出场景的高质量的3D shapes（因为generator里有很强的3D structure aware inductive bias）。由fig1可见。

本文的贡献总结如下：
* 我们提出了一个tri-plane-based 3D GAN框架，其既高效也具有表达力，可以实现高分辨率的geometry-aware的图片生成。
* 我们使用了一个3D GAN的训练策略，通过dual discrimination以及在generator里加pose condition来更好的对数据集里的pose-correlated的特征进行建模（比如说人脸表情），从而更好的实现multi-view consistency。
* 我们在FFHQ和AFHQ CATS数据集上对所提出的方法进行测试并达到了sota的效果，而且还可以生成高质量的3D geometry。


![eg3d1]({{ '/assets/images/EG3D-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1 本文所提出的3D GAN可以实现高分辨率高质量的multi-view consistent的图片生成，并且还可以生成高质量的geometry。而且本文的模型并不需要shape priors，ground truth的3D scans或者multi-view supervision来辅助训练，只需要一个场景的多个角度的图片就可以了。*


**2. Related work**

*Neural scene representation and rendering*

有大量的研究工作使用differentiable 3D-aware的representations来进行neural scene representations，这样的3D-aware representations可以通过优化由neural rendering得到的2D multi-view的图片来被学习到（相关论文太多，参考原论文，基本都是基于NERF的改进）。上述这种representations，显式的那些，比如discrete voxel grids（fig2 b所示），可以被非常快的计算，但是需要很多的memory来存储，从而将这种方法推广到高分辨率或者复杂场景的时候就很困难。implicit representations，或者coordinate networks（fig2 a所示），在memory存储上有优势，并且也能够表达复杂的场景（其将场景表达为一个连续的函数）。在实际应用中，使用这种implicit representations的模型需要使用很大的MLP，从而并不高效。因此，implicit和explicit representations都各自有各自的好处。

local implicit representations以及hybrid explicit-implicit representations将这两种representations结合了起来。由这些研究工作启发，我们设计了一个新的hybrid explicit-implicit 3D-aware网络，其使用了一个memory高效的tri-plane representations来显式的存储axis-aligned planes上的features，然后这些features由一个轻量级的implicit feature decoder来将其综合起来，从而实现了高效的volume rendering，这个过程由fig2 c所示。我们所提出的representations和之前的plane-based hybrid architectures（[Unconstrained scene generation with locally conditioned radiance fields]，[Convolutional occupancy networks]）有一些相似的理念，但具体的设计是特别的。我们所提出的这个representations是我们的模型能够让3D GAN在real time时间内能生成高质量图片的关键。

*Generative 3D-aware image synthesis*

GAN现在在生成高质量的2D图片领域已经很成功了。而且将这样的效果推广到3D领域也已经在进行中了。基于mesh的方法使用的是计算机图形学领域最常见的数据类型，但其并不能生成高保真的2D图片。基于voxel的GAN直接将2D GAN里的CNN扩展到了3D领域。但voxel需要大量的存储空间，而且3D convolution计算缓慢，这些导致了训练3D GAN非常困难。低分辨率的3D GAN里的3D convlution计算可以被2D CNN的上采样层替换掉，但是这样的话就没有3D inductive bias了，从而结果就不具有view consistency。block-based稀疏volume representations克服了上述的某些困难，但是仅仅对于一些简单的场景效果好，而且很难泛化到其他场景里。除了这些显式的特征，还可以用MLP来学习隐式的特征，但由于需要使用较大的MLP，所以训练仍然很复杂，时间较长，从而还是限制了对于高分辨率以及复杂场景的3D GAN的训练。

本文的一个最主要的见解就是一个有着3D inductive bias的高效3D GAN是生成高质量高分辨率且view consistent的图片的关键。而我们通过以下集中方式来实现了这样一个3D GAN模型。首先，和大多数3D GAN不同的是，我们直接使用了一个2D CNN-based feature generator，也就是StyleGAN2，从而就不再需要在voxel grids上进行低效的3D convolutions计算。其次，我们的tri-plane representations可以让我们将neural volume rendering作为一个inductive bias，但是要比NERF里的那种情况要高效很多。我们在进行neural rendering之后，仍然需要2D CNN-based upsampling操作，但我们使用了dual discrimination来避免了upsampling这样一个近似的计算所带来的view inconsistency。和现有的基于StyleGAN2的2.5D GANs不同（它们生成的是图片和其对应的深度图），我们的方法在各个角度都可以生成图片，也就是对整个3D场景都有了学习。

现有的3D-aware GANs，比如StyleNeRF（[A style-based 3D-aware generator for high-resolution image synthesis]），CIPS-3D（[A 3D-aware Generator of GANs Based on Conditionally-Independent Pixel Synthesis]）等都已经给出了很impressive的图片效果了。而这些工作和我们的工作的最主要的区别在于，StyleNeRF和CIPS-3D主要在图片空间内操作，而很少强调3D representations，但我们的工作主要在3D空间内进行。我们的模型会生成更加view consistent的结果，并且还可以给出3D shapes。而且我们的模型在FFHQ和AFHQ数据集上有更好的FID image scores。

![eg3d2]({{ '/assets/images/EG3D-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2 (a) NeRF里的方法，使用MLP和positional encoding来隐式地表达一个场景，效率很低。(b) 显式的voxel grids和很小的implicit decoder来表示特征，这样会快很多，但是很难扩展到高分辨率的情况，因为voxel需要大量的存储空间。(c) 本文提出的representation，既快又好，可以生成高质量的高分辨率的图片。*


**3. Tri-plane hybrid 3D representation**

训练一个能生成高分辨率的GAN需要一个既高效又具有表达力的3D representation。在这一节里，我们将会介绍一个新的hybrid explicit-implicit tri-plane representation来提供上述这些优势。我们在这一节里对于一个单一场景的over fitting（SSO）实验来提出这样的一个representation，在之后的章节里再解释如何将其应用在我们的GAN框架里。

在tri-plane的形势下，我们将显式的特征沿着三个正交轴的feature planes对齐，每个轴上的特征的size都是$$N \times N \times C$$，由fig 2 c表示，其中$$N$$是分辨率，$$C$$是通道数。对于空间中任意一个点的坐标$$x \in \mathbb{R}^3$$，我们将其投射到这三个平面上，从而得到了对应的三个feature vector，$$F_{xy}, F_{xz}, F_{yz}$$（如果不在节点上还要使用某种插值方法，比如bilinear interpolation），然后通过加和的方式将其综合起来，再通过一个轻量级的MLP来增加隐式的特征，最后输出color和density。最后再通过和NeRF一样的volume rendering操作得到图片。

>也就是说，这里只需要学习三个平面上的feature就行了，也就是垂直于X，Y和Z轴的三个平面上的点的features，然后对于空间里任意一点，分别投射到这三个平面上，利用插值或者找最近的那个点，就作为空间中这一点在这个平面上的feature，也就是上面的$$F_{xy}, F_{xz}, F_{yz}$$。

这种混合方法最大的一个优势就是高效，因为现在这个MLP就可以设计的很小，将模型的表达力转移到前面的explicit representations的部分（也就是三个平面分解的部分），从而大大减小了训练的难度，而且也不会丢失渲染精度。为了验证这个tri-plane representation确实是有效的，我们使用一个常见的novel-view synthesis来验证它。我们设置$$N=512, C=48$$，使用的MLP有四层，隐层的宽度是128，并且使用了一个Fourier feature encoding来对第一步的输入MLP前的特征进行处理。数据集是Tanks & Temples数据集。fig3和tab1展示了我们所提出的tri-plane representation的效果，其具有能够表达复杂场景的能力，而且计算量大大减小了。

![eg3d3]({{ '/assets/images/EG3D-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 3 左边是NeRF的效果，中间是使用voxel的效果，右边是本文提出的representation的效果。*

![tab1]({{ '/assets/images/EG3D-0.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

tri-plane representations还有一个非常大的优势：这三个feature planes可以利用某种2D CNN-based generator得到（比如StyleGAN2），从而更加灵活。我们这一节的实验是在SSO设定下进行的，放在后面的GAN设定下，这样的优势会体现得更明显。


**4. 3D GAN framework**

有了一个高效以及具有表达力的3D representation之后，我们就可以从一系列2D图片里来训练一个3D GAN了，而不再需要其它的3D prior或者multi-view的监督信号来辅助。对于每张训练图片，我们使用某种pose detector（预训练好了的）来检测其的相机内部和外部参数。

![eg3d4]({{ '/assets/images/EG3D-4.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 4 本文的3D GAN框架由几个部分组成：一个pose-conditioned StyleGAN2-based特征generator和一个mapping网络，一个tri-plane 3D representation以及配合的轻量级feature decoder，一个neural volume渲染器，一个super-resolution模块，以及一个有着dual discrimination的pose-conditioned StyleGAN2 discriminator。这样一个框架很好的将feature generation和neural volume rendering解耦开，从而可以很好的利用强大的StyleGAN2来提取特征。而且简洁而具有表达力的tri-plane representations也使得real time训练和推断这个框架变得可能。*

fig 4是对整个模型框架的一个overview。我们使用上一个section提到的tri-plane representation来通过neural volume rendering来渲染图片，但是因为之前介绍tri-plane representation时用的是SSO的场景，所以在这个3D GAN训练的场景下还需要做一些修改。和SSO实验不同的是，在SSO实验里，各平面的features是由多个角度的输入来优化的，但在GAN的情况下，我们使用一个2D convolutional StyleGAN2来生成每个平面的features（在4.1里介绍）。而且我们也不是用neural volume rendering直接渲染出RGB图片，而是在得到了tri-plane representation之后，通过一个super-resolution模块来upsample所得到的结果（在4.2里介绍）。再之后，这张图片将会被一个StyleGAN2 discriminator所判断是真是假。整个框架是端到端进行训练的，随机初始化，使用了StyleGAN2里的训练策略。

**4.1 CNN generator backbone and rendering**

在这个GAN设定下所使用的tri-plane representation，是由一个StyleGAN2 CNN生成的。random latent code和相机参数被一个mapping网络接受，来产生一个intermediate latent code，其会作为StyleGAN2网络和super-resolution模块的一个输入。

我们将原本的StyleGAN2的输入大小从三通道的RGB图片改成一个$$256 \times 256 \times 96$$的feature image。然后这96个通道分成三等分，从而得到了三个32通道的平面。

再之后，我们在空间中随机采样点，然后投射到三个平面上，从而得到这个点的feature，通过加和综合在一起，再使用一个轻量级的decoder来处理。这个decoder是只有一个隐层的MLP，其输入并没有positional encoding，view direction input等。这个decoder的输出是一个标量的density $$\sigma$$以及一个32个通道的feature，再之后会利用一个neural volume渲染器来将这个3D的feature volume投影到2D上。

volume rendering的过程和NeRF里的过程一样，有一点不同的是，volume rendering的输出结果是一个feature image，而不是RGB image，这一点和GIRAFFE那篇论文里一样。因为feature image含有更多的信息可以被用来进行super resolution的操作。

**4.2 Super resolution**

虽然tri-plane representation要比之前那些方法使用的representation在计算量上减少很多了，但在分辨率很高的时候仍然不够快。因此我们打算在一个适中的分辨率上进行volume rendering，然后再利用upsampling将所得到的RGB图片提到到更高的分辨率上。

我们的super resolution模块有两个改动的StyleGAN2组成，其将32通道的feature image变成最终的RGB image。

**4.3 Dual discrimination**

和标准的2D GAN训练一样，最后的渲染结果由一个2D convolutional discriminator来判别。我们使用一个StyleGAN2的discriminator，但有两点改进。

首先，我们引入了dual discrimination来避免multi-view inconsistency的问题。我们认为neural rendering的结果$$I_F \in \mathbb{R}^{128 \times 128 \times 32}$$的前三个通道是低分辨率的原图片，记这前三个通道构成的图片为$$I_{RGB}$$。直觉上看，dual discrimination保证了$$I_{RGB}$$和$$I_{RGB}^{'}$$之间的一致性（$$I_{RGB}^{'}$$是super resolution模块的输出。我们使用bilinear插值将$$I_{RGB}$$提升到和$$I_{RGB}^{'}$$一样的分辨率，然后沿着通道连接，从而构成了一个6通道的feature image，喂给discriminator来判别。而真实的图片，同样也先进行降采样，再利用同样的方法构造这样一个6通道的image喂给discriminator。这样的操作就叫做dual discrimination。

dual discrimination不仅会使得最终的输出匹配真实图片的分布，还提供了额外的好处：它会使得neural rendering的结果匹配降采样的真实的图片的分布；其也会使得super resolution后的结果和neural rendering的结果保持一致，从fig 5可以看出。后一点很重要，因为其可以让我们在不引入新的view consistency的约束的情况下，使用super resolution这个模块。

>因为对于真实的图片，其要经过降采样，再通过bilinear插值升采样，再连接成6通道的image作为一个数据，其前三个通道和后三个通道有很强的联系（基本就是一样的，模糊版本），而且真实的图片的例子是view consistent的。从而经过训练，neural rendering所得到的结果的最后三个通道需要和真实图片的最后三个通道的分布相匹配，从而就view consistent了。而真实图片的前三个通道和后三个通道相似，从而super resolution的结果也就需要和neural rendering的结果相似，从而就具有了view consistency的效果。

![eg3d5]({{ '/assets/images/EG3D-5.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 5 dual discrimination保证$$I_{RGB}$$和$$I_{RGB}^{'}$$保持一致，从而使得高分辨率的结果保证了multi-view-consistent的特性。*

而且，我们的discriminator是有相机参数输入这个信息的。

**4.4 Modeling pose-correlated attributes**

大多数的现实的数据集比如FFHQ包含一些bias，其会将相机参数和其他一些特性相关联起来（比如facial expression），如果不小心处理就会导致view inconsistency。比如说，有关人脸的相机参数就和smile有关。我们必须处理这些特征，将其解耦，才能够更好的生成质量更好的图片。

我们提出了generator pose conditioning作为对pose和其它特征之间的关系的建模和解耦方法。为了实现这个目标，mapping network的输入不仅是一个latent code vector $$z$$，还加入了相机参数$$P$$。通过给mapping network提供相机参数，我们允许相机参数来影响生成的结果。

在训练过程中，上述的pose conditioning使得generator可以对数据集里的pose-dependent的bias进行建模，从而我们的模型就可以更加真实的匹配原数据集里的分布。


**5. Experiments and results**


**6. Discussion**



## Animation

### 1. [First Order Motion Model for Image Animation](https://proceedings.neurips.cc/paper/2019/hash/31c0b36aef265d9221af80872ceb62f9-Abstract.html)

*NeurIPS 2019*


### 2. [Animating Arbitrary Objects via Deep Motion Transfer](https://openaccess.thecvf.com/content_CVPR_2019/html/Siarohin_Animating_Arbitrary_Objects_via_Deep_Motion_Transfer_CVPR_2019_paper.html)

*CVPR 2019*


### 3. [Motion Representations for Articulated Animation](https://openaccess.thecvf.com/content/CVPR2021/html/Siarohin_Motion_Representations_for_Articulated_Animation_CVPR_2021_paper.html)

*CVPR 2021*



## Reconstruction

### 1. [Interacting Attention Graph for Single Image Two-Hand Reconstruction](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Interacting_Attention_Graph_for_Single_Image_Two-Hand_Reconstruction_CVPR_2022_paper.pdf)

*CVPR 2022*


### 2. [Human Mesh Recovery from Multiple Shots](https://openaccess.thecvf.com/content/CVPR2022/papers/Pavlakos_Human_Mesh_Recovery_From_Multiple_Shots_CVPR_2022_paper.pdf)

*CVPR 2022*


### 3. [De-rendering 3D Objects in the wild](https://www.robots.ox.ac.uk/~vgg/research/derender3d/)

[POST](https://www.robots.ox.ac.uk/~vgg/research/derender3d/)

*CVPR 2022*


### 4. [OcclusionFusion: Occlusion-aware Motion Estimation for Real-time Dynamic 3D Reconstruction](https://arxiv.org/pdf/2203.07977.pdf)

[POST](https://wenbin-lin.github.io/OcclusionFusion/)

*CVPR 2022*


### 5. [NeuralRecon: Real-Time Coherent 3D Reconstruction from Monocular Video](https://zju3dv.github.io/neuralrecon/)

[POST](https://zju3dv.github.io/neuralrecon/)

*CVPR Oral & Best Paper Candidate*


### 6. [Robust Equivariant Imaging: a fully unsupervised framework for learning to image from noisy and partial measurements](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Robust_Equivariant_Imaging_A_Fully_Unsupervised_Framework_for_Learning_To_CVPR_2022_paper.pdf)

*CVPR 2022*


### 7. [MoFA: Model-based Deep Convolutional Face Autoencoder for Unsupervised Monocular Reconstruction](https://vcai.mpi-inf.mpg.de/projects/MoFA/paper.pdf)

[POST](https://vcai.mpi-inf.mpg.de/projects/MoFA/)

*ICCV 2017 Oral*


### 8. [Unsupervised Part Discovery from Contrastive Reconstruction](https://proceedings.neurips.cc/paper/2021/hash/ec8ce6abb3e952a85b8551ba726a1227-Abstract.html)

*NeurIPS 2021*


### 9. [Self-supervised Single-View 3D Reconstruction via Semantic Consistency](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590664.pdf)

*ECCV 2020*


### 10. [On the Generalization of Learning-Based 3D Reconstruction](https://openaccess.thecvf.com/content/WACV2021/html/Bautista_On_the_Generalization_of_Learning-Based_3D_Reconstruction_WACV_2021_paper.html)

*WACV 2021*



## Editing

### 1. [3D Part Guided Image Editing for Fine-Grained Object Understanding](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_3D_Part_Guided_Image_Editing_for_Fine-Grained_Object_Understanding_CVPR_2020_paper.html)

*CVPR 2020*


---
