---
layout: post
comments: false
title: "[论文]Scene Graph Generation/Semantic Segmentation"
date: 2021-11-29 01:09:00
tags: paper-reading
---

> This post is a summary of scene graph generation or semantic segmentation papers.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---


### 1. [Learning Physical Graph Representations from Visual Scenes](https://arxiv.org/pdf/2006.12373.pdf)

[POST](https://neuroailab.github.io/physical-scene-graphs/)

*NeurIPS 2020*

*Daniel M. Bear, Chaofei Fan, Damian Mrowca, Yunzhu Li, Seth Alter, Aran Nayebi, Jeremy Schwartz, Li Fei-Fei, Jiajun Wu, Joshua B. Tenenbaum, Daniel L.K. Yamins*


**Abstract**

很多结果已经证明了CNN对于视觉物体分类（visual object categorization）这个任务能够学习到很好的representations。然而，CNN并没有显式的为图片里的object，parts以及它们的物理特性（physical properties）进行编码，这就限制了CNN在学习那些需要理解场景里的结构化信息的任务。为了克服这些限制，我们提出了Physical Scene Graphs（PSGs），其将一个场景（scene）表示为一个层次化的graph（hierarchical graph），graph里的nodes就自然对应各个不同scale下的object parts了，edges对应这些parts之间的物理联系（physical connections）。而每个node上还绑着一个向量，这个向量是表示这个node的代表的object part的特性，比如说surface shape、texture等。我们还提出了PSGNet，它是一个网络架构，其输入是一个场景，而其会将这个场景表示为一个PSG，然后再利用这个PSGNet来重构输入的场景，从而达到学会从场景图片里获取PSG的能力（有点像auto-encoder，而专业的表达这个过程的方式是通过一个PSG-structured bottleneck）。PSGNet在CNN上添加了以下的部分：（1）联合低层和高层图片信息的循环反馈链接（recurrent feedback connection）；（2）图池化（graph pooling）和vectorization操作，其将空间各向同性（spatially unform）的feature map转换为object-centric的图；（3）perceptual聚类约束，鼓励网络找到有意义的场景元素。我们证明PSGNet在scene segmentation任务上要比那些自监督scene representation learning的算法效果好，特别是对于现实世界里的很复杂的图片，而且PSGNet还能够泛化到没见过的object种类以及没见过的scene布置上。PSGNet还能够从物理运动（physical motion）里学到知识，从而为静态的场景图片提供更稳定的预测结果。我们提供了一系列的消融实验证明我们提出的PSGNet里的每个部分都是不可或缺的。我们的分析证明PSGNet确实能够学习到含有场景信息的feature embeddings，并且证明了PSGNet对compositional scene inference的作用。


**1. Introduction**

为了对它们所见到的环境做出解释，人工智能必须要对复杂的视觉场景（visual scene）构造出一个representation。最近，一类CV算法，CNN，已经证明了其从visual scene里获取用于分类的信息的能力。然而，人类的感知（同时也是CV的目标）不仅仅是图像分类。人类会将场景进行聚类，从而获取这个场景的以物体为中心的representations，这个representations里包含物体、物体的组成部分、物体的位置、物体的姿态、物体的3D几何信息、物体的物理材料特性、以及物体之间的联系。这样的以物体为中心（object-centric），几何信息丰富（geometrically-rich）的representations在认知领域十分重要，而且也天然的支持更高层次的视觉规划和推断任务。

最近的工作，比如说MONet（[Monet: Unsupervised scene decomposition and representation](https://arxiv.org/pdf/1901.11390.pdf)）和IODINE（[Multi-Object Representation Learning with Iterative Variational Inference](https://proceedings.mlr.press/v97/greff19a.html)），在以物体为中心（object-centric）的场景理解方面开始做出努力。以自监督学习的方式，这些模型可以将简单的人工生成的场景分离成一个个的物体。但是，这些方法对于所得到的物体并没有给出它们的物理结构信息，而且只是从静态的图片里去学习。从而，这些方法在复杂的真实世界的图片里就效果不好了。最近的3D视觉领域的方法，比如说3D-RelNet（[3D-RelNet: Joint Object and Relational Network for 3D Prediction](https://nileshkulkarni.github.io/relative3d/)），通过将关键的几何结构信息（比如说meshes）和由CNN获取的features结合起来来解决场景理解问题。这些方法的确获得了不错的结果，但是它们需要大量的、细节的关于scene结构的ground truth监督信号来完成学习。

在这篇文章里，我们提出了一个新的representation，physical scene graph（PSG）。PSG的概念借鉴了MONet/IODINE和3D-RelNet这两条research lines里的想法，想要去同时（1）表示复杂物体的shapes和textures；（2）显式的将场景分解为physical parts；（3）支持generative models对场景的自上而下的inference reasoning；（4）并且不需要监督信号，通过自监督的方式从真实世界的图片或者非常真实的生成图片里学习信息。PSGs将场景表示为层次化graph（hierarchical graphs），其中PSGs顶层的nodes直觉上表示的是更大的groupings（比如说单个的整个物体），而下层的nodes表示的是每个object的某些部分，而edges表示的是nodes之间的关系。PSGs同时还表示了每个node在图片里的位置（这种特性叫做spatial registered）。node的attributes表示这个node的物理意义上的性质（和这个node在哪个层级也是有关的），比如说物体的位置、物体的surface Normals、物体的形状以及物体的visual appearance。

我们最关键的贡献是提出了一个family的自监督网络框架，PSGNets，其可以从图片输入里学习PSGs。PSGNets是在基础的CNN上加了一些部分来实现的。为了在初始的feature extraction阶段更高效的结合高层次和低层次的视觉信息，我们在CNN的结构基础上加了局部的循环（local recurrent connections）以及长程反馈连接（long-range feedback connections），这样被改造后的CNN叫做convolutional recurrent neural network（ConvRNN）。我们还提出了一个可学习的图池化操作（graph pooling），将spatially-uniform的ConvRNN的feature map转换为object-centric的graph。我们还提出了一个将objects的特性综合起来的运算：graph vectorization operation。一系列交替的graph pooling和graph vectorization模块就构成了层次化的graph constructor。为了让这个graph constructor能够学习到具有物理意义的scene representations，我们在模型里编码了perceptual grouping的信息，包含静态和动态的grouping primitives。为了保证所学习到的node attributes包含场景图片里真实的可以被分开的（disentangled）objects的geometric和visual特征，我们还引入了一个decoder，从PSGs来自上而下的重构原输入的场景图片。

在多个数据集上，PSGNet都比那些使用无监督学习的工作在segmentation的任务上效果要更好，尤其是对于真实世界的图片来说。我们还发现，如果数据能够被提供的话，PSGNet能够充分从物体的运动里学到信息，并且将此应用到静态场景图片的学习里，从而和其它的motion-aware自监督方法比大幅度的提高了scene segmentation的效果。PSGNet所学习到的representations对于那些从没见过的objects和scenes来说泛化的很好，说明PSG结构里强力的限制使得网络确实学习到了普适的场景特性，这要比其它的方法所需要的训练数据量少得多。最后，我们展示了PSG的结构是如何表示关键的geometric和object-centric特性的，从而使得我们可以在一个场景里编辑objects的位置、结构以及改变objects的attributes。


**2. Related Work**

使用非监督方法来学习scene representations是CV领域一个长久的问题。经典的方法研究如何将像素点聚集为patches和object instance（都是2005年以前的方法，就不列出了）。最近，研究者们开发出利用deep generative models来做非监督segmentation的方法，其通过获取scene components之间的联系来实现的：[Monet: Unsupervised scene decomposition and representation](https://arxiv.org/pdf/1901.11390.pdf)，[Genesis: Generative scene inference and sampling with object-centric latent representations](https://openreview.net/pdf?id=BkxfaTVFwH)，[Attend, Infer, Repeat: Fast Scene Understanding with Generative Models](https://proceedings.neurips.cc/paper/2016/file/52947e0ade57a09e4a1386d08f17b656-Paper.pdf)，[Multi-object representation learning with iterative variational inference](http://proceedings.mlr.press/v97/greff19a/greff19a.pdf)。除了静态的图片，研究者们同时还研究了利用运动信息来做object instance segmentation这个任务，他们的目标是同时得到object appearance和motion的信息，而且是无监督的方法：[Neural Expectation Maximization](https://proceedings.neurips.cc/paper/2017/file/d2cd33e9c0236a8c2d8bd3fa91ad3acf-Paper.pdf)，[Sequential Attend, Infer, Repeat: Generative Modelling of Moving Objects](https://proceedings.neurips.cc/paper/2018/file/7417744a2bac776fabe5a09b21c707a2-Paper.pdf)。这些方法都野心勃勃，但是实际上都只能在简单的，而且绝大多数情况下都是人工合成的场景图片上使用。用于从视频里进行motion clustering和segmentation的CV算法获取了很大的成功：[Towards segmenting anything that moves](https://openaccess.thecvf.com/content_ICCVW_2019/papers/HVU/Dave_Towards_Segmenting_Anything_That_Moves_ICCVW_2019_paper.pdf)，[Object discovery in videos as foreground motion clustering](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xie_Object_Discovery_in_Videos_as_Foreground_Motion_Clustering_CVPR_2019_paper.pdf)，但是它们都需要随着时间为每个object segmentation做大量精确的标注。还有一些工作尝试使用无监督的方法从视频里学习segmentation或者grouping，然后再辅助feature learning和instance segmentation任务：[Video segmentation by non-local consensus voting](https://www.wisdom.weizmann.ac.il/~alonf/papers/NonLocalVideoSegmentation.pdf)，[Learning features by watching objects move](https://openaccess.thecvf.com/content_cvpr_2017/papers/Pathak_Learning_Features_by_CVPR_2017_paper.pdf)，[Learning instance segmentation by interaction](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w40/Pathak_Learning_Instance_Segmentation_CVPR_2018_paper.pdf)。然而，这些方法并没有学习object-centric representation，或者没有建立scene elements之间的关系。最近，graph neural network（GNN）证明了其再physical dynamics prediction方面的潜力，但是它仍需要graph-structured的输入或者监督信号：[Learning particle dynamics for manipulating rigid bodies](https://openreview.net/pdf?id=rJgbSn09Ym)，[Visual grounding of learned physical models](http://proceedings.mlr.press/v119/li20j/li20j.pdf)，[Flexible Neural Representation for Physics Prediction](https://proceedings.neurips.cc/paper/2018/file/fd9dd764a6f1d73f4340d570804eacc4-Paper.pdf)，[Learning to simulate complex physics with graph networks](http://proceedings.mlr.press/v119/sanchez-gonzalez20a/sanchez-gonzalez20a.pdf)。而这也是我们希望能从图片输入学习到graph结构的输出的一个动机。

研究者们已经提出了将object-part层次化和object之间的关系考虑入内的scene representations。scene graphs（[Visual genome: Connecting language and vision using crowdsourced dense image annotations](https://arxiv.org/pdf/1602.07332.pdf)）是一个能够同时在一张图片里表示物体以及它们之间关系的representation。然而绝大多数的方法都是从有良好标注的数据里学习scene graph representations，只有少数研究是从无需标注的视频里去学习objects和它们的关系：[R-sqair: Relational sequential attend, infer, repeat](https://grlearning.github.io/papers/138.pdf)，[Relational Neural Expectation Maximization: Unsupervised Discovery of Objects and Their Interactions](https://www.sjoerdvansteenkiste.com/assets/papers/steenkiste2018relational.pdf)，[Unsupervised Discovery of Parts, Structure, and Dynamics](https://openreview.net/pdf?id=rJe10iC5K7)。这些非监督的方法有着和我们的方法最接近的设置，但是它们仅仅只是考虑2D像素层面上的segmentation。最近的一些工作开始关注从原始输入里获取3D的，以物体为中心的scene representations。最引人注目的工作包括Factor3D（[Factoring Shape, Pose, and Layout from the 2D Image of a 3D scene](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tulsiani_Factoring_Shape_Pose_CVPR_2018_paper.pdf)），MeshRCNN（[Mesh r-cnn](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gkioxari_Mesh_R-CNN_ICCV_2019_paper.pdf)）以及3D-SDN（[3D-aware scene manipulation via inverse graphics](https://proceedings.neurips.cc/paper/2018/file/64223ccf70bbb65a3a4aceac37e21016-Paper.pdf)）。3D-RelNet（[3d-relnet: Joint object and relational network for 3d prediction](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kulkarni_3D-RelNet_Joint_Object_and_Relational_Network_for_3D_Prediction_ICCV_2019_paper.pdf)）将Factor3D更进一步拓展为可以预测物体之间的联系。然而，这些3D-aware scene representations都依赖于对物体geometry的标注信息，这就限制了它们再实际场景里对于没见过的物体的预测，因为这个时候就没有物体的3D几何信息了（因为压根就没见过这种物体）。一些无监督学习的模型，比如Pix2Shape（[Pix2shape: Towards unsupervised learning of 3d scenes from images using a view-based representation](https://arxiv.org/pdf/2003.14166.pdf)），GQN（[Neural Scene Representation and Rendering](https://www.gelecekburada.net/wp-content/uploads/google-yapay-zeka-perspektif-tamamlama-gelecek-burada.pdf)）以及neural rendering方法（[State of the art on neural rendering](https://onlinelibrary.wiley.com/doi/am-pdf/10.1111/cgf.14022)）也被提了出来。然而，它们还没有一个能够建立层次化的scene representation或者objects之间的联系。

PSGNet建立在上述这些方法之上，将它们的想法和思路综合起来，建立了一个层次化的（hierarchical），3D-aware的representation，而且不需要任何scene structure的监督信息。


**2. Method**

现在我们介绍PSG representation，PSGNet模型结构的组成部分，以及如何训练PSGNets。我们既解释了high-level的purpose，也解释了PSGNet每个组成部分的结构。正式的定义和实验implementation细节在补充材料里说明。

**2.1 Physical Scene Graphs**

PSGs是层次化的图结构（hierarchical graph），用来表示场景的层次化和物理结构。graph里的vertices，表示的是物体（objects）或者物体的一部分，用一系列hierarchical levels来表示：$$\lbrace V_l \vert l = 0,1,\cdots,L \rbrace$$。level $$l$$和level $$l+1$$之间的edges表示的是局部和整体（part-whole）之间的联系，叫做child-to-parent edges $$P_l$$。level $$l$$内部的vertices之间的edges表示objects之间或者object parts之间的关系，叫做within-level edges $$E_l$$。原则上，不同的within-level edge的集合应该能够编码不同的relationships，但是在这篇文章里，它们都表示的是physical connections。

除了vertices和edges，PSGs有两个额外的结构：（1）attributes vectors $$\lbrace A_l \rbrace$$，用于给每个vertex $$v$$添加标签，表示的是每个scene element的physical properties；（2）spatiotemporal registrations（SRs）$$\lbrace S_l \rbrace$$，其将每个level的PSG vertex显式的连接到一个base Tensor $$\mathcal F$$的某些像素点构成的子集，这个base Tensor $$\mathcal F$$形状是$$(T,H,W,C_0)$$，其中$$T$$表示的就是视频的帧数，$$H,W$$分别表示每一帧feature的大小，$$C_0$$是通道数，所以$$\mathcal F$$就是输入的视频通过CNN之后得到的feature maps。PSG就是由下而上在这个feature maps $$\mathcal F$$上构造出来的，因此PSG的第0个level就是$$\mathcal F$$，$$\mathcal F$$里的每个spatiotemporal点$$(t,h,w)$$就是一个vertex，而这个vertex的attribute就是这个点处长度为$$C_0$$的vector feature，记为$$A_0(v_{thw}) = \mathcal F \left[t,h,w,: \right]$$。一个registration $$S_l$$将这些spatiotemporal点划分为$$\lvert V_l \rvert$$个部分，也就是将base tensor $$\mathcal F$$按照第$$l$$层里vertex的个数进行了一个segmentation（从而反推回到CNN的输入，也就对输入也进行了一次分割）。更进一步的，SRs的层次化结构也要和vertex的层次化结构相匹配：如果base tensor $$\mathcal F$$里的一个spatiotemporal点通过$$S_l$$被归为vertex $$v$$了，那么通过child-to-parent edge $$P_l$$所找到的$$v$$在第$$l+1$$层的parent vertex $$P_l(v)$$，也应该通过$$S_{l+1}$$被归于vertex $$P_l(v)$$。因此，SRs表示的是场景图片里的pixels该如何层次化的被归类到每个object以及object的部分里去。一个node $$\pmb n_{l,i}$$表示的是PSG第$$l$$层的第$$i$$个vertex $$v_i$$、以及它的attribute vector $$A_l(v_i)$$，以及通过registraition $$S_l$$找到的base tensor $$\mathcal F$$上的spatiotemporal点的集合（实际上每层的registration $$S_l$$用值来表示base tensor上的点应该属于哪个vertex：比如说如果$$S_l$$里base tensor的某个点的值为$$i$$，那么它就归于第$$l$$层的第$$i$$个vertex）。


**2.2 The PSGNet Architecture**

PSGNets的输入是一张图片或者一系列图片（视频），输出是一个场景的PSG representation。PSGNet将PSG作为encoder所得出的representation，利用一个encoder-decoder的架构来进行训练：一个PSGNet encoder首先利用一个convolutional recurrent neural network（ConvRNN）（[Task-driven convolutional recurrent models of the visual system](https://proceedings.neurips.cc/paper/2018/file/6be93f7a96fed60c477d30ae1de032fd-Paper.pdf)）来从输入里获取features，然后再在features（也就是base tensor $$\mathcal F$$）上构造一个多层的PSG，最后再利用一个或者一系列decoder来还原输入（如果是视频的话就需要一系列decoders来还原每一帧）。我们接下来将会详细介绍encoder的结构，并在补充材料里提供全部的实现细节。我们之后再来介绍如何设计loss以及如何训练这个PSGNet。值得注意的是，PSG里的任何内容（vertices，edges，attributes，registrations）从没有受到任何ground truth的监督信号（比如说用来表明每个像素应该属于哪个objects的segmentation maps）。因此，PSGNets是自监督的图构造算法（self-supervised graph construction algorithm）。

**2.2.1 ConvRNN Feature Extraction**

PSGNet的目的是构建一个反映场景物理结构的scene representation：决定输入的图片里的每个像素都属于哪个object，以及这些objects都有什么样的空间和几何特性等等。因为这个原因，PSGNet里构造PSG的encoder部分需要在visual features的基础上进行聚类objects以及推断objects的attributes的工作。这种类型的信息仅仅从原输入的RGB图片构成的视频里是很难获得的，但是将RGB图片通过CNN学习后得到的feature maps就很适合作为这种visual features了。然而，CNN网络（特别是一些hourglass网络）最后一层所得到的feature maps往往过于平滑，从而并没有objects之间的boundaries的良好的定义（这个问题也促生了一部分的研究工作，比如说conditional random fileds，来利用ground truth的监督信息来改善这个问题：[Deeplab: semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs](https://arxiv.org/pdf/1606.00915.pdf)），这个问题限制了CNN将场景准确分割为objects的能力。我们因此需要改进一下CNN网络，从而使得其所得到的features（用于作为PSG的base tensor），既能保持CNN所得到的features的高度非线性的特性，又能准确地讲像素点分割为objects。

为了达到这个目的，我们使用convolutional recurrent neural networks（ConvRNN）作为PSGNet的feature extractor。一个ConvRNN是一个在每一层加了local recurrent cell以及从高层到低层加了long-range feedback connections的CNN。给一个输入的视频，第一次forward就是正常的CNN的forward，在计算出最后的feature maps之后，后来的forwards就通过ConvRNN的within-layer和top=down的recurrent connections来调整网络参数，从而达到将低层高分辨率的features和高层高非线性的features以一种可训练的方式结合起来的目的，从而使得输出的特征保留了这两种features的特性。在这篇文章里，作者使用了一个5层的ConvRNN，后续每一层都与第一层卷积层之间连接有connections。在经过三次forward之后，第一层卷积层经过activation之后的输出，就是PSG的base tensor。

**2.2.2 Graph Construction**

graph construction就是一个层次化的可学习的graph pooling和graph vectorization操作构成的序列。graph pooling在一个给定level的nodes $$\lbrace \pmb n_l \rbrace$$之间生成within-layer edges $$E_l$$，然后再将所得到的graph聚类从而生成新的一层的vertices $$(V_{l+1})$$以及child-to-parent edge structure $$(P_l)$$。这同时也自动决定了新的spatiotemporal registration $$S_{l+1}$$。graph vectorization通过这个edge structure来整合变型上一层的attributes $$A_l$$，从而为下一层的vertices生成新的attributes $$A_{l+1}$$。上述过程一直重复进行直到预先设置的结束指令（比如说设置好三层）。PSG的层数$$L$$，以及每一层的pooling和vectorization模块的设计，属于PSGNet的hyperparameters。

**2.2.2.1 Learnable Graph Pooling**

graph pooling操作背后的思想就是来推测PSG每一层的nodes（也就是scene elements），哪些是和其他nodes physically connected的从而该被聚集到一起作为下一层的新的vertex的。一个graph pooling模块通过从输入level的attributes vector $$A_l$$来预测within-level edges $$E_l$$来进行上述计算，有了edges之后，再聚类这一层的graph $$\mathcal G = (V_l, E_l)$$从而得到新一层的vertices $$V_{l+1}$$和child-to-parent edge structure $$P_l$$。每一层的edges $$E_l$$是通过一个learnable affinity function预测而来，这个function的输入是一对attribute vectors，$$A_l(v) \oplus A_l(w)$$，输出是这两个vertices $$v$$和$$w$$是连接起来的概率。这些概率最后会和一个thredshold比较从而决定是连接还是不连接，最后对于所有的vertex pairs生成一个binary adjacency matrix。不同的affinity functions和thresholding的值对结果造成的影响不大，这是因为是非监督学习的缘故。

我们在得到$$E_l$$之后，通过标准的Label Propagation (LP)算法（[Label propagation for clustering](https://arxiv.org/pdf/1709.05634.pdf)）来聚类graph $$\mathcal G_l$$。LP算法首先对于$$V_l$$里的每个vertex都给一个的独特的segment label（随机给的），然后对于每个vertex，找到它附近距离为1的那些vertex里和它label最像的vertex，并聚合，重复上述操作十次。所得到的clusters就被认为是新的vertices $$V_{l+1}$$，并且从前一层$$V_l$$里到后一层$$V_{l+1}$$里就对应定义了新的child-to-parent edges（也就是前一层哪些点聚类为了下一层新的点，那前一层的这些点都要和后一层这个新的点连上）。注意到上述的聚类过程并没有指明最后需要有多少个类（不像k-means），从而使得对于不同的场景可以找到不同数量的objects或者object parts。然而，LP的缺点是不能微分，所以graph pooling模块的参数不能够端到端的被上述PSGNet所优化，graph pooling的参数是利用self-supervised perceptual grouping loss来优化的，在下面的PSGNet training一节会说。上述过程见fig1所示。

LP算法的输入仅仅是edges $$E_l$$、目前这一层的nodes的个数$$\lvert V_l \rvert$$、以及一个超参数用来控制循环的次数。每个node初始化为是它自己的类，从而给出了一个标签的集合$$\left[ \lvert V_l \rvert \right]$$。然后对于$$q > 0$$，第$$q$$个循环的标签是从第$$q-1$$个循环的标签计算而来，对于第$$q$$个循环里的node $$v \in V_l$$，这个node的label被设置为和它相连的nodes在iteration $$q-1$$的时候的labels里最常见的那个label。如果有打成平手的情况，就从平手的里面随机选一个。重复上述过程若干个循环，将同一个label的vertices认为是一个node，从而构造出了新一层的nodes。而child-to-parent edges集合$$P_l$$可以理解为一个函数：$$V_l \rightarrow \left[m \right]$$，其中$$m$$是$$l+1$$层按照上述过程找到的nodes，也就是聚类中心，而$$P_l$$就是将这些聚类中心包含的$$l$$层的点与每个聚类中心连接起来。这种child-to-parent edges $$P_l$$可以被理解为某种依赖于输入的pooling操作。而且之后的graph vectorization也使用了这里聚类的信息，来为每层的node聚集其对应的上一层的child nodes的attributes的信息。

![pgr1]({{ '/assets/images/PGR-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1.*

**2.2.2.2 Graph Vectorization**

在有了新一层的vertices之后，接下来我们就要为每个vertex学习新的attribute vectors，$$A_{l+1}$$。它们编码了scene elements（也就是vertices）的物理信息。新的attribute vectors的计算有以下两个过程：利用之前学习到的上一层的graph结构$$\mathcal G_l$$来聚合上一层的vertices的attributes，然后将这些聚合的attributes通过MLPs来计算这一层的attributes。聚合上一层的attributes能保证PSG变得越来越high level的过程中仍然保存了场景里的重要信息，而利用MLP来预测新的attributes使得新一层的attributes不是简简单单低层的attributes的综合。我们将上述过程叫做vectorization，因为其编码了一个场景的spatiotemporal的信息，而且每个vertex一个attribute。

聚合上一层的attributes是通过perceptual groupings child-to-parent edges $$P_l$$以及registration $$S_{l+1}$$做到的。graph vectorization通过计算每个新的vertex的上一层的child nodes的means、variances、first-order spatial moments和higher-oder statistics来实现聚合上一层nodes的attributes这个操作的。为了保留低层graph levels的更多信息，graph vectorization模块还利用$$S_{l+1}$$在base tensor上的segments上计算了statistics，包括这些segments的1D boundaries和它们四个象限（见fig1的top left部分）。

通过这些聚合的attributes vectors，更多的attributes被graph convolution在$$V_{l+1}$$ vertices的fully-connected graph上计算出来。这使得信息能够以一种可学习的方式在任何两个nodes之间传递。在这篇文章里，上述的graph convolutions是通过MLPs来实现的，$$H_{l+1}^{new}$$。从而，PSG的第$$l+1$$层的attribute vectors通过这个公式来给出：

$$A_{l+1}(v) = A_{l+1}^{agg}(v) \oplus \frac{1}{\lvert V_{l+1} \rvert} \Sigma_{w \in V_{l+1}} H_{l+1}^{new}(A_{l+1}^{agg}(v), A_{l+1}^{agg}(w))$$

![pgr2]({{ '/assets/images/PGR-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. PSG representation和PSGNet结构的一个总览图。棕色的盒子表示的是PSGNet的三个stages：（1）利用一个ConvRNN从视频输入里提取features；（2）从利用ConvRNN提取的features上构造graph；（3）graph rendering用来做端到端的训练。第二个stage，构造graph的部分，由一对可学习的模块组成：graph pooling和graph vectorization，它两一起作用从上一层PSG level生成新的PSG level。graph pooling将上一层PSG level里的nodes分割为新的cluster，表示新的nodes，而graph vectorization则通过总结前面以及更前面的nodes的attributes来为新的nodes计算attributes。图片的中间下面还显示了三个level的PSG。*


**2.3 PSGNet Decoding and Training**

PSGNet training的目的就是使得PSG encoder所得出的PSG中间层特征能够表示输入场景视频的视觉以及物理结构特征。在一个经典的encoder-decoder框架下，中间层特征将会被喂给一个可训练的decoder神经网络用来渲染输出，输出将会以自监督或者ground truth监督的形式进行比较从而训练网络。但是，这个标准的过程并不能保证我们的PSG中间层特征就反映了输入的物理结构，因为只要decoder组有足够的capacity以及我们拥有足够的数据，那么即使中间层特征是entangled，unstructured的，模型仍然可以使得loss变得很小，也就是说训练loss很小，但所获得的中间层PSG特征并不好。于是有一些非监督的scene decomposition方法regularize训练过程，或者加上了结构化约束来做到将scenes分割为离散的objects：[Monet: Unsupervised scene decomposition and representation]()，[Multi-object representation learning with iterative variational inference]()，[beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework]()。

PSGNet不仅仅给encoder-decoder架构里的中间层特尔在加了约束（也就是我们所用的PSG数据结构，将场景表示为层次化的图），还使用没有参数的decoders来渲染输出结果，从而迫使PSGs要更加显式、更加丰富的表示场景里的objects以及它们的性质，因为此时没有有力的decoder来帮助渲染出好的结果了（如果decoder很给力的话，那即使中间层特征差了点，也能渲染出好的结果）。PSGNet的decoders使用paint by numbers的策略，使用预测到的nodes的attributes $$\lbrace A_l \rbrace$$作为颜料（paint），使用scene segmentations $$\lbrace S_l \rbrace$$作为regions来填上这些paints或者去做reconstruction（paint by numbers的意思就是一张图每个位置都有数字，凭借数字找到对应的颜料，从而给这个位置涂色，在这里就是用registration来表示每个node对应的位置，而这个node的attributes就是这个位置该使用的颜料，也就是该涂的颜色）。这个渲染的过程没有任何可训练的参数，PSGNets就被强迫让PSG显式的编码这些scene properties（比如说每个node对应着每个object，从而这个node的attributes就应该显式的表示出来这个node对应的像素点处的RGB值，深度等信息，只有这样才能够正确的得出渲染结果）。我们使用了几种decoding的方式：（1） quadratic texture rendering（QTR），fig1 top right部分，它使用一个quadratic function来计算每个vertex $$v$$的attributes $$A_l$$，并将其涂色到该vertex由registration对应的base tensor对应的pixel的位置处。（2）quadratic shape rendering（QSR），fig1 bottom right部分，它预测的是一个PSG node $$v$$会在原始的输入（base tensor）上对应的2D轮廓。QSR利用[Cvxnets: Learnable convex decomposition]()里提到的方法使用一系列quadratic signed distance function constraints的intersection来构造这样的shape，而这个构造所需要的参数就由这个node的attribute所提供。注意到所生成的2D轮廓并没直接用到这个node的registration $$S_l$$，这点和QTR不一样。

**2.3.1 Training the Feature Extractor and Vectorization Modules with Rendering Losses**

每一个QTR和QSR都使用PSG的node的attribute向量里不同的部分来渲染结果（可能会使用多个QTR或者QSR），而使用的loss function决定了这部分attribute向量将会encode场景什么样的信息。在这篇文章里，我们对于QTR这个任务采用自监督的方式，也就是将PSG每一个level的输出都通过某个QTR进行渲染，得到的结果是RGB图片，以及这张图片在输入视频里的相对时间信息（也就是位于第几帧），loss就是简单的$$L_2$$ loss，与原输入进行对比。我们还使用了一系列的QSR任务，它们只作用在最顶层的PSG level，从而鼓励这些QSR所对应的node的部分attribute向量
编码了场景里objects的轮廓信息。这里的loss使用的是每个QSR输出的每个像素点位置的segment index，和PSG给出的ground truth，registration $$S_L \left[i,j \right]$$之间的一个softmax cross-entropy loss（registration $$S_L$$实际上就是一个大小为原输入图片大小的矩阵，而矩阵每个点处的值就是其对应的node的值，比如说如果是node $$i$$，那对应的值就是$$i$$，而QSR利用每个node计算出来的轮廓，也就是shape，其轮廓内部也就对应这个node对应的像素点，也应该填入这个node对应的值，所以说它们量能够计算一个softmax cross-entropy loss。最后，我们利用监督学习的方法利用由数据集提供的实际的depth和surface normal vector images，用QTR来使得PSG显式的编码场景的几何信息。这些渲染的loss的反向传播可以训练ConvRNN这个feature extractor以及graph vectorization模块，但是因为graph pooling模块的LP操作并不可微分，所以我们还得单独训练graph pooling部分。

**2.3.2 Training Affinity Functions with Perceptual Grouping Principles**

每个graph pooling模块都需要一个loss function来优化它的affinity functions。在这篇文章里，我们使用了四种不同的loss functions来编码四种聚类思想：（1）attribute similarity（P1）：那些attribute很相近的nodes应该被聚类在一起，因为它们很可能是从同一个object来的；（2）statistical co-occurence（P2）：那些经常一起出现的nodes应该被聚类在一起，因为这表明它们和可能来自于同一个object。文中使用一个VAE来编码attribute pairwise differences，并且使用reconstruction loss作为衡量affinity的一个倒数（也就是loss越大，affinity就越小）。如果一对node是经常一起出现的，那他们在训练中会更经常被见到，所以reconstruction loss要比没关系的两个node构成的pair要低；（3）motion-driven similarity（P3）：那些一起移动的nodes应该被聚类在一起，而不需要考虑这些node的appearance信息，因为一起移动的nodes很可能就是属于同一个object的；（4）Self-supervision from motion（P4）：在某一帧图片里的两个nodes，如果它们在之前的帧里被发现一起移动，那它们应该被聚类到一起。



### 2. [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://link.springer.com/content/pdf/10.1007/978-3-319-24574-4_28.pdf)

*Olaf Ronneberger, Philipp Fischer, Thomas Brox*

*MICCAI 2015*

这篇文章很古老，在现在看来很多内容已经不必再说或者已经过时甚至不正确。但文章里提出的U型结构的用于segmentation的网络结构还是很经典的。

![unet1]({{ '/assets/images/UNET-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. 一个U-net的网络结构示意图。每个蓝色的box表示的是一个多通道的feature map，通道数写在每个box上面，而每个box左下角标记了feature map的height和width。白色的box表示cropped的feature maps。箭头则按照图中右下角的标识来辨别。*

U-Net的左侧叫做contracting path，右侧叫做expansive path，因为左侧使得feature maps的width和height逐渐减小，而右侧使得feature maps的width和height逐渐增大。上述卷积层都没有加padding，所以一个stride是1，卷积核是3的卷积层会导致feature map的width和height各减小2。

网络右侧的up convolution层是这样的：先进行一层up sampling，然后再进行卷积核为2的卷积，可以使用tensorflow里的conv_transpose函数或者pytorch里的ConvTranspose2d函数来实现，比如说将大小为$$28 \times 28$$的feature map变成大小为$$56 \times 56$$的feature map，设置ConvTranspose2d里的size为2，stride为2，padding为0，就相当于在$$28 \times 28$$的每一行每一列之间都插入一行一列，就得到了一个$$55 \times 55$的feature map，然后再在周围加上大小为1的padding，从而得到$$57 \times 57$$的feature map，最后进行$$2 \times 2$$的卷积，步长为1，就得到了$$56 \times 56$$的feature map。ConvTranspose2d里的$$stride-1$$表示插入的行数和列数，$$size-padding-1$$表示padding的大小。而关于卷积的计算则和正常的卷积相同。

而网络的最后一层，也就是右侧的最后一层使用了卷积核为1的卷积，从而将通道数为64的feature map变成通道数为2的feature map，而不改变feature map的width和height。这个运算可以理解为：因为这个任务是做segmentation，所以每个像素点的位置都有自己的feature，长度为64，而这个卷积就是将每个像素点位置的特征综合起来，输出为2，表示的是类别，也就是这个像素点属于哪一类，黑还是白（也就是0还是1）。所以说实际上这个网络的输出就相当于是objects的boundaries的输出。而我们实际操作的时候，输出的类别可以大于2，这样就可以表示多个不同的objects的segmentation了。

文章中说，从左侧到右侧的feature mapconcatenation之前做的cropping是必要的，因为每次卷积都会导致边界信息的丢失，所以我们也不需要前面层的边界的信息通过concatenation的方式传递到后面的层，也就是使用centercropping。


### 3. [SeqFormer: Sequential Transformer for Video Instance Segmentation](https://arxiv.org/pdf/2112.08275.pdf)

[CODE](https://github.com/wjf5203/SeqFormer)

*ECCV 2022 Oral*


### 4. [In Defense of Online Models for Video Instance Segmentation](https://arxiv.org/pdf/2207.10661.pdf)

*ECCV 2022 Oral*


### 5. [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://proceedings.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html)

*NeurIPS 2021*


### 6. [GroupViT: Semantic Segmentation Emerges from Text Supervision](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_GroupViT_Semantic_Segmentation_Emerges_From_Text_Supervision_CVPR_2022_paper.pdf)

*CVPR 2022*

### 7. [End-to-End Referring Video Object Segmentation with Multimodal Transformers](https://arxiv.org/pdf/2111.14821.pdf)

[CODE](https://github.com/mttr2021/MTTR)

*CVPR 2022*

### 8. [Unsupervised Semantic Segmentation by Distilling Feature Correspondences](https://openreview.net/forum?id=SaKO6z6Hl0c)

*ICLR 2022*


### 9. [Real-Time High-Resolution Background Matting](https://grail.cs.washington.edu/projects/background-matting-v2/#/)

*CVPR 2021*


### 10. [D2NeRF: Self-Supervised Decoupling of Dynamic and Static Objects from a Monocular Video](https://d2nerf.github.io/)

[POST](https://d2nerf.github.io/)

*CVPR 2022*


### 11. [SCOPS: Self-Supervised Co-Part Segmentation](https://openaccess.thecvf.com/content_CVPR_2019/html/Hung_SCOPS_Self-Supervised_Co-Part_Segmentation_CVPR_2019_paper.html)

*CVPR 2019*


### 12. [Causal Discovery in Physical Systems from Videos](https://proceedings.neurips.cc/paper/2020/hash/6822951732be44edf818dc5a97d32ca6-Abstract.html)

*NeurIPS 2020*


### 13. [Neural Scene Decomposition for Multi-Person Motion Capture](https://openaccess.thecvf.com/content_CVPR_2019/html/Rhodin_Neural_Scene_Decomposition_for_Multi-Person_Motion_Capture_CVPR_2019_paper.html)

*CVPR 2019*


### 14. [Unsupervised Disentanglement of Pose, Appearance and Background from Images and Videos](https://ieeexplore.ieee.org/abstract/document/9340560)

*TPAMI 2022*


### 15. [PiCIE: Unsupervised Semantic Segmentation Using Invariance and Equivariance in Clustering](https://openaccess.thecvf.com/content/CVPR2021/html/Cho_PiCIE_Unsupervised_Semantic_Segmentation_Using_Invariance_and_Equivariance_in_Clustering_CVPR_2021_paper.html)

*CVPR 2021*


### 16. [Unsupervised Image Representation Learning with Deep Latent Particles](https://arxiv.org/pdf/2205.15821.pdf)

[POST](https://taldatech.github.io/deep-latent-particles-web/)

*ICML 2022*


### 17. [GANSeg: Learning To Segment by Unsupervised Hierarchical Image Generation](https://openaccess.thecvf.com/content/CVPR2022/html/He_GANSeg_Learning_To_Segment_by_Unsupervised_Hierarchical_Image_Generation_CVPR_2022_paper.html)

*CVPR 2022*


### 18. [Unsupervised Part Segmentation Through Disentangling Appearance and Shape](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Unsupervised_Part_Segmentation_Through_Disentangling_Appearance_and_Shape_CVPR_2021_paper.html)

*CVPR 2021*


### 19. [Adversarial Dual-Student with Differentiable Spatial Warping for Semi-Supervised Semantic Segmentation](https://arxiv.org/pdf/2203.02792.pdf)

*Arxiv 2022*


### 20. [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/pdf/1606.00915.pdf)

*TPAMI 2017*


### 21. [GANSeg: Learning To Segment by Unsupervised Hierarchical Image Generation](https://openaccess.thecvf.com/content/CVPR2022/html/He_GANSeg_Learning_To_Segment_by_Unsupervised_Hierarchical_Image_Generation_CVPR_2022_paper.html)

*CVPR 2022*


### 22. [SCOPS: Self-Supervised Co-Part Segmentation](https://openaccess.thecvf.com/content_CVPR_2019/html/Hung_SCOPS_Self-Supervised_Co-Part_Segmentation_CVPR_2019_paper.html)

*CVPR 2019*


### 23. [End-to-End Referring Video Object Segmentation with Multimodal Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Botach_End-to-End_Referring_Video_Object_Segmentation_With_Multimodal_Transformers_CVPR_2022_paper.pdf)

*CVPR 2022*

[CODE](https://github.com/mttr2021/MTTR)


### 24. [PiCIE: Unsupervised Semantic Segmentation Using Invariance and Equivariance in Clustering](https://openaccess.thecvf.com/content/CVPR2021/html/Cho_PiCIE_Unsupervised_Semantic_Segmentation_Using_Invariance_and_Equivariance_in_Clustering_CVPR_2021_paper.html)

*CVPR 2021*


### 25. [MonoCInIS: Camera Independent Monocular 3D Object Detection Using Instance Segmentation](https://openaccess.thecvf.com/content/ICCV2021W/3DODI/html/Heylen_MonoCInIS_Camera_Independent_Monocular_3D_Object_Detection_Using_Instance_Segmentation_ICCVW_2021_paper.html)

*ICCV 2021*



---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*

