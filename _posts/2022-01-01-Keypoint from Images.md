---
layout: post
comments: false
title: "Keypoint from Images"
date: 2020-01-01 01:09:00
tags: paper-reading
---

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---


## 2D keypoints from images (supervised)

### \[**ECCV 2016**\] [Stacked hourglass networks for human pose estimation](https://arxiv.org/pdf/1603.06937.pdf)

*Alejandro Newell, Kaiyu Yang, Jia Deng*

[CODE](http://www-personal.umich.edu/~alnewell/pose)


![hourglass0]({{ '/assets/images/HOURGLASS-0.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

这篇文章是2D keypoints detection的经典之一（因为它的Hourglass网络结构）。

hourglass网络的设计受启发于获取各个尺度的信息的需求。尽管局部信息对于识别比如faces或者hands等局部物体来说很重要，但最终的pose estimation需要对整个human body有一个全面的理解。人的orientations，四肢的arrangements，以及相邻关节的关系等等，这些信息都需要同一张图片不同尺度的信息才能良好的获得。hourglass网络是一个简单而又有效的设计，其能够将各个尺度的信息综合起来从而输出pixel-wise的预测。

网络必须要有某些机制来有效的处理和合并不同尺度的特征。有些研究者尝试使用独立的不同的网络来处理不同分辨率的图片输入，再将这些输入综合起来处理。而本文并不采用这种方法，本文使用的是一个单一的pipeline，使用skip connections来在各个分辨率下保存空间信息。

hourglass网络的设置如下：卷积层和max pooling层将输入图片的分辨率降低到一个很低的值。在每个max pooling层，在做max pooling之前，输入分叉为两部分，不进入max pooling的那部分会做更多的卷积操作，留着之后使用。最后在达到了最低分辨率之后，网络开始进行upsampling以及结合之前不同分辨率下的features的操作。为了将两个相邻分辨率的特征结合起来，作者对较低分辨率的feature map使用了最近邻upsampling方法，再加上较高分辨率的那个feature map，从而得到输出。hourglass网络的结构是对称的，之前每用max pooling降低分辨率一次，之后就使用upsampling提升分辨率一次，所以最终的分辨率会和输入图片分辨率一样。在之后，再加上两个卷积大小为$$1 \times 1$$的卷积层，从而得到网络最终的输出。网络的最终输出是一系列的heatmaps，每个heatmap表示的是每个keypoint在图片中每个像素点出现的概率值的大小。

上述一个hourglass模块如fig 1所示。

![hourglass1]({{ '/assets/images/HOURGLASS-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. 一个hourglass模块的介绍。图中每个box都是fig 2里的一个residual模块。*


![hourglass2]({{ '/assets/images/HOURGLASS-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. 左侧：一个residual模块。右侧：intermediate supervision process的一个解释。*


上述的hourglass结构里，每个小部分仍然具有改动的空间。不同的改动会造成最后的不同效果。作者使用了几种不同的layer设计，无非是在每个residual module里使用不同大小的卷积核，或者增加$$1 \times 1$$的卷积层。最后作者选定的是，在hourglass模块里只使用$$3 \times 3$$和$$1 \times 1$$的卷积核，而且每个模块都是residual模块。

从而最后网络的设计为，初始输入的图片分辨率为$$256 \times 256$$，为了节约计算成本，作者在将图片输入hourglass模块之前，先通过了一个带有残差链接的$$stride=2$$，$$padding=3$$，卷积核为7的卷积层，和一个max pooling层，从而分辨率变为$$64 \times 64$$，之后在进入上述所说的hourglass模块里。对于hourglass里所有层输出的feature maps，其通道数都是256。


本文提出的最终的网络结果是多个hourglass模块堆叠而成，前一个hourglass模块的输出是后一个hourglass模块的输入。这样的结构使得网络可以重新衡量初始的估计值，改进效果。而且使用这种结构，网络中间某个hourglass模块的输入也可以拿来计算loss，具体做法是，对于中间某个hourglass模块的输出，其经过$$1 \times 1$$卷积得到heatmap输出，从而可以拿来被计算loss，而这个heatmaps再经过$$1 \times 1$$的卷积层回到操作之前的通道数，再和这个hourglass的输出加起来，作为下一个hourglass模块的输入。具体过程如fig2右侧所示。

每个hourglass模块的权重并不共享，而且每个hourglass拿来计算的loss使用的是同一种loss以及同一个ground truth。

本文的evaluation是根据标准的Percentage of Correct Keypoints（PCK）metric来计算的，给定一个normalized的distance，落在ground truth这个distance以内的都算判断正确，而PCK metric计算的是判断正确的keypoints占的比例。本文使用了FLIC和MPII两个数据集来测试效果，对于FLIC数据集，这个distance是利用躯干的大小进行了normalize，而对于MPII数据集，这个distance是根据头的大小进行了normalize。

> 这篇文章并不能解决多人的问题，对于多人的情况，只会考虑最靠近图片中心的那个。如果要看多人的算法，可以去看OpenPose那篇论文。

> 本文的方法能够检测到被遮挡住的keypoints，但是检测的效果不是那么好，所以这仍然是一个需要被解决的问题。


本文详细阐述了使用hourglass模块构造成的网络对于human pose estimation任务的效果。intermediate supervision对于好的效果来说是很必要的。虽然说还有一些问题没有解决，但是本文提出的方法对于绝大多数情况下的human pose estimation的效果还是很好的。


### \[**CVPR 2017** $$\&$$  **TPAMI 2019**\] [OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields](https://arxiv.org/pdf/1812.08008.pdf)
*Zhe Cao, Gines Hidalgo, Tomas Simon, Shih-En Wei, and Yaser Sheikh*


[post](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

OpenPose是一个成熟的可以进行multi-person的2D关键点检测的算法（已商业化）

对于单人的pose estimation有很多论文已经做得很好了，但对于多人来说，有以下几个困难的地方：1）首先，我们并不知道图里到底有几个人，而且每个人所在的位置、大小都不清楚；2）其次，人与人之间可能存在干涉，比如说遮挡、关节的旋转等等，很难分清楚到底哪部分属于哪一个人；3）以往的论文里的算法的复杂度都会随着图里人数量的增加而增加，从而很难实现实时。

对于multi-person 2D pose estimation，top-down的方式很常见，也就是先检测图片里有几个人，然后对每个人实行pose estimation，因为单人的pose estimation已经做得很好了，这个算法并不复杂。但它存在着两个很大的问题：首先如果一开始检测人的时候就检测错了或者遗漏了，那之后是没有补救办法的；其次，这样的方法需要对检测出来的每个人都做单人pose estimation，这会使得算法的复杂度和人的数量成正比。所以说，bottom-up的方法也被提了出来，这种方法有能够解决上述两个问题的潜力。但之前的bottom-up方法仍然效率不高，因为它们在最后还是需要利用全局信息来辅助判断，从而要花不少时间。

这篇文章里利用Part Affinity Field实现了实时的multi-person 2D pose estimation。Part Affinity Field是一个2D向量的集合，表示的是四肢的位置和方向信息。利用bottom-up的方式，将detection和association结合起来（模型有两个主要部分，一个是PAF refinement，另一个是body part prediction refinement，而PAF就是实现association的方式）逐步推进，可以在利用很小的计算资源的情况下达到很好的效果。

![overview]({{ '/assets/images/OPENPOSE-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. 算法流程。输入是一张大小为$$w \times h$$的RGB图片，而输出为图中每个人的生理结构上的2D keypoints的位置。首先，一个feedforward网络输出一个集合$$S$$，用来表示各个身体部位位置的2D confidence maps，和一个part affinity fields（也就是2D的向量）的集合$$L$$用来表示各个身体部位之间的从属程度。集合$$S = (S_1, S_2, ..., S_J)$$有J个confidence maps，每个对应一个身体部位，其中$$S_j \in R^{w \times h}$$，$$j = \lbrace 1,2,...,J \rbrace$$。而集合$$L = (L_1, L_2, ..., L_C)$$有$$C$$个向量，每个对应一个肢体，其中$$L_c \in R^{w \times h \times 2}$$，而$$c \in \lbrace 1,...,C \rbrace$$。将身体部位的pairs描述为肢体（因为这里的身体部位就是一个keypoint，而keypoint pair就是将两个keypoint连起来，就表示了一部分肢体）。$$L_c$$里的每个点都是一个2D的vector。最终，confidence maps和PAFs（也就是集合S和集合L）通过greedy inference联系了起来用于输出图中所有人的2D keypoints位置。*

![algorithm]({{ '/assets/images/OPENPOSE-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. 上方：多人pose estimation。同一个人的身体部分被连了起来，也包括了脚的keypoints（大脚趾，小脚趾和脚后跟）。下左：关于连接右手肘和手腕的肢体的PAFs。颜色表明了方向。下右：在关于连接右手肘和手腕的肢体的PAFs的每个像素点处的2D向量包含了肢体的位置和方向信息。*


![architecture]({{ '/assets/images/OPENPOSE-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 3. multi-stage CNN的结构。前一部分的stage用来预测PAFs $$L^t$$, 而后一部分的stage用来预测confidence maps $$S^t$$。每个stage的输出和图片feature连接起来，作为下一个stage的输入。*

输入的图像通过一个CNN（用的是VGG-19的前10层）来生成一系列的feature maps $$F$$，之后再输出到第一个stage当中。在这个stage里，网络输出一个集合的PAFs $$L^1 = \phi ^ 1 (F)$$，其中$$\phi ^ 1$$表示stage1里用来inference的CNN。在之后的stages里，前一个stage的输出和原始的图像feature maps $$F$$连接起来作为输入，用来输出更加精确的结果$$L^t = \phi ^ t (F, L^{t-1}), 2 \leq t \leq T_p$$，其中$$\phi ^ t$$表示stage t里用来inference的CNN，$$T_p$$是PAF stage的总数。在$$T_p$$个PAF stage之后，再来计算confidence maps，利用的是最新的PAF结果：$$S^{T_p} = \rho^t (F, L^{T_p}), t = T_p$$，$$S^t = \rho^t (F, L^{T_p}, S^{t-1}), T^p < t \leq T_p + T_C$$，其中$$\rho^t$$指的是stage t用来inference的CNN，而T_C是confidence maps所需要循环的次数。

![PAF]({{ '/assets/images/OPENPOSE-4.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 4. 右前小臂肢体的PAF。尽管在一开始的stage里还有一些不清晰，但在之后的stage里可以看到PAF很清晰。.*

对于PAF的stage $$t_i$$和confidence map的stage $$t_k$$的loss function是：

$$ f_L^{t_i} = \Sigma_{c=1}^C \Sigma_p W(p) ||L_c^{t_i}(p) - L_c^{\ast}(p)||^2_2 $$

$$ f_S^{t_k} = \Sigma_{j=1}^J \Sigma_p W(p) ||S_j^{t_k}(p) - S_j^{\ast}(p)||^2_2 $$

其中$$L_c^{\ast}$$是PAF的ground truth，$$S_j^{\ast}$$是身体部分confidence map的ground truth，$$W$$是一个非0即1的二分掩码，如果某个位置没有标注就是0，有标注就是1。这个$$W$$是用来避免因为没有标注而导致的错误训练。我们在每个stage都使用loss function，用来解决梯度消失的问题，因为每个stage结尾都有loss function，对梯度的值进行了补充。从而整体的的loss function就是：

$$ f = \Sigma_{t=1}^{T_p} f_L^t + \Sigma_{t=T_p + 1}^{T_p + T_C} f_S^t $$


为了能够在训练过程中计算上述的$$f_S$$，我们从有标注的2D keypoints上生成confidence maps $$S^{\ast}$$的ground truth。一个confidence map是一个2D的矩阵，用来表示一个keypoint出现在一个像素点的概率值。理想状态下，如果图里只有一个人，那么每个confidence map应该只有1个峰（该keypoint没有被遮挡住的情况下）;如果有多个人，那么每个confidence map对于没有被遮挡住的这一类keypoint都应该有峰（比如说某个confidence map专门表示人的nose的keypoint）。

先来对于每个人$$k$$生成confidence map，$$S_{j,k}^{\ast}$$。$$x_{j,k} \in R^2$$表示人$$k$$的身体部分$$j$$在图中位置的ground truth。从而$$S_{j,k}^{\ast}的位置$$p \in R^2$$处的值就是：

$$ S_{j,k}^{\ast}(p) = exp(-||p-x_{j,k}||^2_2 / \sigma^2) $$

其中$$\sigma$$控制峰的大小。从而整个图片（可能包含多个人）的ground truth就是：

$$ S_j^{\ast}(p) = max_k S_{j,k}^{\ast}(p) $$


![cal]({{ '/assets/images/OPENPOSE-5.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 5. 身体部位association策略。（a) 几个人的两种身体部分（也就是keypoint）分别用红蓝点表示，而所有的点之间都连上了线。（b) 利用中间点进行连线。黑线是正确的，绿线是错误的，但他们都满足连接了一个中间点。(c) 利用PAF来连接，黄色的箭头就是PAF的结果。利用肢体来表示keypoint的位置和keypoint之间的方向信息，PAF减少了错误association的可能性。.*

给定一些已经检测到了的body parts（Fig 5里的红色和蓝色的点），那我们该如何将它们组合起来从而构建未知数量的人的肢体呢？我们需要对每一对body part keypoints都有一个confidence measure，也就是说，measure它们是否属于同一个人。一种可能的方式就是检测这一对keypoints的中间是否还有附加的midpoint。但是当人聚集在一起的时候，很容易出错。这种方式之所以不好是因为1）它仅仅有位置信息，并没有一对keypoint之间的方向信息；2）它仅仅用了midpoint，而不是这两个keypoints之间的所有部分当成一个肢体来使用。

Part Affinity Fieds (PAFs)解决了这些问题。它对于每一对keypoints构成的肢体提供了位置和方向信息。每一个PAF都是一个2D的vector field，在Fig 2里有显示。对于每个肢体的PAF的每个位置的值，其都是一个2D的向量，包含了位置和这个肢体一个keypoint指向另一个keypoint的方向。每个类别的肢体都有一个对应的PAF（由对应的body part keypoints对组成）。

![fig]({{ '/assets/images/OPENPOSE-6.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

考虑fig 6所示的一个简单的肢体。$$x_{j_1, k}$$和$$x_{j_2, k}$$是人$$k$$的身体部位$$j_1$$和$$j_2$$的ground truth，而这两个部位组成了肢体$$c$$。对于肢体$$c$$上的一点$$p$$，$$L_{c,k}^{\ast}(p)$$是一个单位向量，从$$j_1$$指向$$j_2$$；对于其它的点，$$L_{c,k}^{\ast}(p)$$的值都是0。

为了能在训练过程中计算$$f_L$$的值，我们需要定义PAF的ground truth，也就是对于人$$k$$，$$L_{c,k}^{\ast}$$在$$p$$点的值为$$L_{c,k}^{\ast}(p) = v$$ if $$p$$ on limb $$c, k$$ and $$0$$ otherwise。

这里

$$v = (x_{j_2, k} - x_{j_1, l}) / ||x_{j_2, k} - x_{j_1, l}||$$

是肢体的有方向的单位向量。一个肢体上的点不仅仅只有两个keypoints连线上的，而是有一个距离阈值，比如说：

$$ 0 \leq v (p - x_{j_1,k}) \leq l_{c,k}$$ 和 $$|v_{verticle} (p - x_{j_1,k})| \leq \sigma_l$$

其中肢体宽度$$\sigma_l$$自定义的，肢体长度由两个keypoints决定，也就是

$$l_{c,k} = ||x_{j_2, k} - x_{j_1, k}||$$

$$v_{verticle}$$是垂直于$$v$$的。

而整个图片的PAF的ground truth是对于所有人取了均值：

$$ L_c^{\ast}(p) = 1/n_c(p) \Sigma_k L_{c,k}^{\ast}(p) $$

在测试过程中，我们通过计算连接两个keypoints的线段间的PAF的积分来衡量这两个keypoints是否构成了一个肢体。对于两个身体部分$$d_{j_1}$$和$$d_{j_2}$$，我们计算：

$$ E = \int_{u=0}^{u=1} L_c(p(u)) (d_{j_2} - d_{j_1})/||d_{j_2} - d_{j_1}|| du $$

其中$$p(u) = (1-u) d_{j_1} + u d_{j_2}$$。在实践中，我们通过等距离采样来近似这个积分值。


对于每个身体部位keypoint的location，我们都有好几个备选的值，这是因为图中有多个人或者因为计算错误。而这些keypoints组成的肢体就会有很多种可能了。我们用5.4里定义的积分来计算每个肢体的积分值。从而问题变成了，如何在众多的有着不同积分值（也就是score）的肢体集合中，选择合适的肢体并将其正确连接起来，而这是个NP-hard的问题，如fig 7所示。

![fig]({{ '/assets/images/OPENPOSE-7.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 7. Graph matching。(a) 原始的图片，已经有了身体部位keypoint标注了。(b) K-partite graph。(c) 树状结构。(d) 二分图。*

在这篇文章里，我们使用一种greedy relaxation的方法，持续性的产生高质量的匹配。我们猜测这种方法有效的原因是上述计算的积分值（也就是每个肢体的score）潜在的含有global信息，因为PAF的框架具有较大的感受野。

具体来说，首先，我们获得整张图片身体部分keypoint的集合，$$D_J$$，其中$$D_J = \lbrace d_{j_m}: j \in \lbrace 1, ..., J \rbrace, m \in \lbrace 1, ..., N_j \rbrace \rbrace$$，$$N_j$$是身体部位$$j$$的候选数量，而$$d_j^m \in R^2$$是身体部位$$j$$的第$$m$$个候选位置。我们需要将每个身体部位keypoint连接到属于同一个人的同一个肢体的其它身体部位keypoint上，也就是说，我们还需要找到正确的肢体。我们定义$$z_{j_1, j_2}^{m,n} \in \{0, 1\}$$来表示两个身体部位keypoint的候选，$$d_{j_1}^m$$和$$d_{j_2}^n$$是否连在一起，我们的目标是为$$Z = \{z_{j_1, j_2}^{m,n}: j_1, j_2 \in \{1, ..., J\}, m \in \{1, ..., N_{j_1}\}, n \in \{1, ..., N_{j_2}\}$$找到最优的值。

如果我们考虑一个特定的keypoint的pair $$j_1$$和$$j_2$$（比如说neck和right-hip），叫做c-肢体，而我们的目标是：

$$ \max\limits_{Z_c} E_c = \max\limits_{Z_c} \Sigma_{m \in D_{j_1}} \Sigma_{n \in D_{j_2}} E_{m,n} z_{j_1, j_2}^{m,n}$$

$$s.t., \forall m \in D_{j_1}, \Sigma_{n \in D_{j_2}} z_{j_1, j_2}^{m, n} \leq 1$$

$$ \forall n \in D_{j_2}, \Sigma_{m \in D_{j_1}} z_{j_1, j_2}^{m,n} \leq 1$$

其中，$$E_c$$是所有的c-肢体的积分值的和（可能有多个c-肢体，因为可能有多个人），$$Z_c$$是$$Z$$的只关于c-肢体的子集，$$E_{m,n}$$是keypoint $$d_{j_1}^m$$和$$d_{j_2}^n$$之间的定义的积分值，上述要优化的目标的条件，使得我们所学习到的结果里不会有两个肢体公用同一个keypoint。我们可以用Hungarian算法来获取上述优化的结果。

现在我们考虑所有的肢体，那么上述优化的式子即是需要考虑整个$$Z$$并且需要计算所有肢体的所有可能结果，计算$$Z$$是一个K-维的匹配问题（K是肢体的数量）。这个问题是个NP-hard的问题，有很多relaxations的算法存在。在我们这篇论文中，我们添加了两个relaxation。首先，完整的图会对于每两个不同类别的keypoint都有edge，而我们将这个图简化为其能表示人的pose的spanning tree就可以，而多余的edge就不要了。其次，我们将上述K-维的匹配问题解构为一系列二分匹配的子问题并且独立的解决这些问题，所利用的就是每个spanning tree的相邻的两个node所对应的值以及它们之间的连线，所以说是独立的。第二个relaxation之所以可行，直觉上来说，spanning tree里相邻的两个node之间的关系是由PAF网络学习到的，而非相邻的两个node之间的关系是由CNN网络学习到的。

有了上述两个relaxations，我们的问题被简化为：

$$ \max\limits_{Z} E = \Sigma_{c=1}^C \max\limits_{Z_c} E_c $$

从而我们将这个优化问题分解为独立的每个pair的优化问题，而这个在之前所述，可以用Hungarian算法解决。我们再将有共同keypoint的肢体联合起来，这样其就表示出了一个完整的人的pose，或者说骨架。我们的第一个relaxation，将完整的图简化为spanning tree使得整个算法获得了很大程度的加快。

我们目前的模型仍然有多余的PAF连接（比如说耳朵和肩膀的连接，手腕和肩膀的连接等）。这样冗余的连接使得我们的算法对于人群很密集的时候准确度较高。对于冗余的PAF连接，也就是有冗余的肢体，我们在parsing算法进行一些简单的修改就行。




### \[**CVPR 2019**\] [D2-Net: A Trainable CNN for Joint Description and Detection of Local Features](https://openaccess.thecvf.com/content_CVPR_2019/papers/Dusmanu_D2-Net_A_Trainable_CNN_for_Joint_Description_and_Detection_of_CVPR_2019_paper.pdf)

*Mihai Dusmanu, Ignacio Rocco, Tomas Pajdla, Marc Pollefeys, Josef Sivic, Akihiko Torri, Torsten Sattler*


这篇文章解决了在复杂图片条件下找到可靠的像素点之间的对应关系的问题。这篇文章提出了一个简单的CNN网络用来完成两个任务：dense feature descriptor和feature detector。通过将detection安排在后续操作中，这篇文章方法所获得的keypoints和其它基于low-level features的早期detection的效果相比要好很多。这个模型的训练label仅仅需要像素对应关系这样一个annotations就可以。

在images之间建立像素点的对应关系是一个很基础的CV问题，其可以被应用在3D CV，视频压缩，跟踪，定位等等任务里。稀疏的局部特征是预估对应关系的一个重要方法。这些方法基于一种detect-then-describe的模式，先利用一个feature detector来找到一批keypoints，然后根据这些keypoints和其周围的点构建一些image patches，再利用descriptor对这些image patches给出feature description，作为这些keypoints的features。稀疏的局部特征方法有以下几个优势：1）对应关系可以通过nearest neighbor search和欧式距离很快的被找到。2）而且稀疏的特征所消耗的内存小，使得方法能够在大规模问题上应用。3）而且基于这种方法的keypoint detector一般都会考虑low-level的图像信息，比如说corners。从而局部特征可以被精确的在图像中定位，这对于很多后续任务比如说3D reconstruction来说是很重要的。

稀疏局部特征方法在很多图像条件下都得到了很好的使用。但是他们在极端的图像appearance变化的情况下就不好使了，比如说白天和夜里，或者季节变化，或者很弱纹理的场景。研究表明稀疏局部特征方法在这些情况下不好使的原因是keypoint detector检测到的keypoint的repeatability很差：因为按照上述描述的detector-then-describe流程，local descriptor会考虑keypoint和周围点的一块较大范围内的信息，从而潜在的会encode更high-level的结构，但keypoint detector只会考虑一个keypoint点的信息，范围太小。从而，这些keypoints的detections在appearance变化很大的时候就不稳定了。这是因为low-level信息会被图像里的low-level statistics影响更大，比如说像素点的intensity。

但是不管怎样，实验表明即使keypoints detection不是那么的稳定，local descriptors仍然可以被较好的匹配。因此，那些放弃了detection stage转而使用dense descriptors（也就是每个点都学习一个feature descriptor）并进行dense匹配的方法在复杂的条件下效果更好。当然，这种dense的方法会导致计算量增大。

在这篇文章里，作者旨在同时做好这两件事情，也就是，找到在复杂条件下也能有很好效果的稀疏的特征集合，而且设计高效的匹配算法。为了达到这个目标，作者提出了一个describe-and-detect的方法来进行稀疏局部特征detection和description：和之前先利用low-level信息进行feature detection的方法不同，这个方法将detection stage放在了后面。该方法首先利用CNN计算feature maps，再利用这些feature maps来计算descriptors（在每个特定的像素点，直接取所有通道的值构成的向量为这个像素点的feature description）和检测keypoints（在feature maps上找到局部最大值）。通过这样操作，feature detector和feature descriptor就紧密相连了。由detector检测到的keypoints就会是那些匹配效果很好的descriptors了。同时，使用CNN深层输出的feature maps使得我们能够基于high-level的信息来计算feature detection和feature description。实验表明这种方法和dense方法相比所需要的内存要小得多，同时其也比之前的detect-then-describe方法对于复杂条件下的images的匹配效果要好得多，和dense方法效果差不多甚至更好。

当然，这篇文章的方法也是有缺点的：和传统的detection-then-describe的稀疏局部特征方法相比，文章的方法效率不高，因为需要dense descriptors。而且基于high-level的detection可以获得稳定的输出但却获得不了精确的定位，当然文中的方法对于视觉定位等后续任务来说已经足够精确了。


和传统的使用了两个stages的pipeline的detect-then-describe方法不一样，这篇文章提出使用dense feature extraction来获得一个同时是detector和descriptor的representation。因为detector和descriptor使用了同样的representation，文章将这个方法叫做D2。方法流程如fig 1所示。

![performance]({{ '/assets/images/D2.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. detect-and-describe (D2) network。用一个feature extraction CNN $$\mathcal F$$来获取图片的feature maps，这个feature maps起到两个作用：(i) 在每个位置$$(i,j)$$将所有通道的值连起来，获取了这个点的local descriptor $$d_{ij}$$；(ii) detections通过在feature map上使用一个non-local-maximum suppression，再在每个descriptor间使用一个non-maximum suppression来获得。keypoint detection score $$s_{ij}$$是通过一个soft local maximum score $$\alpha$$和一个每个descriptor的ratio-to-maximum score $$\beta$$计算得来的。*

第一步是在输入图片$$I$$上使用一个CNN $$\mathcal F$$来获取一个3D的tensor $$F = \mathcal F (I), F \in R^{h \times w \times n}$$，其中$$h \times w$$是feature maps的resolution，而$$n$$是通道数。

关于这个3D tensor $$F$$的最直观的解释就是一个descriptor向量$$\pmb{d}$$的dense集合：

$$d_{ij} = F_{ij}, \pmb{d} \in R^n$$

其中$$i=1, \cdots, h, j=1, \cdots, w$$。利用这些descriptors之间的欧式距离可以很容易的计算两张图片之间的像素对应关系。在训练阶段，这些descriptors被训练为不同图片对应的同样的点具有相似的descriptors，即使是图片有很大的appearance变化。在实践中，每个descriptors被除以了它的$$L_2$$ norm来归一化这个descriptor。

对于3D tensor $$F$$的另一个不同的解释就是将它们看成一系列2D responses $$D$$的集合：

$$D^k = F_{::k}, D^k \in R^{h \times w}$$

其中$$k=1,\cdots,n$$。在这个解释里，feature extraction函数$$\mathcal F$$可以被看成$$n$$个不同的feature functions $$\mathcal D^k$$，每个都输出一个2D的feature map，$$D^k$$。再之后，我们利用这些feature maps来选择keypoints的locations，如下所述。

在传统的feature detectors里，比如说DoG，detection map会通过一个空间的non-local-maximum suppression来被稀疏化。但是在这篇文章的设定下，有多个detection maps $$D^k, k=1,\cdots,n$$，在他们中的任何一个上面都可以进行detection。因此，对于一个点$$(i,j)$$，如果它想要成为一个keypoint，则要满足：

$$(i,j)$$ is a keypoint $$\iff D_{ij}^k$$ is a local max in $$D^k$$，而且$$k = argmax_t D{ij}^t$$

也就是说，对于每个像素点$$(i,j)$$，上述设置会让我们找到最显著的detector $$D^k$$（channel selection），然后再验证其在$$D^k$$上是否是一个局部最大值点。

在实践中，上述的hard detection procedure会被softened来满足back propagation。首先我们定义一个soft local-max score：

$$\alpha_{ij}^k = \frac{exp(D_{ij}^k)}{\Sigma_{(i^{'},j^{'}) \in \mathcal N(i,j)} exp(D_{i^{'}j^{'}}^k)}$$

其中$$\mathcal N(i,j)$$是像素点$$(i,j)$$周围包括自己一共9个点构成的集合。

然后我们再来定义soft channel selection，对于每个descriptor计算一个ratio-to-max来模仿每个通道的non-maximum suppression：

$$\beta_{ij}^k = D_{ij}^k / max_t D_{ij}^t$$

为了将上述两个criteria都考虑进来，我们对于所有的feature maps来最大化上述两个scores的积来获得一个简单的score map：

$$\gamma_{ij} = max_k (\alpha_{ij}^k \beta_{ij}^k)$$

最后，点$$(ij)$$的soft detection score $$s_{ij}$$通过一个image-level的归一化来得到：

$$s_{ij} = \gamma_{ij} / \Sigma_{(i^{'}, j^{'})} \gamma_{i^{'}j^{'}}$$

尽管CNN feature extractor $$\mathcal F$$本身就具有一定的scale invariance的能力，但是其本身对于scale并不是invariant的，而且匹配也会在视角有大的变化的情况下失败。

为了获取对于scale变化鲁棒的features，这篇文章使用了image pyramid（其在hand-crafted local feature detectors里常用）。而image pyramid仅仅在测试的时候使用。

给定一张图片$$I$$，构造一个包含了三种不同resolutions $$\rho = 0.5, 1, 2$$（对应于一般，不变，两倍分辨率的情况）的image pyramid $$I^{\rho}$$，并且在这三个分辨率的图像上都提取feature maps，记为$$F^{\rho}$$。

然后再得到$$\tilde F^{2} = F^{2} + F^{1} + F^{0.5}, \tilde F^{1} = F^{1} + F^{0.5}, \tilde F^{0.5} = F^{0.5}$$，在$$\tilde F^{\rho}$$上提取features。而不同分辨率的图片会得到大小不一样的feature map，从而在加之前使用bilinear interpolation来使得尺寸相同。

在实际操作的时候，对于最小分辨率的图片，标注下来keypoints检测的位置，然后上采样到上一个分辨率的位置上，如果在这个分辨率下keypoint检测的位置不在这个区域里（$$2 \times 2$$的区域），那就忽略这个keypoint。



### \[**ICCV 2019**\] [Joint Learning of Semantic Alignment and Object Landmark Detection](https://openaccess.thecvf.com/content_ICCV_2019/html/Jeon_Joint_Learning_of_Semantic_Alignment_and_Object_Landmark_Detection_ICCV_2019_paper.html)


### \[**CVPR 2020**\] [Transferring Dense Pose to Proximal Animal Classes](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sanakoyeu_Transferring_Dense_Pose_to_Proximal_Animal_Classes_CVPR_2020_paper.pdf)

[POST](https://gdude.de/densepose-evolution/)
[CODE](https://github.com/asanakoy/densepose-evolution)


### \[**WACV 2021**\] [Conditional Link Prediction of Category-Implicit Keypoint Detection](https://openaccess.thecvf.com/content/WACV2021/html/Yi-Ge_Conditional_Link_Prediction_of_Category-Implicit_Keypoint_Detection_WACV_2021_paper.html)


### \[**ICCV 2021**\] [On Equivariant and Invariant Learning of Object Landmark Representations](https://openaccess.thecvf.com/content/ICCV2021/html/Cheng_On_Equivariant_and_Invariant_Learning_of_Object_Landmark_Representations_ICCV_2021_paper.html)


### \[**WACV 2021**\] [Learning of Low-Level Feature Keypoints for Accurate and Robust Detection](https://openaccess.thecvf.com/content/WACV2021/html/Suwanwimolkul_Learning_of_Low-Level_Feature_Keypoints_for_Accurate_and_Robust_Detection_WACV_2021_paper.html)



## 2D keypoints from images (Unsupervised)

### \[**ICCV 2017**\] [Unsupervised learning of object landmarks by factorized spatial embeddings](https://openaccess.thecvf.com/content_ICCV_2017/papers/Thewlis_Unsupervised_Learning_of_ICCV_2017_paper.pdf)

[CODE](https://github.com/alldbi/Factorized-Spatial-Embeddings)


自动学习一个object category的结构信息仍然是CV领域未解决的问题。在这篇文章里，作者提出一个新的无监督的方式来学习一个类别的物体的keypoints，从而表示这个类别物体的结构。作者的方法基于factorizing image deformations，这种deformation可能是由viewpoint变化引起的，或者是因为物体本身的形变引起的。作者提出的方法是认为deformation前后的图片的keypoints具有consistency的性质。作者还表明，用这种方法所学习到的keypoints，即使不需要加上限制条件使得同一个类别的不同图片（也就是不同物体）之间的keypoints具有对应关系，它们就已经自动对应上了。

为了获取一个对物体更深的理解，我们需要对它们的和视角无关的本质的结构信息进行建模。通常来说，这种结构信息会显式的用landmarks，parts或者skeletons来表示。给定足够的人工标注，我们是可以训练出来一个深度神经网络对于输入的图片，给出图片中物体的上述的结构信息的。但是，不使用人工标注信息来使得模型具有从图片中学习到这种结构信息的这个任务仍然是没有被解决的。

在这篇文章里，作者提出了一个新的方法使用无监督学习的方式来从输入图片中学习和视角无关的物体的结构信息，如fig 1所示。

![fse1]({{ '/assets/images/fse1.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*fig1. 作者提出了一个新的方法，可以在没有任何标注的情况下学习视角无关的keypoints。这个方法使用了一个viewpoint factorization的过程，其学习了一个可以进行image deformation的keypoint detector。这个模型可以用来学习rigid或者deformable objects。*


$$S \subset \mathbb R^3$$表示一个物体的3维surface，比如说一只鸟，$$\pmb x: \Lambda \rightarrow \mathbb R$$表示的是这个物体的一张图片，其中$$\Lambda \subset \mathbb R^2$$是image domain，如fig2所示。$$S$$是物体的一个固有的特性，和$$\pmb x$$是无关的。作者考虑的是学习一个函数$$q = \Phi_S(p; \pmb x)$$将$$S$$上的点$$p \in S$$映射到对应的图片$$\pmb x$$上的某个点$$q \in \Lambda$$上去。

![fse2]({{ '/assets/images/fse2.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*fig2. 物体结构的建模过程。。*

作者提出了一个利用viewpoint factorization来自动学习$$\Phi_S$$的方法。为了实现这个目标，我们考虑从另一个角度来看这个物体从而获得的另一张图片$$\pmb{x^{'}}$$。利用$$g$$来表示由于视角变化导致的图片歪曲，从而$$\pmb{x^{'}} \approx \pmb x \circ g$$。利用$$\Phi_S$$，我们可以将$$g$$$分解为：

$$g = \Phi_S(\dot; \pmb{x^{'}}) \circ \Phi_S(\dot; \pmb x)^{-1} \tag{1}$$

也就是说，我们将这个歪曲$$g: q \rightarrow q^{'}$$分解为先找到图片$$\pmb x$$里的点$$q$$对应在$$S$$上的点$$p=\Phi_S^{-1}(q ;\pmb x)$$，然后再找到这个点$$p$$在图片$$\pmb{x^{'}}$$上对应的点$$q^{'} = \Phi_S(p;\pmb{x^{'}})$$。

公式1表示的factorization也可以用下述等价的约束条件来表示：

$$\forall q \in S: \Phi_S(p; \pmb x \circ g) = g(\Phi_S(p; \pmb x)) \tag{2}$$

这个约束条件表明$$S$$上所检测到的点$$p$$需要随着视角的变化仍然保持一致。

为了学习这个函数$$\Phi_S$$，作者将这个函数利用一个深度神经网络来表示，并采用一种Siamese的结构，输入是$$(\pmb x, \pmb{x^{'}}, g)$$。注意到，如果我们任意给定一个物体的两个视角的图片，那么$$g$$一般是不知道的，所以说与其再利用某种方法来估计$$g$$，不如直接随机的构造出$$g$$，再利用$$g$$从$$\pmb x$$上构造出$$\pmb{x^{'}}$$。

>尽管训练的过程使用的是同一张图片和这张图片经过transformation之后得到的歪曲的图片作为训练图片对，但是训练好的模型仍然能够在同一种类物体的不同图片之间学习到keypoints的对应关系。对于那些角度变化特别大的情况，则不能保证效果，这就是future works了。


上述描述的是rigid物体的建模，也就是说$$S$$是不变的，而其也可以直接拓展到deformable的物体上去。假设$$S$$经过了某些deformation $$w$: \mathbb R^3 \rightarrow \mathbb R^3$$。从而形变后的surface就是$$S_{'} = \lbrace w(p):p \in S \rbrace$$。作者同时还提出了一个共用的surface $$S_0$$，其表示的是这个类别的object的一个标准的surface，也就是说其它的surface都可以从这个surface变形而来。将从$$S_0$$到$$S$$的deformation记为$$\pi_S$$，那么对于$$S_0$$上任意一个点$$r$$，就有$$\pi_S(r) \in S$$，而且$$\forall w: w(\pi_S(r)) = \pi_{S^{'}}(r)$$。然后，对于$$S_0$$上的点$$r$$到图片$$\pmb x$$上的点$$q$$的映射$$\Phi(r, \pmb x)$$，可以用$$\Phi_S$$来表示：$$\Phi(r; \pmb x) = \Phi_S(\pi_S(r); \pmb x)$$。从而公式2就写成：

$$\forall r \in S_0: \Phi(r; \pmb x \circ g) = g(\Phi(r; \pmb x)) \tag{3}$$

这也就是说即使是由形变造成的物体变化，变化前后所检测到的物体的keypoints也是要对应的。


除了物体的形变，上述的构造对于同一个类别的不同物体之间的差异也是可以解释的。为了实现这种解释，只需要做出一个假设：同一个种类的所有物体的surfaces对于一个category-specific surface $$S_0$$来说，都是isomorphic的，如fig2所示。

和derformable object的情况不一样，几何性质（也就是之前说的$$g$$）是不足以使得从$$S_0$$映射到不同object $$S$$的映射$$\pi_S$$能够相关联的（因为这个时候$$g$$已经无法简单的定义出来了），从而不同图片的keypoints之间的对应关系就不好建立了。然而，我们还希望这些这些$$\pi_S$$具有semantically consistent的性质，也就是还存在对应关系，比如说$$\pi_S(r)$$表示的是图片$$S$$里右眼的位置，而$$\pi_{S^{'}}(r)$$表示的也是图片$$S^{'}$$里右眼的位置。本文的一个重要的贡献就在于，如果将模型按照3.1里所说的那样进行训练，那么训练好的模型对于一个种类的不同图片，能够自动检测到semantically consistent的keypoints。


我们需要确定如何将从$$S_0$$映射到图片$$\Lambda$$的函数$$\Phi(\dot; \pmb x)$$表示为一个神经网络的输出。作者的方法是在$$K$$个离散的点采样来表示：$$\Phi(\pmb x) = (\Phi(r_1; \pmb x), \cdots, \Phi(r_K; \pmb x))$$。在这样的设定下，函数$$\Phi(\pmb x)$$就可以被理解为检测的是图片$$\pmb x$$的$$K$$个keypoints，$$p_k = \Phi(r_k; \pmb x)$$。作者并没有对这些keypoints加以别的约束。

如果$$\Phi$$是由一个神经网络所表示的，我们可以使用现有的任何一种用于keypoint detection的框架来实现它（比如HRnet，Hourglass等）。绝大多数这种框架都是预测的score maps $$\Psi(\pmb x) \in \mathbb R^{H \times W \times K}$$，对于每个keypoint $$r_k$$和图片的像素点位置$$u \in \lbrace 1, \cdots, H \rbrace \times \lbrace 1, \cdots, W \rbrace \subset \mathbb R^2$$，$$\Psi_{u r_k}(\pmb x)$$表示了一个得分。之后，可以对于每个通道，将每个score map利用softmax进行处理。从而得到了概率分布的score map。再之后，就可以利用求均值的方式求出每个keypoints的位置了。

主要的loss如下：

$$\mathcal L_{align} = \frac{1}{K} \sum_{r=1}^K \lVert \Phi(\pmb x \circ g) - g(\Phi(\pmb x)) \rVert^2 \tag{4}$$

上述的loss会使得网络学习到的keypoints都是consistent的，但并没有阻止网络所学习到的都是同一个点，也就是这些点全都聚集到了一起。

为了避免这个现象，作者加了一个diversity loss来使得这些probability maps对应到图片的不同区域。最直接的方式就是直接对两个probability相互重合的区域增加惩罚：

$$\mathcal L_{div}(\pmb x) = \frac{1}{K^2} \sum_{r=1} \sum_{r^{'}=1} \sum_u p(u \vert \pmb x, r) p(u \vert \pmb x, r^{'})$$

上述公式的一个缺点就是计算量是quadratic的。一个简便的方法是：

$$\mathcal L_{div}^{'} (\pmb x) = \sum_u (\sum_{r=1}^K p(u \vert \pmb x, r) - max_{r=1, \cdots, K} p(u \vert \pmb x, r))$$

对于一张图片$$\pmb x$$和用thin-plate splines（薄板样条插值）TPS实现的transformation $$g_1, g_2$$，网络的输入图片对为：$$g_1 \circ \pmb x$$和$$g_2 \circ (g_1 \circ \pmb x)$$。


### \[**CVPR 2018**\] [Unsupervised Discovery of Object Landmarks as Structural Representataions](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Unsupervised_Discovery_of_CVPR_2018_paper.pdf)

*Yuting Zhang, Yijie Guo, Yinxin Jin, Yijun Luo, Zhiyuan He, Honglak Lee*

[POST](https://www.ytzhang.net/projects/lmdis-rep/)


深度神经网络可以使用很丰富的latent representations来表示图片，但它们一般做不到直接表示图片里物体的结构信息。这篇文章通过无监督的方式来学习图片里物体的结构信息。作者提出了一个autoencoding的框架来将检测到的landmarks显式的表示为structural representations。encoding模块输出的是landmarks坐标，其加了限制从而能够输出有效的landmarks信息；而decoding模块以这些landmarks作为输入重构输入图片，从而形成一种end-to-end的结构来学习这些landmarks。这种方式所检测到的landmarks是semantically meaningful的，和人工标注的landmarks更为接近。文章所提出的方法所检测到的landmarks同时也是那些预训练模型所得到的图片features的补充。而且，文章所提出的方法自然的构建了一个无监督的界面，可以让用户通过控制这些landmarks从而改变物体的形状。

CV致力于理解那些能够反映物体物理特性并且和物体appearance无关的物体结构信息。这样的固有结构信息可以作为high-level的visual understanding的intermediate representations。然而，对于物体结构的人工标注或者设计（比如，skeleton，semantic parts）非常的耗时耗力，而且对于绝大多数的物体种类来说是不现实的，从而如何利用无监督的方式自动解决探索object structure的问题就成了十分吸引人的命题。现在的神经网络可以通过学习latent representations来高效的解决很多不同的视觉任务，包括image classification，segmentation，object detection，human pose estimation，3D reconstruction，image generation等等。一些已有的研究工作表名这些representations自然编码了很多视觉特征信息。然而，很少有证据表明深度神经网络可以自然表示出一类物体的固有的结构信息。

我们的目标在于使用非监督的方式学习物体的结构信息。作为一种表示物体结构的固有的一种表示形式，landmarks能够表示同一种类但是不同个体的物体稳定的局部语义信息。[Unsupervised learning of object landmarks by factorized spatial embeddings]()提出了一个无监督学习的方式在突破可能有transformations的情况下利用CNN来检测稳定的landmarks表示物体的局部语义信息。然而，这个方法并没有显式的鼓励这些landmarks出现在那些可以用作image modeling的位置上。


![usr1]({{ '/assets/images/USR-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. 用作无监督landmark discovery任务的autoencoding框架。*

作者将landmark localization问题描述为在图片里检测keypoints的任务。具体来说，每个landmark都有一个对应的detector，其通过卷积的方式输出一个detection score map（也就是一个heatmap），其中最大值的位置所对应的就是landmark的位置。在这个框架里，作者使用一个深度神经网络来将输入图片$$\pmb I$$转换为输出是$$K+1$$个通道的detection confidence map $$\pmb D \in \left[0,1\right]^{W \times H \times (K+1)}$$。这个confidence map会检测$$K$$个landmarks，而第$$K+1$$个通道表示的是背景。$$\pmb D$$的分辨率$$W \times H$$可以和输入图片的分辨率一样，或者更小，但是它们需要是同比例的。

作者提出一个轻量化的hourglass网络，来从输入图片中获取原始的detection score map $$\pmb R$$：

$$\pmb R = hourglass_l (\pmb I; \theta_l) \in \mathbb R^{W \times H \times (K+1)} \tag{1}$$

其中$$\theta_l$$表示的是网络参数。hourglass网络可以让detectors兼顾检测局部特征和利用全局信息。在获得了$$\pmb R$$之后，再在$$\pmb R$$的每个位置都做一次softmax，也就是每个位置沿着所有的通道数做一次softmax（包括表示背景的score map），从而所得到的$$\pmb D$$的每个位置，沿着所有通道的值加起来就是1，$$\pmb D$$里的每个值都在0到1之间：

$$\pmb D_k(u,v) = \frac{exp(\pmb R_k(u,v))}{\Sigma_{k^{'}=1}^{K+1} exp(\pmb R_{k^{'}}(u,v))} \tag{2}$$

其中$$\pmb D_k$$是$$\pmb D$$的第$$k$$个通道。

将$$\pmb D_k$$看作一个加权的map，作者使用加权的均值坐标作为第$$k$$个landmark的坐标，也就是：

$$(x_k, y_k) = \frac{1}{\zeta_k} \sum_{v=1}^H \sum_{u=1}^W (u,v) \cdot \pmb D_k(u,v) \tag{3}$$

其中$$\zeta_k = \sum_{v=1}^H \sum_{u=1}^W \cdot \pmb D_k(u,v)$$是空间归一化因子。这样的设置可以使得从decoder反向传播来的梯度经过landmark的坐标，除非$$\pmb D_k$$所有的权重都在一个点上或者均匀分布，这在概率上是不可能出现的。作者将landmark和landmark detector简记为：

$$\pmb l = \left[x_1, y_1, \cdots, x_K, y_K \right]^T = landmark(\pmb I; \theta_l) \tag{4}$$

fig 1上面蓝色部分的左半部分表示的就是landmark detector部分。


$$\pmb l$$里的元素应该是所检测到的landmark coordinates，但是到目前为止，并没有限制条件使得它们是landmarks，到现在为止它们只是任意的latent representations。因此，作者提出了下列soft constraints作为正则项来迫使这些所检测到的representations具有landmarks的特性。

* Concentration constraint

作为一个单一location的detection confidence map，$$\pmb D_k$$的值需要集中于一个局部的区域内。$$\pmb D_k / \zeta_k$$可以被认为是一个2维的概率分布，从而可以计算出其沿着$$x$$轴和$$y$$轴的方差，$$\sigma_{det, u}^2, \sigma_{det, v}^2$$。作者定义了如下的constraint loss来鼓励这两个方差都很小：

$$L_{conc} = 2\pi e (\sigma_{det, u}^2 + \sigma_{det, v}^2)^2 \tag{5}$$

上述的公式5表示的是以公式3计算出来的$$(x_k,y_k)$$为中心，以$$(\sigma_{det,u}^2 + \sigma_{det, v}^2)/2 \cdot \mathbb I$$为协方差矩阵的二维高斯分布的熵的exponential。这个高斯分布是$$\pmb D_k / \zeta_k$$的一个近似，而公式5越小，分布就越集中于中心点的位置，从而符合我们的要求。

这个近似的分布用公式表示则是：

$$\overline{ \pmb{ D_k}}(u,v) = (1 / WH) \mathcal N((u,v); (x_k, y_k), \sigma_{det} \mathcal I) \tag{6}$$

其之后还会被用到。

* Separation constraint

理想情况下，autoencoder的目标函数可以自动使得$$K$$个landmarks分布在不同的区域内从而很好地完成decoder里的reconstruction任务。然而，由于随机的初始化，由公式3定义的landmark coordinates的计算方法很可能导致landmarks互相靠得很近。这会导致优化过程陷入梯度下降算法无法逃脱的局部最优点。为了解决这个问题，作者提出了一个显式的loss来空间分隔这些landmarks：

$$L_{sep} = \sum_{k \neq k^{'}}^{1, \cdots, K} exp(\frac{-\lVert (x_{k^{'}}, y_{k^{'}}) - (x_k, y_k) \rVert_2^2}{2 \sigma_{sep}^2}) \tag{7}$$

* Equivariance constraint

一个特定的landmark应该位于一个微信的局部特征的地方（这个局部特征也应该由明确的语义信息）。这需要landmarks对于image transformations来说具有equivariance的特性。更具体来说，如果相对应的视觉信息仍然存在于transformed之后的图片中的话，一个landmark应该要跟着transformation变化而变化（比如说camera或者object的移动）。将coordiante transformation表示为$$g(\cdot, \cdot)$$，从而原image $$\pmb I$$和transformed之后的image $$\pmb I^{'}$$就有着如下的对应关系：$$\pmb I^{'}(u,v) = \pmb I(g(u,v))$$，并且$$\pmb I^{'}$$的landmarks $$\pmb l^{'}$$就可以用针对$$\pmb I$$所学习的landmark detector来获得：$$\pmb l^{'} = \left[ x_1^{'}, y_1^{'}, \cdots, x_K^{'}, y_K^{'} \right] = landmark(\pmb I^{'})$$，也就是说，$$g(x_k^{'}, y_k^{'}) = (x_k ,y_k)$$，从而引入如下的限制：

$$L_{eqv} = \sum_{k=1}^K \lVert g(x_k^{'}, y_k^{'}) - (x_k, y_k) \rVert_2^2 \tag{8}$$

如果$$g$$是已知的话，那么这个loss就是定义好了的。受到[Unsupervised learning of object landmarks by factorized spatial embeddings]()的启发，作者使用一个thin plate spline（TPS）来使用随机参数模拟$$g$$。作者使用随机的translation，rotation以及scaling来为TPS确定global affine component；之后再perturb一系列control points来确定TPS的local component。除了使用[Unsupervised learning of object landmarks by factorized spatial embeddings]()里所提到的传统的TPS control points，作者还使用3.1里所提到的detector检测到的keypoints作为control points和这些control points交替使用。而且，如果训练集合是以视频的形式出现的，可以将dense motion flow用作$$g$$，而下一帧就是$$\pmb I^{'}$$。

 
> 这篇文章提出的方法并没有显式的保证不同object instances上所检测到的landmarks的对应关系。landmarks的不同object instances之间的的semantic稳定性主要是依靠这样一个事实确保的：对于CNN来说，由同一个卷积核所获取的视觉特征一般来说都会有semantic相似性。


**Local latent descriptors**

对于简单的图片，比如说MINIST，landmarks就足够描述物体的形状了。但对于大多数实际的图片来说，landmarks并不足以表示所有的视觉内容，所以就需要额外的latent representations来encode补充信息。一方面，我们不能引入过多的全局信息，因为这样会导致模型不容易学习到landmark的局部信息（因为这样的全局信息就足够decoder完成reconstruction任务了）；而另一方面，我们也需要一定的全局信息来帮助landmarks的定位。为了解决这个trade-off，作者给每个landmark都计算了一个low-dimensional的local descriptor。

一个hourglass网络被用来获取一个feature map $$\pmb F$$，其和detection confidence map $$\pmb D$$的尺寸是一样的：

$$\pmb F = hourglass_f(\pmb I; \theta_f) \in \mathbb R^{W \times H \times S} \tag{9}$$

对于每个landmark来说，作者使用一个average pooling来获取这个landmark的local feature descriptor，其中这个average pooling的权重是由一个中心在这个landmark点的soft mask构成的。具体来说，我们将公式6定义的$$\overline{ \pmb{ D_k}}$$作为这个soft mask（公式6定义的这个分布实际上是detection confidence map的一个高斯分布的近似）。之后，一个可学习的linear operator（实际上就是几层MLP）被加在每个landmark用上述方式所获取的features上从而为每个landmark学习到一个独立的feature descriptor（注意到每个landmark都有一个自己的linear operator，不是共享的）。

> 作者认为之前所学习到的feature $$\pmb F$$使得所有的点的features都公用一个空间，而此处对于每个landmark都用一个linear projection将其映射到自己独立的空间里。

具体来说，第$$k$$个landmark的latent descriptor就是：

$$\pmb f_k = \pmb W_k \sum_{v=1}^H \sum_{u=1}^W (\overline{ \pmb{ D_k}}(u,v) \cdot \pmb F(u,v)) \in \mathbb R^{C} \tag{10}$$

其中$$C < S$$。每个landmark独有的这个linear operator使得每个landmark descriptor可以编码独有的信息。公式10也可以被用来获取一个背景的descriptor。使用之前所说的方法利用一个高斯分布来近似背景的confidence map是不合理的，所以直接令$$\overline{ \pmb{ D_{K+1}}} = \pmb D_{K+1} / \zeta_{K+1}$$。注意到$$\pmb f_k$$对于feature map $$\pmb F$$和detection confidence map $$\pmb D_k$$来说都是可微的。

将所有的landmarks的descriptors和背景的descriptor放在一起，我们就有了$$\pmb f = \left[ \pmb f_1, \pmb f_2, \cdots, \pmb f_{K+1} \right] \in \mathbb R^{C \times (K+1)}$$。fig1的下半部分的左边部分表示了获取这个landmark descriptors $$\pmb f$$的过程。


**Landmark-based decoder**

作者先从所检测到的landmark coordinates恢复detection confidence map $$\tilde{ \pmb{ D}} \in \mathbb R^{W \times H \times (K+1)}$$。具体来说，作者使用一个二维高斯分布来表示每个landmark的confidence map，其中心点是每个landmark的coordinates，其协方差矩阵是$$\sigma_{det}^2 \mathbb I$$，其中$$\sigma_{dec}$$不是之前计算出来的，是一个超参数。

$$\tilde{ \pmb{R_k}}(u,v) = \mathcal N((u,v); (x_k, y_k), \sigma_{dec}^2 \mathbb I), k=1,2,\cdots,K, \tilde{ \pmb{ R_{K+1}}} = \pmb 1 \tag{11}$$

然后再从$$\tilde{ \pmb{ R}}$$中，对于每个位置，沿着所有的通道做一次归一化，从而恢复detection score map $$\tilde{ \pmb{ D}}$$：

$$\tilde{ \pmb{ D_k}}(u,v) = \tilde{ \pmb{ R_k}}(u,v) / \sum_{k=1}^{K+1} \tilde{ \pmb{ R_k}}(u,v) \tag{12}$$

fig1的上半部分的右半部分描述了这个过程。

对于每个landmark的descriptor $$f_k$$（也包括背景的），作者对于每个landmark来说都利用一个landmark-specific linear operator $$\tilde W_k$$将这个descriptor再次进行线性变化，之后加上一个activation function（文中使用的是LeakyReLU），试图将其再次转换到所有feature公用的空间里（也就是之前$$\pmb F$$所代表的空间）。使用$$\tilde D$$作为global unpooling的权重，feature map的恢复过程如下：

$$\tilde {\pmb {F}}(u,v) = \sum_{k=1}^{K+1} \tilde{ \pmb{ D_k}}(u,v) \cdot \tau(\tilde{ \pmb{ W_k}} \pmb f_k) \in \mathbb R^{W \times H \times S} \tag{13}$$

其中$$\tau$$就是activation function。这个过程由fig1下半部分的右半部分所描述。

本文里的方法使得反向传播算法可以通过landmark coordinates进行传播。每个landmark对应的高斯分布的方差$$\sigma_{dec}^2$$决定其周围的邻居可以给这个landmark贡献多少信息。在训练一开始的时候，需要比较大的$$\sigma_{dec}^2$$来使得训练可行，也就是说每个landmark要依靠的邻居点还很多，而随着训练的进行，需要更准确的定位，也就是需要比较小的方差。对于这样两个相互矛盾的需求，作者同时使用了拥有不同$$\sigma_{dec}$$值来获取多个版本的$$\tilde{ \pmb{ D}}$$和$$\tilde{ \pmb{ F}}$$：$$(\tilde{ \pmb{ D_1}}, \tilde{ \pmb{ F_1}}), (\tilde{ \pmb{ D_2}}, \tilde{ \pmb{ F_2}}), \cdots, (\tilde{ \pmb{ D_M}}, \tilde{ \pmb{ D_M}})$$。

最后，将上述这些$$(\tilde{ \pmb{ D_1}}, \tilde{ \pmb{ F_1}}), (\tilde{ \pmb{ D_2}}, \tilde{ \pmb{ F_2}}), \cdots, (\tilde{ \pmb{ D_M}}, \tilde{ \pmb{ D_M}})$$沿着通道这个维度连起来，再输入一个hourglass网络里，最终获取重建的图片：

$$\tilde{ \pmb{ I}} = hourglass_d (\left[\tilde{ \pmb{ D_1}}, \tilde{ \pmb{ F_1}}), (\tilde{ \pmb{ D_2}}, \tilde{ \pmb{ F_2}}), \cdots, (\tilde{ \pmb{ D_M}}, \tilde{ \pmb{ D_M}} \right]; \theta_d) \tag{14}$$

fig1里的最右边的灰色线表示了这个过程。


**Overall training objective**

图片reconstruction loss驱动了整个training的进行，定义这个loss为$$L_{recon} = \lVert \pmb I - \tilde{ \pmb{ I}} \rVert_F^2$$，而且输入的图片$$\pmb I$$是被归一化到0到1之间的。从而，整个网络的loss，$$L_{AE}$$定义为：

$$L_{AE} = \lambda_{recon} L_{recon} + \lambda_{conc} L_{conc} + \lambda_{sep} L_{sep} + \lambda_{eqv} L_{eqv} \tag{15}$$



### \[**NeurIPS 2020** $$\&$$ **TPAMI 2021**\] [Unsupervised Learning of Object Landmarks via Self-Training Correspondence](https://proceedings.neurips.cc/paper/2020/hash/32508f53f24c46f685870a075eaaa29c-Abstract.html)

[CODE-V1](https://github.com/malldimi1/UnsupervisedLandmarks)
[CODE-V2](https://github.com/malldimi1/KeypointsToLandmarks)


这篇文章对于无监督学习物体关键点提出了一种新的范式。和已有的利用image generation或别的代理方法来进行无监督学习不同，本文提出的是一种自学习的方式，从general的keypoints candidates集合出发，来逐渐筛选并学习keypoints的detector和descriptor，逐渐将keypoints调整到更精确的位置上。为了达到这个目标，作者提出了一个iterative的算法，其在（1）利用feature clustering来生成假标签；和（2）对于每个假标签，利用对比学习来学习它的features，这两个任务之间反复横跳着更新。本文中所用的detector和descriptor共享一个backbone，这样做的好处是可以使得keypoint的位置能够逐步收敛到稳定的点，将那些不稳定的点filter掉。和先前的工作相比，作者提出的方法可以在物体有很大角度变化的情况下仍然学习到对应的keypoints。


现有的无监督学习方法经常依赖于某些代理任务，而学习landmarks这样的一个主要任务通常是通过某种latent过程来实现的（比如说[Discovery of Latent 3D Keypoints via End-to-end Geometric Reasoning]）。还有一些方法使用某些代理任务，比如说equivariance：[Unsupervised learning of landmarks by descriptor vector exchange]，[Unsupervised learning of object frames by dense equivalent image labelling]，[Unsupervised learning of object landmarks by factorized spatial embeddings]，或者image generation：[Unsupervised learning of object landmarks through conditional image generation]，[Object landmark discovery through unsupervised adaption]，[Unsupervised discovery of object landmarks as structural representataions]。基于equivariance原则的方法表明，一个detector必须在经过已知的某种自定义的image deformation下保持连续（也就是说一张图片在经过image deformation前后，这个detector所检测到的keypoints要保证correspondence，其中这个image deformation是已知的），从而他们就利用这个原则来学习模型。而基于image generation的方法的流程是使用一个generator来reconstruct原输入图片，这个generator的输入是原图片经过了deformation之后的图片，而且这个generator是依赖于某个detector的输出的。detector和generator通过一个bottleneck来交换信息可以从输入里蒸馏出物体的几何信息，因为一个generator如果想从根据一张图片经过deformation之后的图片作为输入还能够reconstruct原输入照片，那它就需要对图片里物体的几何特征以及landmarks有足够的理解（也就是说，输入图片经过了某种deformation之后传给了generator，然后希望generator能够恢复原输入，这个时候generator所拿到的图片里的object的几何特征变了，但纹理特征没有，也就是说它需要找到deformation前后landmarks的对应关系来学习到这个deformation对于几何特征造成了什么样的影响， 才能够reconstruct原输入）。 

尽管这些方法在某些场景下取得了好的效果，但这些场景下物体一般都只有比较小的角度变化（比如说正面的人脸，人的身体，猫脸等），这些方法在以下两个角度上具有局限性。首先，代理任务并不能确保detector就能够显式的学习到物体的landmarks，因此会产生那些不太会被人类标注员所标注的landmarks。其次，这些方法都需要人为生成deformations，因为只有这样才能够产生一对图片，其landmarks之间的对应关系是知道的（deformation前后）。但是通过这样一对图片来学习（一张图片以及经过了deformation的这张图片）会导致模型对于更复杂的类内variation不具有鲁棒性，比如说背景变化、角度变化（经过了3D rotation）、或者说那些articulated的物体（比如说人的身体）。

在这篇文章里，作者观察到，尽管landmark detectors在无监督的方式下很难被训练，generic keypoint detectors却很容易获取。基于这个事实，作者提出了一个新的方法来从generic keypoint detector来获取landmark detector。generic keypoints，也可以叫做salient或者interest points，就是一张图片里那些“发生了某种事情”的地方（也就是说这个地方在appearance上发生了变化）以及边缘所在的地方。本文使用pixel-based的descriptors来表示keypoints，这样做的好处是可以更方便的找到两张图片里相对应的keypoints。generic keypoints可以直接用Sobel filters（比如SIFT）或者预训练好的模型直接得到（比如SuperPoint）。

基于keypoints和landmarks之间的相似性和不同，这篇文章的目的是将一系列预先给定的generic keypoints变换为语意连贯的能够表示object parts的landmarks，在训练的过程中对keypoints进行filter以及refine。

作者所提出的方法的出发点是一个具有generic keypoints标注的某类物体的数据集，而且这些keypoints里包括那些能够表示物体part的语义信息的landmarks，而方法主要要做的就是filter以及refine这些keypoints。从这出发，作者的目标是提出一个自训练的方法来使用一种完全无监督的方式学习一个landmark detector。这篇文章所提出的方法和SuperPoint类似，都是具有一个detector head、一个descriptor head和共用的backbone，通过自训练的方式来iteratively定位keypoints而且给这些keypoints学习具有correspondence的descriptors。作者的目标是将一个keypoint detector转变为一个landmark detector，这些landmark具有语义信息。为了达成这个目标，作者的方法在（1）使用descriptor clustering进行keypoints伪标签以及keypoints correspondence的学习（2）以及使用伪标签进行自学习来训练模型，这两个任务之间alternately反复横跳。

作者发现，和先前的方法相比，本文的方法能够学习到那些具有很大的3D角度变换的landmarks，这一点从fig1就能看出来。

![unsuper1]({{ '/assets/images/UNSUPER-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. 和之前那些方法都不同，本文提出的方法能够解决物体在进行很大角度变换之后的landmarks的对应关系，而且能解决对称的问题（也就是对称的物体的两个对称的landmark也是不同的）。*


$$\mathcal{X} = \lbrace x \in \mathbb{R}^{W \times H \times 3} \rbrace$$是某个物体种类的$$N$$张图片的集合（比如说人脸、身体等）。在$$\mathcal{X}$$上运行一个generic keypoint detector（可以是SIFT，或者别的预训练好的model比如说SuperPoint，等），现在训练集合$$\mathcal{X}$$就变成了$$\lbrace x_j, \lbrace p_i^j \rbrace_{i=1}^{N_k} \rbrace$$，其中$$p_i^j \in \mathbb{R}^2$$是一个keypoint二维坐标，$$N_j$$是图片$$x_j$$里所检测到的keypoints的数量。这些$$p_i$$是没有任何顺序的，而且不同图片之间的$$p_i$$也是没有correspondence的。而且，有一些keypoints可能是outliers，这是由于背景导致的。我们的最终目标是要学习一个heatmap $$\mathcal{Y} \in \mathbb{R}^{H_o \times W_o \times K}$$，这个$$H_o, W_o$$输入尺寸等比例缩小（也可能不变），$$K$$是landmarks的个数，每个heatmap表示一个landmark的分布概率。从输出的结构来看，landmarks是有顺序的，也就是说两张图的landmarks是有correspondence的。

作者将解决问题的办法分为了两个stage。在第一个stage里，作者将会训练一个网络$$\Phi$$，其会输出一系列的keypoints，而且每个keypoint也有其对应的descriptors（实际上descriptors的输出是pixel-based，也就是每个pixel都有对应的descriptors，不仅仅是keypoints对应的那些pixel），而且不同图片的keypoints之间的correspondence也是输出的一部分，这个过程可以补全某些图片里缺失的keypoints，也可以过滤掉某些图片里背景的keypoints。在第二个stage，我们就可以用第一个stage的输出来和之前有监督的方式一样训练网络了。

![unsuper2]({{ '/assets/images/UNSUPER-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. 本文所提出方法的stage1的流程。所用到的网络是具有共用的backbone的两个输出头的网络，一个头是detector head，另一个是descriptor head。在训练过程中，算法将会在利用descriptor的features进行clustering从而给keypoints打上伪标签，与利用这种伪标签进行自学习，这两个过程之间反复横跳。和之前的那些方法都不同，本文所使用的方法并不需要一张图片以及它的deformation作为输入图片对，这种输入会减弱模型学习类内variation的能力。correspondence recovery是利用文中提出的修改版K-means实现的。本文的方法还可以用来恢复丢失的landmarks。上图还显示了使用t-SNE来可视化数据集里的keypoint对应位置的features的结果，看起来不错。*


文中方法的第一个stage是学习一个keypoint detector $$\Phi$$，这个detector的网络结构和之前的工作的结构是差不多的：有一个共享的backbone $$\Phi_b: \mathcal{X} \rightarrow \mathcal{F}$$，来从输入图片学习intermediate features $$\mathcal{F}$$，之后再接上两个heads，一个用来检测object landmarks，叫$$\Phi_d$$，一个用来输出landmarks的descriptors（也就是features，而且其实它输出的是pixel-based也就是dense的descriptors，但也只有landmarks位置的features被用到了），叫做$$Phi_f$$。

detector head $$\Phi_d$$，对于输入图片$$x_j$$，会输出一个单通道的spatial confidence map $$H_j = \Phi_d(\Phi_b(x_j)) \in \mathbb{R}^{H_o \times W_o \times 1}$$，其表示的是在任意一个像素点位置出现landmark的概率大小，这些landmarks没有顺序，从而在不同图片之间也没有correspondence。之后，再利用非极大抑制从$$H_j$$里获取landmark的二维坐标$$p_i^j$$。$$\Phi_d$$的主要作用是找到这张图片里物体因为遮挡、旋转、自遮挡等造成一开始在generic keypoints里遗失的landmarks，以及过滤掉背景里的keypoints。而且其给出了landmarks的二维坐标这样一个伪标签。

feature extractor head $$\Phi_f$$会对于$$x_j$$产生一个dense的feature map $$F_j = \Phi_f(\Phi_b(x_j)) \in \mathbb{R}^{H_o \times W_o \times d}$$，而这个descriptors将会被用来找到不同图片产生的keypoints之间的correspondence。在detector head所找到的每个landmark $$p_i^j$$的位置，我们都有这个landmark的descriptor $$f_i^j$$（从$$F_j$$的对应位置找到就行）。


* Correspondence recovery

在训练集$$\mathcal{X}$$上作用了3.2里所说的$$\Phi$$之后，我们将会得到$$\lbrace x_j, \lbrace p_i^j, f_i^j \rbrace_{i=1}^N_j \rbrace$$。再之后，我们利用每个keypoint的feature来给它们贴上一个假标签。这个过程叫做correspondence recovery，因为其可以让我们找到不同图片里的keypoints之间的对应关系。注意，这个过程是offline的，也就是说是没有可训练参数的，而且这个对feature进行clustering从而对每张图片里的每个检测到的keypoint贴上标签（也就是获取correspondence）的过程是针对整个数据集的。correspondence recovery是利用K-means对于所有的图片的所有的keypoints位置对应的features构成的集合$$f$$上进行的。需要注意的是，我们不能将同一张图片里的两个不同的keypoints贴上同一个标签，也就是assign给同一个cluster。

> 本文中所使用的correspondence recovery，也就是贴标签的过程，是由K-means来实现的，也就是给每张图片的每个keypoint一个cluster标签，贴上同一个cluster标签的不同图片的keypoints就互相correspond。

上述的clustering过程定义如下：

$$\min\limits_{C \in \mathbb{R}^{d \times M}} \frac{1}{N} \sum\limits_{j=1}^N \sum\limits_{i=1}^{N_j} \min\limits_{y_i^j \in \lbrace 0,1 \rbrace^M} \lVert f_i^j - Cy_i^j \rVert_2^2$$

$$s.t. 1_m^T y_i^j = 1$$

$$\lVert \sum\limits_{i} y_i^j \rVert_0 = N_j$$

其中，$$M$$是clusters的数量，$$y_i^j$$是给每个landmark $$p_i^j$$所贴的cluster的标签（也就是归属于哪个cluster），$$y_i^j$$是一个one-hot向量，长度为M，也就是说只有在cluster类别的位置值是1，其余都是0，有且仅有一个1，上述条件里的第一个条件就是这个约束。而第二个条件实际上包括在第一个条件以内，这里不太明白。$$C \in \mathbb{R}^{d \times M}$$是一个centroid矩阵，表明$$M$$个cluster中心的features。作者在这里采用了简化版的Hungarian algorithm来解决上述优化问题（在NeurIPS那个版本里就直接使用了Hungarian algorithm）。对于一个给定的图片$$x_j$$，对于每个cluster的中心的feature，找到这张图片里的keypoints们的features里最靠近这个cluster中心feature的那个keypoint，这个keypoint就被贴上这个cluster的标签。因为选取了最靠近的点，所以这个点是唯一的（如果有多个极值点，就随便选一个），从而就可以使得每张图片里不会出现两个keypoints贴的cluster标签是同一个的问题。而且对于每个cluster，每张图片只选择一个keypoint能够贴上这个cluster的标签也可以过滤掉一些噪声keypoints，因为每张图片里最能代表这个cluster的keypoint已经被找到了，那么其它的就可能是噪声点了。

作者就使用了上述修改版的K-means方法来对于所有图片里检测到的keypoints进行correspondence recovery的操作。作者表示，需要将$$M$$设置为远大于$$K$$的值：这样会导致feature space的过分割问题，但这个却可以使得每个landmark可以对应好几个clusters，这样带来的好处就是它可以解决图片中物体视角变化非常大的情况，因为由于视角变化，实际上这两张图片里的同样区域的clusters的features不太一样，但它们都被贴在了同一个landmark上，可以由后续操作将它们合并。这一点也是本文的方法和之前的工作的一个不同，本文的方法可以解决角度变化很大的情况。

除了上述所说的对K-means的一个修改，作者还约束了每张图片最多只能检测到$$K$$个landmarks。和NeurIPS那个版本不同，那个版本里每张图片到底能检测到多少landmarks是不确定的，因为后续还会合并相似的clusters。而本文则约束每张图片最多检测$$K$$个landmarks，和其它的无监督方法学习keypoints的论文保持一致。为了实现这个约束，作者在$$\Phi_d$$得到了那个单通道的spatial confidence map之后，加上约束最多检测$$K$$个keypoints。为了实现上述目标，修改后的K-means算法需要被执行两次：第一次，设定clusters的数量为$$K$$个，然后对于每个cluster，给feature最靠近这个cluster的feature的keypoint贴上这个cluster的标签，这样的操作就会导致每张图片最多只有$$K$$个keypoint被贴了标签，也就是最多只有$$K$$个keypoint被保留了下来（如果利用超参数从$$\Phi_d$$的输出里找到的keypoints的数量大于$$K$$，那么经过这个操作之后就剩下$$K$$个，如果少于$$K$$，那么就剩下这个数量）。第二次，设定clusters的数量为$$M$$个，其中$$M$$要比$$K$$大，然后再对于每个cluster，选取每张图片里feature最靠近这个cluster的feature的keypoint，给它贴上这个cluster的标签，这样的话，由于$$M$$的数量远大于$$K$$，所以就有keypoints贴了不止一个clusters的标签，就如上一段所说的那样了。作者还说，即使是进行了两次K-means，速度还是很快的。fig2里的t-SNE展示了经过clustering之后的features（但也没啥用，t-SNE本来就不太靠谱）。


* Training losses

在进行了correspondence recovery之后，训练集从之前的仅仅包含二维图片$$\mathcal{X}$$，变成了现在又多了两个集合：表示keypoints位置的$$p_i^j$$，这个是从$$\Phi_d$$得来的；以及表示keypoints对应关系的$$y_i^j$$，这个是从correspondence recovery得来的。下一步就是利用这些假标签，来对backbone，$$\Phi_b$$，以及两个heads，$$\Phi_d$$和$$\Phi_f$$进行训练更新。然后在训练完了之后，我们就又可以利用参数更新之后的网络，来得到新的spatial confidence map，以及新的descriptors，从而就又可以计算新的correspondence recovery，从而又可以再进行这一步，从而开始循环反复横跳的训练，也就是所说的self-training iteratively。

这一步训练detector head所用到的loss是标准的MSE loss：

$$\mathcal{L_d}(x_j) = \lVert H(x_j) - \Phi_d(\Phi_b(x_j)) \rVert^2$$

其中$$H$$表示由$$p_i^j$$所计算出来的ground truth 2D heatmap，其是一个中心为$$p_i^j$$的高斯分布，其中$$i=1,2,\cdots, N_j$$。

对于feature extractor head $$\Phi_f$$的训练，作者使用了contrastive loss来训练它们。对于某个训练循环时刻$$t$$，这个时候的训练集是$$\mathcal{X_t} = \lbrace x_j, \lbrace p_i^j, y_i^j \rbrace_{i=1}^{N_j} \rbrace$$，我们的目标是更新$$\Phi_f$$，使得其输出的feature，对于两个keypoints，$$p_i^j$$和$$p_{i^{'}}^{j^{'}}$$，其对应位置的features仅仅在$$y_i^j=y_{i^{'}}^{j^{'}}$$的情况下相等（尽可能的相近）。为了实现这个目标，作者使用了contrastive loss，其目标就是将对应于同一个cluster的features尽可能的拉近，将对应于不同clusters的features尽可能的推远。对于一个给定的图片对$$x_j$$和$$x_{j^{'}}$$，以及这两张图片上分别检测到的keypoints，$$i$$和$$i^{'}$$，contrastive loss定义如下：

$$\mathcal{L_f}(x_i^j, x_{i^{'}}^{j^{'}}) = 1_{\left[ y_i^j = y_{i^{'}}^{j^{'}} \right]} \lVert f_i^j - f_{i^{'}}^{j^{'}} \rVert^2 + 1_{\left[ y_i^j \neq y_{i^{'}}^{j^{'}} \right]} max(0, m -\lVert f_i^j - f_{i^{'}}^{j^{'}} \rVert^2)$$

正如很多使用了constrastive loss的无监督学习方法一样，如何选择positive pairs和negative pairs是使用contrastive loss的关键。positive pairs可以由以下两种方式获得：两张不同的图片里对应着同一个cluster的两个keypoints；一张图片以及它的deformation。而negative pairs的选取方法就很多了。在NeurIPS的那个版本里，不同图片里具有不同cluster标签的keypoints就可以构成一个negative pair。但在这篇文章里，作者进行了改进：仅仅在同一张图片里选取negative pairs。因为有过分割的现象，所以同一个landmark在不同的图片里也可能对应着不同的clusters，如果将它两选为negative pair，就会出问题。而且，正如上文所说，每个cluster在每张图片里仅仅能够出现一次，所以任意选择两个同一张图片里的keypoints，就一定对应着不同的clusters，也就一定是一个negative pair。而且更进一步的，对于同一张图片，只要选取的位置$$i$$距离$$i^{'}$$足够的远，即使这个点根本不对应任何keypoint，它两的features也构成一个negative pair。fig3阐述了本文和NeurIPS那篇文章里选取negative pair的不同思路。

![unsuper3]({{ '/assets/images/UNSUPER-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 3. 在NeurIPS那篇文章里，negative pairs是不同图片里具有不同cluster标签的keypoints对。因为过分割，所以多个clusters可以对应同一个landmark，所以这样的选取可能会导致问题，如图中红线所示。而本文中从同一张图片里选取negative pair的方式，因为clustering的过程而确保了其一定不会对应相同的cluster。*

将$$\Phi_b, \Phi_d$$和$$\Phi_f$$的参数分别表示为$$\theta_b, \theta_d$$和$$\theta_f$$，那么stage1的训练过程可以总结为下属算法Algorithm1：

![unsuper4]({{ '/assets/images/UNSUPER-4.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}


* Bootstrapping

最一开始，在$$t=0$$的时刻，训练集$$X_0$$仅仅包含$$\lbrace x_j, \lbrace p_i^j \rbrace_{i=1}^{N_j} \rbrace$$，而并没有keypoint之间的correspondence $$f_i^j$$。在NeurIPS那篇文章里，最初始的features是由某个预训练好的网络给的（SuperPoint），然后就可以进行后续的循环了。但在这篇文章里，作者采用了一个warm up的预训练stage，也就是只利用一张图片和它的deformation构成的图片对作为输入来训练网络。这样的话keypoints之间的对应关系就已经知道了，这就是positive pairs。也就是利用equivariance性质来初始化backbone和feature extractor head。


* Learning an object landmark detector

在Stage1完成之后，我们所得到的结果是一个数据集$$\mathcal{X}$$，包含一系列keypoints，以及keypoints对应的含有landmark信息的descriptors。但我们的终极目标是，输出固定数量（$$K$$个）landmark坐标。

在stage1结束之后，keypoints被贴上了cluster的标签，而cluster的数量$$M$$要远大于$$K$$，所以这个时候在$$K$$个类上训练一个landmark detector也不是trivial的，因为我们并不知道到底哪些clusters需要对应同一个landmark。在NeurIPS那个版本里，这个过程是通过逐渐合并clusters来减少clusters数量到$$K$$个来实现的。然而，因为现在每张图片的keypoints数量最多是$$K$$个，也因为采用了更好的negative pair的选取策略，作者发现即使有$$M$$个clusters，这些clusters也自动形成了$$K$$个分离的很好的clusters（fig8展示了$$K=30$$时候的情况）。这个事实就使得我们并不再需要在NeurIPS里所使用的逐渐合并clusters的这样一个流程。

为了最终将训练集变为只有$$K$$个clusters，作者再次使用了具有$$K$$个聚类中心的K-means。再之后，因为


### \[**NeurIPS 2018**\] [Unsupervised Learning of Object Landmarks through Conditional Image Generation](https://proceedings.neurips.cc/paper/2018/hash/1f36c15d6a3d18d52e8d493bc8187cb9-Abstract.html)

  
### \[**NeurIPS 2019**\] [Unsupervised Learning of Object Keypoints for Perception and Control](https://proceedings.neurips.cc/paper/2019/hash/dae3312c4c6c7000a37ecfb7b0aeb0e4-Abstract.html)


### \[**NeurIPS 2019**\] [Object landmark discovery through unsupervised adaptation](https://proceedings.neurips.cc/paper_files/paper/2019/file/97c99dd2a042908aabc0bafc64ddc028-Paper.pdf)


### \[**WACV 2022**\] [LEAD: Self-Supervised Landmark Estimation by Aligning Distributions of Feature Similarity](https://openaccess.thecvf.com/content/WACV2022/html/Karmali_LEAD_Self-Supervised_Landmark_Estimation_by_Aligning_Distributions_of_Feature_Similarity_WACV_2022_paper.html)


### \[**CVPR 2018**\] [Unsupervised discovery of object landmarks as structural representation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Unsupervised_Discovery_of_CVPR_2018_paper.pdf)

[POST](https://www.ytzhang.net/projects/lmdis-rep/)


### \[**NeurIPS 2018**\] [Unsupervised Learning of Object Landmarks through Conditional Image Generation](https://proceedings.neurips.cc/paper/2018/hash/1f36c15d6a3d18d52e8d493bc8187cb9-Abstract.html)


### \[**NeurIPS 2019**\] [Unsupervised Learning of Object Keypoints for Perception and Control](https://proceedings.neurips.cc/paper/2019/hash/dae3312c4c6c7000a37ecfb7b0aeb0e4-Abstract.html)


### \[**ICCV 2019**\] [Unsupervised Learning of Landmarks by Descriptor Vector Exchange](https://openaccess.thecvf.com/content_ICCV_2019/html/Thewlis_Unsupervised_Learning_of_Landmarks_by_Descriptor_Vector_Exchange_ICCV_2019_paper.html)


### \[**CVPR 2020**\] [Self-Supervised Learning of Interpretable Keypoints From Unlabelled Videos](https://openaccess.thecvf.com/content_CVPR_2020/html/Jakab_Self-Supervised_Learning_of_Interpretable_Keypoints_From_Unlabelled_Videos_CVPR_2020_paper.html)


### \[**NeurIPS 2019**\] [Unsupervised Keypoint Learning for Guiding Class-Conditional Video Prediction](https://proceedings.neurips.cc/paper/2019/hash/801272ee79cfde7fa5960571fee36b9b-Abstract.html)


### \[**CVPR 2020**\] [Unsupervised Learning of Intrinsic Structural Representation Points](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Unsupervised_Learning_of_Intrinsic_Structural_Representation_Points_CVPR_2020_paper.html)


### \[**CVPR 2021**\] [Unsupervised Human Pose Estimation Through Transforming Shape Templates](https://openaccess.thecvf.com/content/CVPR2021/html/Schmidtke_Unsupervised_Human_Pose_Estimation_Through_Transforming_Shape_Templates_CVPR_2021_paper.html)

[POST](https://infantmotion.github.io)


### \[**NeurIPS 2019**\] [Object landmark discovery through unsupervised adaptation](https://proceedings.neurips.cc/paper/2019/hash/97c99dd2a042908aabc0bafc64ddc028-Abstract.html)


### \[**ICLR 2021**\] [Unsupervised Object Keypoint Learning using Local Spatial Predictability](https://openreview.net/forum?id=GJwMHetHc73)


### \[**ICLR 2021**\] [Semi-supervised Keypoint Localization](https://openreview.net/forum?id=yFJ67zTeI2)


### \[**CVPR 2022**\] [Self-Supervised Keypoint Discovery in Behavioral Videos](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_Self-Supervised_Keypoint_Discovery_in_Behavioral_Videos_CVPR_2022_paper.pdf)


### \[**Arxiv 2021**\] [Weakly Supervised Keypoint Discovery](https://arxiv.org/pdf/2109.13423.pdf)


### \[**Arxiv 2021**\] [Pretrained equivariant features improve unsupervised landmark discovery](https://arxiv.org/pdf/2104.02925.pdf)


### \[**Arxiv 2019**\] [Learning Landmarks from Unaligned Data using Image Translation](https://openreview.net/pdf?id=xz3XULBWFE)


### \[**CVPR 2021 Oral**\] [Self-supervised Learning of Interpretable Keypoints from Unlabelled Videos](https://www.robots.ox.ac.uk/~vgg/research/unsupervised_pose/data/unsupervised_pose.pdf)

[POST](https://www.robots.ox.ac.uk/~vgg/research/unsupervised_pose/)


### \[**WACV 2022**\] [LEAD: Self-Supervised Landmark Estimation by Aligning Distributions of Feature Similarity](https://openaccess.thecvf.com/content/WACV2022/html/Karmali_LEAD_Self-Supervised_Landmark_Estimation_by_Aligning_Distributions_of_Feature_Similarity_WACV_2022_paper.html)


### \[**Arxiv 2020**\] [Unsupervised Learning of Facial Landmarks based on Inter-Intra Subject Consistencies](https://arxiv.org/pdf/2004.07936.pdf)


### \[**Arxiv 2020**\] [Unsupervised Landmark Learning from Unpaired Data](https://arxiv.org/pdf/2007.01053.pdf)

[CODE](https://github.com/justimyhxu/ULTRA)


### \[**Arxiv 2021**\] [LatentKeypointGAN: Controlling GANs via Latent Keypoints](https://xingzhehe.github.io/LatentKeypointGAN/)


### \[**NeurIPS 2021**\] [Unsupervised Part Discovery from Contrastive Reconstruction](https://proceedings.neurips.cc/paper/2021/hash/ec8ce6abb3e952a85b8551ba726a1227-Abstract.html)


### \[**Arxiv 2019**\] [Video Interpolation and Prediction with Unsupervised Landmarks](https://arxiv.org/pdf/1909.02749.pdf)


### \[**CVPR 2022**\] [Few-shot Keypoint Detection with Uncertainty Learning for Unseen Species](https://openaccess.thecvf.com/content/CVPR2022/papers/Lu_Few-Shot_Keypoint_Detection_With_Uncertainty_Learning_for_Unseen_Species_CVPR_2022_paper.pdf)


### \[**Arxiv 2021**\] [TACK: Few-Shot Keypoint Detection as Task Adaptation via Latent Embeddings](https://sites.google.com/view/2021-tack)


### \[**NeurIPS 2022 Spotlight**\] [AutoLink: Self-supervised Learning of Human Skeletons and Object Outlines by Linking Keypoints](https://arxiv.org/pdf/2205.10636.pdf)

[CODE](https://github.com/xingzhehe/AutoLink-Self-supervised-Learning-of-Human-Skeletons-and-Object-Outlines-by-Linking-Keypoints)

这篇文章要解决的问题是无监督的从一类物体的图片里学习到2D keypoints。模型的输入是一类物体（比如人脸）的RGB图片，输出是2D keypoints，数量和顺序是固定的。也就是说，模型在输入图片后，输出$$K$$个heatmap，然后从$$K$$个heatmap里获取$$K$$个2D keypoints。所以不同图片的2D keypoints之间的correspondence也是直接就有的。

很多无监督2D keypoints的方法都是一个auto encoder的结构，但这篇文章的创新点有如下几个：

* 首先，网络除了输出$$K$$个heatmap之外，还会输出一个$$K \times K$$的graph，用来表示没两对keypoints之间的权重，而且这个graph是global的，也就说对于每个输入图片，其是共同的。
* 其次，对于每对输出的keypoints，构建一个edge heatmap，大小和输入图片一样，是以这两个keypoint的连线为中心的Gaussian分布，再利用上面的graph，得到每个像素点位置的global edge heatmap的值
* 在得到这个global edge heatmap之后，还是一样需要一个decoder来reconstruct原输入图片，显然只有这个骨架是没办法还原的，所以文章的做法是对于输入图片，mask掉其绝大部分区域，和这个edge heatmap沿着channel连起来，作为decoder的输入。

这篇文章的方法很简单，却很有效果。

有个要注意的技术细节是，reconstruction loss不仅仅是reconstructed的图片和原图片之间的mse loss，而是perception loss，也就是将这两个图片都输入某个预训练好的网络，比如在ImageNet上预训练好的VGG19，然后对比很多层的输出之间的差异的和。这样做要比仅仅在像素层面比较区别更加鲁棒。


### \[**CVPR 2024 Highlight**\] [Unsupervised Keypoints from Pretrained Diffusion Models]([https://ubc-vision.github.io/StableKeypoints/](https://openaccess.thecvf.com/content/CVPR2024/papers/Hedlin_Unsupervised_Keypoints_from_Pretrained_Diffusion_Models_CVPR_2024_paper.pdf))

*Eric Hedlin, Gopal Sharma, Shweta Mahajan, Xingzhe He, Hossam Isack, Abhishek Kar, Helge Rhodin, Andrea Tagliasacchi, Kwang Moo Yi*

[CODE](Unsupervised Keypoints from Pretrained Diffusion Models)



## 3D keypoints from images (Unsupervised)

### [Unsupervised Learning of Visual 3D Keypoints for Control](http://proceedings.mlr.press/v139/chen21b/chen21b.pdf)

*Boyuan Chen, Pieter Abbeel, Deepak Pathak*

[Code](https://github.com/buoyancy99/unsup-3d-keypoints) [Post](https://buoyancy99.github.io/unsup-3d-keypoints/)

这篇文章使用了一种非监督的方式，设计了一个end-to-end的模型，直接从2D图片里学习到3D keypoints。其是通过一个multi-view consistency约束以及一个下游任务来训练网络的。

绝大多数keypoint detection的方法要么就是hand-crafted keypoint，要么就是使用监督信息进行学习。利用非监督的方法学习keypoints的最近的工作有：[Unsupervised Learning of Object Structure and Dynamics from Videos](https://proceedings.neurips.cc/paper/2019/file/d82c8d1619ad8176d665453cfb2e55f0-Paper.pdf)和[Unsupervised Learning of Object Keypoints for Perception and Control](https://proceedings.neurips.cc/paper/2019/file/dae3312c4c6c7000a37ecfb7b0aeb0e4-Paper.pdf)，但他们学习到的是2D的keypoints。但是控制机器人的话，我们需要3D的keypoints。

这篇文章使用了一种非监督的方式，设计了一个end-to-end的模型，直接从2D图片里学习到3D keypoints。其是通过一个multi-view consistency约束以及一个下游任务来训练网络的。为了让模型具有普适性并且效果好，其需要满足以下三个性质：(a) 在3D空间里的consistency，也就是说从同一个场景的不同的scenes里所学习到的3D keypoints在3D空间里应该位于同一个位置。(b) 在时间上consistency：同一个keypoint在时间上也应该是连续的 (c) Joint learning with control：因为我们是为了control任务来找的keypoints，联合训练可以使得我们找到的keypoints更加有效。

给定某个camera view的一张image，我们首先预测在image space里的keypoints locations和depth。对于这些从不同的camera views获得的这些keypoints，利用一个differentiable unprojection操作来获得每个keypoint的world coordinate。通过multi-view consistency loss来学习到的不同view之间的consistency可以使得不同view的keypoints可以映射到同一个world coordinate上。这个world coordinate再投射到每个camera view对应的image plane上来重构输入的原image。这样的设计构造了一个differentiable 3D keypoint bottleneck。我们的keypoint学习和RL任务是同时被优化的。整个过程由fig 1所示。我们的模型叫做Keypoint3D。

我们使用了一个multi-view encoder-decoder的结构不使用监督信息来学习3D keypoints。给定同一个场景的$$N$$个view，我们为每个view都给一个encoder和一个decoder。我们为学习到3D keypoints提供了三个unsupervised的signal：1) 我们通过让不同view所学习到的keypoints都能映射到同一个3D空间内的3D keypoint来迫使所学习到的keypoints具有geometrically consistent的特性。2) 我们利用reconstruction loss来惩罚decoder的不准确的reconstruction。3) 我们利用RL任务的reward来反向传播到encoder里从而训练encoder的参数。

$$I_n \in R^{H \times W \times C}$$表示相机$$n$$的输入image，$$n \in 1,\cdots, N$$，而相机$$n$$具有extrinsic matrix $$V_n$$，和intrinsic matrix $$P_n$$。$$K$$是我们keypoints的个数。对于一个3D空间里的点$$\left[x,y,z\right]^T$$和camera $$n$$，我们可以使用extrinsic matrix $$V_n$$和perspective intrinsic matrix $$P_n$$来将其投射到camera coordinate $$\left[u, v, d\right]^T$$，其中$$u,v \in \left[0,1\right]$$是camera plane上归一化后的coordinate，$$d>0$$是depth value，也就是那个点距离camera plane的距离。operator $$\Omega_n: \left[x,y,z\right]^T \longrightarrow \left[u,v,d\right]^T$$表示上述的这种投射，而其的inverse记为$$\Omega_n^{-1}$$。$$\Omega_n, \Omega_n^{-1}$$都是differentiable的，而且可以被解析表示。


**Step1 Keypoint Encoder**

对于每个camera $$n$$，我们将$$I_n$$喂给一个fully convolutional encoder $$\phi_n$$来获得$$k$$个confidence maps，$$C_n^k \in R^{S \times S}$$，以及depth maps，$$D_n^k \in R^{S \times S}$$。对于每个confidence map，我们使用一个spatial softmax来计算一个probability heatmap $$H_n^k$$：

$$H_n^k(i,j) = \frac{exp(C_n^k(i,j)}{\Sigma_{p=1}^S \Sigma_{q=1}^{S} exp(C_n^k(p,q))}$$

heatmap $$H_n^k \in R^{S \times S}$$里的每个值表示的是一个3D keypoint $$k$$出现在从camera $$n$$的角度产生的2D image plane上的这个点的概率。而depth map $$D_n^k$$表示的是在每个camera plane的位置，这个3D keypoint距离camera plane的距离。

然后我们就可以计算keypoint $$k$$在camera $$n$$的camera plane下的坐标了：

$$E\left[u_n^k\right] = \frac{1}{S} \Sigma_{u,v} u H_n^k(u,v)$$

$$E\left[v_n^k\right] = \frac{1}{S} \Sigma_{u,v} v H_n^k(u,v)$$

$$E\left[d_n^k\right] = \Sigma_{u,v} u D_n^k(u,v) H_n^k(u,v)$$

注意到，对于$$u,v$$的计算都除以了$$S$$，也就是图片尺寸，是因为正如我们之前提到的，我们计算的是归一化之后的相机坐标下的keypoint的坐标。

记$$\left[\hat u_n^k, \hat v_n^k, \hat d_n^k \right]^T = \left[E\left[u_n^k\right], E\left[v_n^k\right], E\left[d_n^k\right]\right]^T$$。

为了进一步增加我们的方法的可靠性，我们并不直接将上述的encoder的结果作为keypoint的camera plane的坐标值，而是利用一个高斯分布，其均值为这个值，方差为1，在整个image平面上随机选取，进一步增加了模型的可靠性。


**Step2 Attention**

在预测了每个camera coordinate frame下每个keypoints的坐标之后，我们要想办法将每个keypoint的$$n$$个不同camera下的坐标统一起来。一个最简单的方法就是取平均。但是在某些角度下的keypoints可能被遮挡，从而预测效果并不好。为了解决这个问题，我们利用之前的confidence maps来设计一个加权平均。这使得我们对于那些不那么自信的view里获得的keypoint的权重要小一些，从而不影响整体的效果。

我们可以为每个camera $$n$$获取的keypoint $$k$$设置一个confidence score，其和confidence map $$C_n^k$$的平均值成比例，而且对于$$K$$个keypoints，还做了归一化处理：

$$A_n^k = \frac{exp(\frac{1}{S^2}\Sigma_{p=1}^S \Sigma_{q=1}^S C_n^k(p,q))}{\Sigma_{i=1}^K exp(\frac{1}{S^2}\Sigma_{p=1}^S \Sigma_{q=1}^S C_n^i(p,q))}$$

这个就可以被理解为，对于camera $$n$$来说，其对于每个估计到的keypoint分配的概率，这$$K$$个keypoints的概率总和为1。


**Step3 Extracting world coordinates**

给定之前由encoder预测到的camera plane内的keypoints $$\left[ \hat u_n^k, \hat v_n^k, \hat d_n^k \right]^T$$，$$n= 1,\cdots, N$$，$$k=1, \cdots, K$$，我们可以将其反投射回world coordinates：$$\left[ \hat x_n^k, \hat y_n^k, \hat z_n^k \right]^T = \Omega_n^{-1}(\left[ \hat u_n^k, \hat v_n^k, \hat d_n^k \right]^T)$$。这个就是从camera $$n$$获取到的keypoint $$k$$的world coordinate。对于每个keypoint，我们都有$$N$$个预测的结果，我们利用之前计算的$$A_n^k$$来为每个keypoint计算一个加权的world coordinate：

$$\left[\bar x^k, \bar y^k, \bar z^k \right]^T = \Sigma_{n=1}^N \frac{A_n^k}{\Sigma_{m=1}^N A_n^m} \left[ \hat x_n^k, \hat y_n^k, \hat z_n^k \right]^T$$

**Step4 Keypoint Decoder**

我们在decoder之前，还需要将$$K$$个keypoints都再投射回camera plane来增强模型的学习能力。对于每个camera $$n$$和每个keypoint，我们有$$\left[\bar u, \bar v, \bar d \right]^T = \Omega_n(\left[\bar x^k, \bar y^k, \bar z^k \right]^T)$$。为了获取空间结构信息，对于每个camera和每个keypoint，我们都构建一个高斯分布$$G_n^k \in R^{S \times S}$$，均值为$$\left[\bar u, \bar v \right]$$，方差为$$I_2 / \bar d$$。这个分布使得离得近的那些keypoint，在camera plane上具有更分散的分布。

我们记$$\bar A^k = \frac{1}{N} \Sigma_{n=1}^N A_n^k$$为对于所有view的平均attention，每个camera的decoder $$\psi_n$$就将stacked的高斯maps $$G_n$$作为输入来重构输入image $$I_n$$，而$$G_n = K stack(\left[G_n^1 \bar A^1, \cdots, G_n^K \bar A^K \right])$$。


![Model Structure]({{ '/assets/images/CONTROL-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Overview of our Keypoint3D algorithm. (a) 对于每个camera view，一个CNN将输入image编码为$$K$$个heatmaps以及depth maps；(b) 我们将这些heatmaps当作概率来计算camera plane下keypoint横纵坐标的。我们同时也用heatmap和depth map来计算每个keypoint的深度d，这些$$\left[u,v,d\right]$$再被反投射回到world coordinate里；(c) 我们利用之前的heatmaps来计算每个camera view对于每个keypoint的置信概率，然后对于每个keypoint，我们计算出一个加权的world coordinate；(d) 我们再将每个keypoint的world coordinate投射到每个camera plane上；(e) 从而对于每个camera plane，都有$$K$$个这样的投射，建立$$K$$个高斯map，将它们叠起来，作为decoder的输入，来重构原输入图片；(f) 除了上述的这些loss，我们还将所学习到的3D keypoint的world coordinate与下游任务相结合，来共同优化这个网络。*


