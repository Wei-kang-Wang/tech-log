---
layout: post
comments: false
title: "Keypoint from Point Clouds"
date: 2020-01-01 01:09:00
tags: paper-reading
---

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---


## Unsupervised Methods

### 1. [KeypointDeformer: Unsupervised 3D Keypoint Discovery for Shape Control](https://openaccess.thecvf.com/content/CVPR2021/papers/Jakab_KeypointDeformer_Unsupervised_3D_Keypoint_Discovery_for_Shape_Control_CVPR_2021_paper.pdf)

*Tomas Jakab, Richard Tucker, Ameesh Makadia, Jiajun Wu, Noah Snavely, Angjoo Kanazawa*

*CVPR 2021*

[page](http://tomasjakab.github.io/KeypointDeformer)

现在在Internet上有非常多的3D shapes，给用户提供简单的interface，让他们可以在保留关键shape性质的情况下对3D object做semantically manipulating。文章里为interative editing提出自动检测intuitive和semantically有意义的control points，从而通过控制这些control points来对每个object类别的3D模型进行deformation，并且还保留了他们的shape的细节。

更准确的说，文章将3D keypoints作为shape editing的intuitive和simple interface。Keypoints是那些在一个object类别所有的3D shape间都semantically consistent的3D points。文章提出一个学习框架使用非监督学习的方式来找到这样的keypoints，并且设计一个deformation model来利用这些keypoints在保留局部形状特征的前提下改变物体的shape。这个模型叫KeypointDeformer。

Fig 1描述了KeypointDeformer在inference时候的过程。给一个新的3D shape，KeypointDeformer在它的surface上预测3D keypoints。如果一个用户将chair leg上的keypoint向上移动，整个chair leg都会超相同的方向形变（fig 1下面一行）。KeypointDeformer在这些可操纵的keypoints上提供了可选择的categorical deformation prior，比如说如果一个用户将一个airplane一侧wing上的keypoints向后移动，这一侧的wing会整体向后移动，而另一侧的wing也会随之移动同样的程度（fig 1上面一行）。当用户仅仅希望移动一侧wing的时候，我们的方法同样也允许这种操作。KeypointDeformer可以仅仅对于shape进行editing，也可以对两个shapes做shape alignment，还可以生成新的shapes来扩充datasets。

![overs]({{ '/assets/images/DEFORM-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1 用非监督学习到的3D keypoints来进行shape deformation。我们用非监督学习的方式学习到的3D keypoints可以对object的shape进行intuitive控制。这个figure显示了交互式控制的独立的步骤。红色的箭头标注了keypoints被操作要移动的方向。注意到移动keypoints造成的shape形变是局部的并且object shape是按照intuitive的方式形变的：比如说，将airplane的wing上的keypoint向后拉，则整个wing也会向后倾斜，而保持原本shape的细节不变化。*

尽管3D keypoints对于shape editing来说很有效果，但是获取3D keypoints和deformation models的明确的监督信息不仅很贵，而且是ill-defined的。文章提出了一个unsupervised框架来将寻找3D keypoints和构建deformation model这两个任务同时完成。模型有两个在一起作用的模块：1）一个detecting keypoints的部分；2）一个将keypoints的位移信息传递到shape的其它部分从而进行deformation的deformation model。模型利用将一个source shape align到一个target shape上这样一个任务来训练网络，而且这两个shapes可以是同一个object category里差别很大的两个instances。模型还利用了一个keypoint regularizer，来促进学习到semantically consistent的keypoints。这些keypoints分布的很好，靠近object的surface并且隐式的保留着shape symmetries。KeypointDeformer训练之后所得到的就是一个deformable model，可以基于自动监测到的3D control keypoints来deform一个shape。因为keypoints是低维的，我们还可以在这些keypoints上学习一个category prior，这样就可以进行semantic shape editing了。

文章有以下几个关键的优势：

* 其给了用户一个intuitive并且简单的方法来交互式的控制object shapes
* keypoint prediction和deformable的模型都是unsupervised的
* 由文章的方法所找到的3D keypoints对于shape control来说比其他的keypoints都要好，包括人为标注的
* 文章的unsupervised 3D keypoints对于同一类别的object的不同的instances来说是semantically consistent的，从而给了我们两个point cloud的sparse correspondences。


**Related Work**

*Shape deformation*

文章的方法和geometric modeling里的detail-preserving deformations十分相关，包括[Differential Representations for Mesh Processing](http://mesh.brown.edu/dgp/pdfs/sorkine-cgf2006.pdf)，[As Rigid As Possible Surface Modeling](https://diglib.eg.org/bitstream/handle/10.2312/SGP.SGP07.109-116/109-116.pdf?sequence=1&isAllowed=n)和[Mean Value Coordinates for Closed Triangular Meshes](https://www.cse.wustl.edu/~taoju/research/meanvalue.pdf)。这些方法通过各种类型的限制（比如说points在一个optimization框架里）来允许进行shape editing，但它们一个最主要的问题就是它们仅仅依赖于geometric properties而并没有考虑到semantic attributes或者category-specific shape priors。这样的priors可以通过利用stiffness性质给object surface涂色获得，或者从一系列已经知道correspondence的meshes上学习得到。然而，这种监督信息十分昂贵，而且对于新的shapes来说就不管用了（training set没有见过的shapes，或者有新的priors的shapes）。[Semantic Shape Editing Using Deformation Handles](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.723.3455&rep=rep1&type=pdf)用一个提供了多个控制shape的sliders的data-driven的框架来解决了这个问题。然而这个方法需要一系列从专家标注的信息中提取的predefined attributes。

另一个相关的问题是[deformation transfer](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.126.6553&rep=rep1&type=pdf)，也就是利用两个shapes之间已知的correspondences将source mesh上的deformation转移到target mesh上。近期有些工作利用deep learning来隐式的学习shape correspondences来align两个shapes，比如[1](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yifan_Neural_Cages_for_Detail-Preserving_3D_Deformations_CVPR_2020_paper.pdf)，[2](https://arxiv.org/pdf/1804.08497.pdf)和[3](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_3DN_3D_Deformation_Network_CVPR_2019_paper.pdf)。

*User-guided shape editing*

文章的方法和最近的利用deep learning来学习可以提供对shape做interactive editing的generative模型。[Tulsiani的文章](https://openaccess.thecvf.com/content_cvpr_2017/papers/Tulsiani_Learning_Shape_Abstractions_CVPR_2017_paper.pdf)用primitives来抽象代表shapes，然后通过surface的primitives的deformation来edit shape。但是，shape editing并不是它们主要的目标，而且也不清楚直接进行primitive transformation能多大程度的保留local shape details。近似的工作进一步改进了这个方法，他们通过学习一个[point-based](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hao_DualSDF_Semantic_Shape_Manipulation_Using_a_Two-Level_Representation_CVPR_2020_paper.pdf)、[shape handles](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gadelha_Learning_Generative_Models_of_Shape_Handles_CVPR_2020_paper.pdf)或者[disconnected shape manifolds](https://openaccess.thecvf.com/content_ICCV_2019/papers/Mehr_DiscoNet_Shapes_Learning_on_Disconnected_Manifolds_for_3D_Editing_ICCV_2019_paper.pdf)的primitives的generative model来改进原先的基于primitives的model的缺点。这些方法通过找到最佳匹配用户editing的latent primitive representations来做到interactive editing。但是他们的方法所用到的用户interface比较复杂，需要素描或者直接操控primitives。而且最关键的，因为这些editing是基于generative models的，这些方法可能会改变original shape的local details。而相对而言，我们直接对原shape进行deform，会有更好的shape detail的保留。我们将提出的方法和[DualSDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hao_DualSDF_Semantic_Shape_Manipulation_Using_a_Two-Level_Representation_CVPR_2020_paper.pdf)的结果进行对比来阐述上述的优势。


*Unsupervised keypoints*

在2D keypoint discovery领域，unsupervised方法有很多论文都已经有了不错的结果，[Unsupervised learning of object landmarks through conditional image generation](https://proceedings.neurips.cc/paper/2018/file/1f36c15d6a3d18d52e8d493bc8187cb9-Paper.pdf)，[Self-supervised learning of interpretable keypoints from unlabelled videos](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jakab_Self-Supervised_Learning_of_Interpretable_Keypoints_From_Unlabelled_Videos_CVPR_2020_paper.pdf)，[Self-supervised learning of a facial attribute embedding from video](https://arxiv.org/pdf/1808.06882.pdf)，[Unsupervised learning of landmarks by descriptor vector exchange](https://openaccess.thecvf.com/content_ICCV_2019/papers/Thewlis_Unsupervised_Learning_of_Landmarks_by_Descriptor_Vector_Exchange_ICCV_2019_paper.pdf)[Unsupervised learning of object landmarks by factorized spatial embeddings](https://openaccess.thecvf.com/content_ICCV_2017/papers/Thewlis_Unsupervised_Learning_of_ICCV_2017_paper.pdf)[Unsupervised discovery of object landmarks as structural representations](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Unsupervised_Discovery_of_CVPR_2018_paper.pdf)，但在3D keypoint discovery领域，unsupervised的方法却还没有被研究完全。[Discovery of latent 3d keypoints via end-to-end geometric reasoning](https://proceedings.neurips.cc/paper/2018/file/24146db4eb48c718b84cae0a0799dcfc-Paper.pdf)利用3D pose information作为supervision来从两张关于同一个object的不同角度的图片中检测3D keypoints。我们这篇文章聚焦于在从3D shapes上学习3D keypoints。[Unsupervised learning of intrinsic structural representation points](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Unsupervised_Learning_of_Intrinsic_Structural_Representation_Points_CVPR_2020_paper.pdf)输出一个结构化的3D representation来获取sparse或者dense的shape correspondences。和我们这篇文章里进行3D keypoints discovery方法最像的文章是[Unsupervised learning of category-specific symmetric 3d keypoints from point sets](https://arxiv.org/pdf/2003.07619.pdf)，他们采用了显式的对称性限制条件。在这篇文章里，我们是为了shape control这个任务来使用非监督的方式寻找keypoints。尽管我们重点在于shape editing，但是我们的模型构造使得我们学习到了semantic consistent的3D keypoints。这样的以非监督方式学到的3D keypoints对于机器人来说没准是有用的，那些机器人相关的任务可以将3D keypoints作为latent representation用来控制机器人，而他们现在还需要手动定义3D keypoints来作为控制机器人的信号。


**Method**

目标是学习一个keypoint detector，$$\Phi: x \longrightarrow p$$，来将一个3D object shape $$x$$映射到一个semantically consistent的3D keypoints的集合$$p$$。我们同时也想学习一个输入为keypoints的conditional deformation model，$$\Psi: (x, p, p^{'}) \longrightarrow x^{'}$$，将shape $$x$$利用deformed control keypoints映射到shape $$x^{'}$$，其中$$p$$描述的是initial（source）keypoint locations，$$p^{'}$$描述的是target keypoint locations。为keypoints和deformation model获取显式的监督信息十分expensive而且ill-defined。因此，我们提出了一个unsupervised的learning框架来训练上述提到的两个functions。我们通过设计了一个pair-wise shape alignment的辅助任务来实现，这个辅助任务的核心想法就是将keypoints learning和deformation model联合起来学习，从而可以对两个任意的shapes做alignment。更仔细地说，模型首先利用一个Siamense network在source和target shapes上预测3D keypoint locations。之后我们利用检测到的keypoints的对应关系来deform source shape（检测到的keypoints是默认有序的，从而有着对应关系）。为了保持local shape detail，我们使用了一个基于keypoints的cage-based deformation方法。我们使用了一个新颖的但十分简单高效的keypoint regularization term，使得keypoints是well-distributed的，并且距离object surface很近。Fig 2显示了我们的模型的整体框架。

![framework]({{ '/assets/images/DEFORM-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. Model。我们的模型使用预测到的unsupervised keypoints $$p$$和$$p^{'}$$来将source shape $$x$$ aligns到target shape $$x^{'}$$。unsupervised keypoints描述了object的pose并用作deformation的control points。整个模型使用在deformed source shape $$x^{\*}$$和target shape $$x^{'}$$之间的similarity loss和keypoint regularization loss来进行end-to-end的训练。在interactive shape manipulation的test time，用户可以选择只输入一张source shape $$x$$，keypoint predictor $$\Phi$$就会预测一些unsupervised 3D keypoints $$p$$出来。然后用户可以手动控制keypoints $$p$$使其变成target keypoints$$p^{'}$$，然后再用deformation model $$\Psi$$来生成deformed source shape $$x^{\ast}$$，如fig 1，fig 9或者project page上的补充材料里的视频所示。*

**1. Shape Deformation with Keypoints**

我们将每个object表示为一个point cloud $$x \in R^{3 \times N}$$，是从object mesh里均匀采样得来的。我们先从source和target里预测keypoints。keypoint predictor $$\Phi$$使用$$x$$作为输入，输出一个ordered set，$$p = (p_1,...,p_K) \in R^{3 \times K}$$，表示的是3D keypoints。这个keypoint predictor的encoder对于source和target是公用的，使用Siamese architecture来实现。而shape deformation function $$\Psi$$的输入是source shape $$x$$，source keypoints $$p$$和target keypoints $$p^{'}$$。在test阶段，输入一张图片，得到了其的source keypoints，用户editing之后得到了target keypoints，之后生成输入图片的deformation，整个interactive shape deformation过程如fig 2所示。

为了在deform object的过程中保持它的local shape details，我们使用最近刚出现的[differentiable cage-based deformation algorithm](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yifan_Neural_Cages_for_Detail-Preserving_3D_Deformations_CVPR_2020_paper.pdf)。cages是个很经典的shape modeling方法，其使用一个粗糙的封闭的mesh将shape包起来。deform cage mesh就会导致里面包裹的shape也发生deformation。cage-based deformation function $$\beta: (x,c,c^{\ast}) \longrightarrow x^{\ast}$$的输入是source control cage $$c$$，deformed control cage $$c^{\ast}$$，以及source shape $$x$$（也就是一开始从mesh里采样得到的point cloud）。我们通过一开始用一个球体包住source shape $$x$$，之后再将每个cage vertex $$c_V$$向object的中心推进直到它和object surface之间只有一个很小的距离这样一种方法来为每个shape都自动的获取包裹其的cage。fig 2显示了所得到的cage的样子。尽管cages对于shape-preserving deformation来说是个有用的方法，但通过deform cages来获得内部的shape的deformation并不是那么的直观，特别是对新手用户来说，因为cage vertex并不直接落在shape的表面上，并没有一个粗糙的structure，而且在不同的shape之间（同一个object或者同一类object的不同姿态的shape）并不semantically consistent。我们提出keypoints用作操纵cage deformation的方式更为合理。

为了用我们检测到的keypoints来控制object deformation，我们需要将这些keypoints和cage vertices联系起来。我们通过使用一个linear skinning function，首先计算source和target keypoints之间的relative distance，$$\delta p = p^{'}-p$$，然后将一个可学习的influence matrix，$$W \in R^{C \times K}$$乘上$$\delta p$$，在加到source cage vertices，$$c_V$$上，就获得了新的target cage vertices，$$c_{V}^{\ast}$$。其中，$$p,p^{'},\delta p$$都是$$K \times 3$$的矩阵，$$c_V, c_V^{\ast}$$是$$C \times 3$$的矩阵，而$$K$$和$$C$$分别表示keypoints和cage vertices的个数。所以deformed cage vertices，$$c_V^{\ast}$$计算方式为：

$$c_V^{\ast} = c_V + W \delta p$$

为了满足对于每个shape来说cage是唯一的这样的事实，我们将上述的influence matrix，$$W$$，设置为输入shape $$x$$的一个函数。详细的说，influence matrix是一个composition，$$W(x) = W_C + W_I(x)$$，其中$$W_c$$是对于每一类object的所有instances都共用的canonical matrix，而$$W_I(x)$$则是每个instance独自的offset，是利用influence predictor $$W_I = \Gamma(x)$$以source shape $$x$$为输入计算而来。我们同时也通过最小化其Frobenius norm来regularize这个instance specific offset，$$W_I$$，为了防止它过拟合influence matrix $$W$$。我们将这个regularizer命名为$$L_{inf}$$。最后，我们限制$$W$$使得每个keypoint最多只能影响$$M$$个最近的cage vertices来实现locality。


**2. Losses和Regularizers**

我们的KeypointDeformer是通过最小化source和target shape之间的similarity loss，再加上keypoint regularization loss和instance-specific influence matrix regularization term，利用SGD实现的end-to-end的训练。

**Similarity loss**

理想情况下，我们希望利用已知的meshes之间的correspondence来计算deformed source shape $$x$$和target shape $$x^{'}$$之间的similarity。但是这样的correspondence是不存在的，因为我们希望能在最普遍的object category CAD模型上训练。我们通过将source shape和target shape都表示为point cloud，然后再计算他们之间的Chamfer distance来近似这个similarity loss。这个loss记为$$L_{sim}$$。


**Farthest Point Keypoint regularizer**

我们提出了一个简单有效的keypoint regularizer $$L_{kpt}$$来使得预测的keypoints $$p$$是well-distributed的，也就是再object surface上，并且能保持这个shape category本身的symmetric structure。具体来说，我们设计了一个**Farthest Sampling Algorithm**来从输入的shape mesh里采样一个无序集合$$q = \{q_1,...,q_J\} \in R^{3 \times J}$$作为point cloud。采样的起始点是随机的，所以每次我们计算这个regularization loss的时候我们都使用的不同的point cloud $$q$$。给定这些随机的farthest points，regularizer最小化所预测的keypoints $$p$$和这些采样到的点$$q$$之间的Chamfer distance。也就是说，这个regularizer希望keypoint detector $$\Phi$$能够学习到那些和$$q$$分布类似的keypoints。Fig 3展示了$$q$$的特性。这些采样到的点对于提供了输入的object shape $$x$$的一个均匀的覆盖，其再不同的instances之间比较稳定，而且保持了最初input shape的symmetric结构。

![LOSS]({{ '/assets/images/DEFORM-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 3. Farthest Point Keypoint regularizer. 我们使用一个随机的初始点来做farthest point sampling，来regularize预测到的keypoints。(a) 展示了对于一个给定的点，其被farthest point sampling algorithm所选到的频率。颜色越深表明这个点被选到的概率越大。采样点的期望locations对原shape有一个很好的覆盖而且保留了原shape的symmetry特征。而且，它们的子集在不同的object instances之间保持了semantically stable。使用采样点的期望locations作为keypoint location的prior效果很好，因为keypoint predictor会学会对这些采样点里的噪声比较robust。在airplane的例子里我们可以看到，fuel tank的顶点（红圈标记）并没有被keypoint predictor用作keypoint，(b) 而wing的顶点（绿圈）则被选中为keypoint，因为其在数据集里更加consistent（大多数飞机都有wings，但很多并没有fuel tank）。* 

这个regularization的另一个intuition是我们可以将这些采样的farthest points $$q$$理解为keypoint locations的一个noisy prior。这个prior并不是完美的——在某些shape上可能会遗失某些重要的点，或者有一些不合理的点——但是neural network keypoint predictor会以一种对这些noise robust的方式学到keypoint的locations，而且会偏向于学习那些consistent的keypoints，如fig 3所示。

**Full objective**

总结来说，我们的training objective是：

$$L = L_{sim} + \alpha_{kpt}L_{kpt} + \alpha_{inf}L_{inf}$$

其中$$\alpha_{kpt}$$和$$\alpha_{inf}$$是scalar loss系数。我们的方法很简单而且并不需要对于shape deformation增加额外的shape specific regularization，比如说point-to-surface距离，normal consistency，symmetry losses。这是因为keypoints提供了一个shapes之间的low-dimensional的correspondence，而且cage deformations是这些keypoints的一个linear function，从而阻止了那些会导致local deformation的极端的deformations。


**3. Categorical Shape Prior**

因为我们利用一系列semantically consistent的keypoints来代表一个object shape，我们可以通过计算training set里的shape对应的keypoints的PCA来获取categorical shape prior。这个prior可以用来指导keypoint manipulation，也就是上面提到的$$W_C$$。比如说，如果用户想要改变一个airplane一个wing上的一个keypoint，根据寻找到能够最佳重构这个被改变的keypoint的新位置的PCA basis coefficients，其余的keypoints就会被这些basis coefficients”同步协调“。从而这些keypoints就会根据这个prior（也就是这个PCA）落到新的位置。这个prior还可以通过采样一系列新的keypoints来生成新的shapes：调整某些keypoints，然后PCA经过计算basis coefficients来对所有的keypoints位置进行调整，从而得到了新的keypoints位置，然后利用上述的deformation model来生成新的shape，就可以将这个新的shape加入已有的3D shape datasets里。




### 2. [Unsupervised Learning of Category-Specific Symmetric 3D Keypoints from Point Sets](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700545.pdf)

*Clara Fernandez-Labrador, Ajad Chhatkuli, Danda Pani Paudel, Jose J. Guerrero, Cedric Demonceaux, Luc Van Gool*

*ECCV 2020*

>又是ethz Luc Van Gool大佬的作品。

从同一类别的objects的集合里自动找到category-specific的3D keypoints是一个非常有挑战性的问题，这篇文章要用一种unsupervised方法来学习上述这种3D keypoints。

从没有标签的3D数据中直接获取3D keypoints——来表示有意义的shapes和semantics——从而使得这些keypoints的作用和手动定义的那些一样，因为很困难并没有受到过多的关注。

因为objects本身就是位于三维空间内，所以3D keypoints对于geometric reasoning任务效果更好是很自然的。对于给定的3D keypoints，它们在2D图片中的counterpart可以很简单的用camera projection来实现。然而，能直接从3D数据（point clouds）上预测keypoints会有优势，因为很多时候multiple camera views或者multiple images是不可获取的。在这篇文章里，我们关注如何从3D point clouds中直接学习keypoints locations。实际上，带有keypoints的3D structures对于很多的应用包括registration，shape completion，shape modeling都是足够使用的了，并不需要它们的2D counterparts。

因为deformation或者对比同一个类别不同的objects，3D objects就会有shape variations，找到consistent的keypoints对于geometric reasoning就很重要了。我们可不可以自动找到这些keypoints，其对于同一个类别的同一个objects的deformation以及同一个类别的不同objects之间的差异都是consistent的？这是这篇文章主要想要回答的问题。更进一步，我们想要用unsupervised方法直接从3D point clouds上找到这样的keypoints。我们将这些keypoints称为category-specific，它们被期望于能描述objects的shape信息，并能够在同一类别的所有objects之间保持有序的对应关系。更加严格地说，我们将category-specific keypoints所需要满足的性质定义为：1）对于同一类别的具有不同shapes或者不同alignments的不同的objects具有很好的泛化效果，也就是说对于形状不同或者没有对齐（有旋转角度）的同一个类别的不同的objects，找到的仍然是这些keypoints；2）能做到一对一的有序的对应并且具有semantic consistency；3）在保持shape symmetry的情况下能够代表这一类别的object的形状特征。

但是在3D point clouds上学习category-specific keypoints是一个非常难的问题。而如果数据还是misaligned（可能会有旋转，因为平移对于CNN来说没有影响）并且要使用unsupervised方法，那这个问题就更难了。相关的工作并没有全部考虑这些因素，而是只考虑了一部分：[USIP: Unsupervised Stable Interest Point Detection from 3D Point Clouds](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_USIP_Unsupervised_Stable_Interest_Point_Detection_From_3D_Point_Clouds_ICCV_2019_paper.pdf)使用aligned数据并且学习的不是category-specific keypoints，而是任意object的通用keypoints，[6-DoF Object Pose from Semantic Keypoints](https://arxiv.org/pdf/1703.04670.pdf)在2D图像上使用监督信息，[Discovery of Latent 3D Keypoints via End-to-end Geometric Reasoning](https://proceedings.neurips.cc/paper/2018/file/24146db4eb48c718b84cae0a0799dcfc-Paper.pdf)使用aligned 3D数据并且使用多张知道pose的2D图片，其在没有显式考虑shapes的情况下获得了category specific的3D keypoints。[3D Landmark Model Discovery from a Registered Set of Organic Shapes](https://clementcreusot.com/publications/papers/creusot2012-PCP.pdf)使用预先定义好的local shape descriptors和一个template模型，而且只针对faces。

在这篇文章里，我们说明拥有上述性质的category-specific keypoints可以用unsupervised的方式，通过基于未知的linear basis shapes结合non-rigidity来对它们建模的方式学习到。在考虑具有对称性的object类别的时候，我们还在deformation模型上加入了未知的reflective symmetry。对于那些没有对称性的object类别，我们使用symmetric linear basis shapes来对所谓的symmetric deformation spaces进行建模，比如说human body deformations。我们所提出的learning方法并不需要假设shapes是aligned好的、或者预先计算好了basis shapes或者已经知道了对称平面，所有的这些值都是通过end-to-end的方式学到的。我们的模型相比较于之前NRSfM([Multiview Aggregation for Learning Category-Specific Shape Reconstruction](https://arxiv.org/pdf/1907.01085.pdf)和[Symmetric Non-Rigid Structure from Motion for Category-Specific Object Structure Estimation](https://arxiv.org/pdf/1609.06988.pdf))里的方法来说十分有效而且灵活。我们通过将一个object类别的shape basis和对称平面都作为neural network的参数来学习的办法来实现。训练的时候每次输入是一个3D point cloud，并没有使用Siamese-like的模型结构。在inference的时候，网络会预测basis coefficients和pose，再用来估计instance-specific keypoints。


**Related Work**

objects的category specific keypoints在NRSfM方法里用得很多，但是却很少有论文来研究如何找到它们。考虑模型的输出，我们的方法和[Discovery of Latent 3D Keypoints via End-to-end Geometric Reasoning](https://keypointnet.github.io/keypointnet_neurips.pdf)很像，这篇文章通过解决一个辅助任务来学习category-specific 3D keypoints，这个辅助任务是学习同一个object的不同角度的view之间的rigid transformation，而且这个方法假设图片是aligned好的。尽管这个方法在2D和3D上都显示了比较好的结果，但是其并没有显式的对shape进行建模。结果就是，这个方法需要同一个object的不同角度的照片是aligned好的，从而才能计算keypoint correspondences。在[6-DoF Object Pose from Semantic Keypoints
](https://arxiv.org/pdf/1703.04670.pdf?ref=https://githubhelp.com)里同样对于六自由度estimation考虑了类似的任务，其使用了一个low-rank shape prior来为3D keypoints提供condition。尽管low-rank shape modeling是个很有力的工具，但这篇文章还是需要对heatmap prediction进行监督，而且依赖于aligned shapes和预先计算好的shape basis。[Single Image 3D Interpreter Network](https://arxiv.org/pdf/1604.08685.pdf)同样也使用low-rank shape prior，但是它们的训练完全基于监督的方法。而且，上述所有的方法都是从images上利用heatmaps的方式学习到keypoints，再将其提升到3D空间内的。不同于上述的这些工作，[3D Landmark Model Discovery from a Registered Set of Organic Shapes
](https://clementcreusot.com/publications/papers/creusot2012-PCP.pdf)使用了deformation model和symmetry来直接从3D数据上预测keypoints，但是其需要一个face template，aligned shapes以及已知的shape basis。某一类别的objects的shape modeling在NRSfM工作中早就已经被研究透了。linear low-rank shape basis，low-rank trajectory basis，isometry或者piece-wise rigidity是NRFfM的提出的不同的方法。最近，有一些工作使用low-rank shape basis来设计可被学习的模型。另一个能被用来进行model shape category的方法是reflective symmetry，其也和object pose密切相关。尽管[Symmetric Non-Rigid Structure from Motion for Category-Specific Object Structure Estimation](https://yuan-gao.net/pdf/ECCV2016%20-%20Sym-NRSfM.pdf)说明low-rank shape basis可以使用未知的reflective symmetry进行构造，如何将其改造成可学习的NRSfM方法并不简单。还有最近的工作假设symmetry plane是从几个已知的planes里挑选出来的。这些方法都没有为non-rigidly deforming objects比如说human body构造symmetry。[Unsupervised Learning of Probably Symmetric Deformable 3D Objects from Images in the Wild](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wu_Unsupervised_Learning_of_Probably_Symmetric_Deformable_3D_Objects_From_Images_CVPR_2020_paper.pdf)在一个warped canonical空间内概率化的考虑模型symmetry，来对不同objects进行3D reconstruction。

shape modeling是我们工作的一个重要的方面，另一个重要的方面是如何从一个unordered point set上学习到ordered keypoints。尽管在deep neural networks领域对于point sets来说有了好几个很好的成果（[PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)，[PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://proceedings.neurips.cc/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf)），但这些成果和在images上取得的巨大成功相比就不算啥了。[Discovery of Latent 3D Keypoints via End-to-end Geometric Reasoning](https://proceedings.neurips.cc/paper/2018/file/24146db4eb48c718b84cae0a0799dcfc-Paper.pdf)利用一个Siamese architecture通过一个正确预测rotation的代理任务来用非监督学习的方式预测3D keypoints。[3DFeat-Net: Weakly Supervised Local 3D Features for Point Cloud Registration](https://openaccess.thecvf.com/content_ECCV_2018/papers/Zi_Jian_Yew_3DFeat-Net_Weakly_Supervised_ECCV_2018_paper.pdf)利用alignment作为代理任务来预测keypoints。在下面的章节里，我们将会展示如何利用low-rank symmetric shape basis来对shape instances进行建模，以及如何利用shape modeling来预测有序的category-specific keypoints。


**3. Background and Theory**

**Notations**

我们利用花体拉丁字母（比如$$\mathcal{V}$$）或者加粗大写拉丁字母（比如**V**）来表示sets和matrices。小写或者大写的普通字母，比如K来表示scalars。小写的加粗拉丁字母来表示vectors，比如**v**。小写的拉丁字母来表示indices，比如$$\mathcal{i}$$。大写的希腊字母来表示mappings或者functions，比如$$\Pi$$。我们使用$$\mathcal{L}$$来表示loss functions。operator mat将一个向量**v** $$\in R^{3N \times 1}$$转换为一个矩阵**M** $$\in R^{3 \times N}$$。

**3.1 Category-specific Shape and Keypoints**

我们的shape使用point clouds表示的，由一个无序的points集合来表示$$S=\lbrace s_1, s_2, \cdots, s_M \rbrace, s_j \in R^3, 1 \leq j \leq M$$。所有的同一个类别的这样的point clouds定义的shape组成了category shape space $$\mathcal{C}$$。我们将$$\mathcal{C}$$里的第i个category-specific shape instance记为$$S_i$$。category shape space $$\mathcal{C}$$可以是一系列离散的shapes，也可以是由deformation function $$\Psi_C$$生成的一个光滑的category-specific shapes流形。我们这篇文章关注点在于从point cloud $$S_i$$里学到有用的3D keypoints。为了达到这个目标，我们这一节定义category-specific keypoints，并且介绍生成keypoints的模型。

*category-specific keypoints* 我们将一个shape $$S_i$$的category-specific keypoints表示为一系列的points，$$P_i = (p_{i1}, p_{i2}, \cdots, p_{iN}), p_{ij} \in R^3, 1 \leq j \leq N$$。和shape $$S_i$$不同，这个集合$$P_i$$是有序的。我们的目标就是学习一个mapping $$\Pi_C: S_i \longrightarrow P_i$$来为$$\mathcal{C}$$内任意一个shape $$S_i$$学习到category-specific keypoints。在之前我们已经定义了category-specific的keypoints应该是什么样的。如果用数学语言来描述的话就是这样的：

* 1) Generalization: $$\Pi_C(S_i) = P_i, \forall S_i \in C$$
* 2）corresponding points and semantic consistency: 给定$$S_a, S_b \in \mathcal{C}$$，我们希望$$P_{aj} \iff P_{bj}$$。而且$$P_{aj}$$和$$P_{bj}$$需要有相同的semantics。
* 3) representative-ness: $$vol(S_i) = vol(P_i)$$以及$$p_{ij} \in S_i$$，其中$$vol(.)$$是一个对于shape计算volume的算子。如果$$S_i \in \mathcal{C}$$有reflective symmetry，那么$$P_i$$也得有相同的symmetry。


**3.2 Category-specific Shapes as Instances of Non-rigidity**

近期有一些工作将category shapes里的不同的instances利用non-rigid deformations来进行建模（[C3DPO: Canonical 3D Pose Networks for Non-Rigid Structure From Motion](https://openaccess.thecvf.com/content_ICCV_2019/papers/Novotny_C3DPO_Canonical_3D_Pose_Networks_for_Non-Rigid_Structure_From_Motion_ICCV_2019_paper.pdf)，[Single Image 3D Interpreter Network](https://dspace.mit.edu/handle/1721.1/114448)，[Multiview Aggregation for Learning Category-Specific Shape Reconstruction](https://proceedings.neurips.cc/paper/2019/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html)，[Deep Non-Rigid Structure from Motion](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kong_Deep_Non-Rigid_Structure_From_Motion_ICCV_2019_paper.pdf)）。这个想法的基于这些shapes通常都有geometric similarities这样一个事实。结果是，存在一个deformation function $$\Psi_C: S_T \longrightarrow S_i$$，将一个global shape property $$S_T$$（shape template或者basis shapes）映射到一个category shape instance $$S_i$$。然而，对$$\Psi_C$$进行建模是很困难的，而且在很多情况下$$\Psi_C$$可能并不存在一个很简单的表述。这个问题，就是为什么dense Non-rigid Structure from Motion (NRSfM)任务这么困难的原因。从另一个角度来看，我们可以考虑一个deformation function $$\Phi_C: P_T \longrightarrow P_i$$，从global keypoints property $$P_T$$映射到每个instance的category-specific keypoints $$P_i$$。$$\Phi_C$$要满足，$$\Phi_C$$的一对对应点，也是$$\Psi_C$$的一对对应点。并且如果以点对来定义$$\Psi_C$$和$$\Phi_C$$，应该有$$\Phi_C \subset \Psi_C$$。和$$\Psi_C$$不同的是，$$\Phi_C$$的建模可能会很简单。因此，我们选择在keypoints空间内$$P = (P_1, P_2, \cdots, P_L)$$来寻找non-rigidity modeling。而所学到的$$\Phi_C$$就可以是$$\Psi_C$$的一个抽象。non-rigidity可以被用来定义prediction function $$\Pi_C$$：

$$\Pi_C(S_i;\theta) = \Phi_C(r_i;\theta) = P_i \tag{1}$$

其中$$\theta$$是$$\Pi_C$$的函数parameters，$$r_i$$是每个instance特有的vector parameter。在我们的设定下，我们希望能从$$\mathcal{C}$$的shape instances里用非监督的方式来学习到参数$$\theta$$。在NRSfM的设定里，对shape deformation进行建模的两种常见方法是low-rank shape prior和isometric prior。在这篇文章里，我们对instance-wise symmetry和deformation space的symmetry使用low-rank shape prior进行建模。

**3.3 Low-rank Non-rigid Representation of Keypoints**

NRSfM关于low-rank shape basis的方法是rigid orthographic factorization prior的一个自然的拓展，在[Recovering Non-Rigid 3D Shape from Image Streams](http://vision.jhu.edu/reading_group/Bregler2.pdf)和[Shape and motion from image streams under orthography: a factorization method](http://users.eecs.northwestern.edu/~yingwu/teaching/EECS432/Reading/Tomasi_TR92.pdf)。关键的想法是很大一部分object deformations都可以用$$K$$个不同pose的basis shapes的线性组合来表示，而且这个$$K$$是个不大的数。在rigid的情况下，这个$$K$$就是1。在non-rigid的情况下，这个$$K$$大一些，具体的数字取决于deformations的复杂程度。考虑$$\mathcal{C}$$里$$F$$个shape instances，并且在每个shape instance的keypoints instance $$P_i$$里考虑$$N$$个点。下面的式子描述了使用shape basis进行的projection：

$$P_i = \Phi_C(r_i;\theta) = R_i mat(B_C c_i) \tag{2}$$

其中$$B_C = (B_1, \cdots, B_K), B_C \in R^{3N \times K}$$构成了low-rank shape basis。vector $$c_i \in R^K$$表示对于instance $$i$$的不同的shape basis的线性系数。从而每个instance的keypoints就可以完全被basis $$B_C$$和系数$$c_i$$所表示了。之后，projection matrix $$R_i \in SO(3)$$就是instance $$i$$的rotation matrix。$$mat(.)$$表示将得到的$$3N \times 1$$的矩阵转换为$$3 \times N$$的矩阵。

在我们这个问题里，$$P_i, c_i, B_C, R_i$$都是不知道的。我们要通过上述的式子来学习，其中$$\theta$$包括了$$\Phi_C$$的function parameter，basis $$B_C$$，而$$r_i$$则包括了instance-wise pose $$R_i$$和系数$$c_i$$。


**3.4 Modeling Symmetry with Non-Rigidity**

很多object种类的shapes都有固定的reflective symmetry。为了寻求并使用这种symmetry，我们考虑两个不同的priors：instance-wise symmetry和symmetric deformation space。

*Instance-wise symmetry* 

关于一个固定平面的instance-wise reflective symmetry在很多rigid object类别上都能被观察到（比如[A Scalable Active Framework for Region Annotation in 3D Shape Collections](https://dl.acm.org/doi/pdf/10.1145/2980179.2980238)和[3D ShapeNets: A Deep Representation for Volumetric Shapes](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wu_3D_ShapeNets_A_2015_CVPR_paper.pdf))。在NRSfM里，这样的symmetry被和shape basis prior结合起来使用过。然而，用来同时学习symmetry和shapes的一个简便的representation还未被研究过。最近的learning-based的工作[Multiview Aggregation for Learning Category-Specific Shape Reconstruction](https://proceedings.neurips.cc/paper/2019/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html)和[Normalized Object Coordinate Space for Category-Level
6D Object Pose and Size Estimation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Normalized_Object_Coordinate_Space_for_Category-Level_6D_Object_Pose_and_CVPR_2019_paper.pdf)通过在几个平面上穷举式的寻找来使用symmetry prior，然后再用来预测symmetric dense non-rigid shapes。但是这样的方法在shapes并没有aligned的情况下是不行的。我们将公式2改造一下就可以加入instance-wise symmetry：

$$P_{i\frac{1}{2}} = R_i mat(B_{C\frac{1}{2}}c_i), P_i = \left[P_{i\frac{1}{2}}, A_C P_{i\frac{1}{2}} \right] \tag{3}$$

其中$$P_{i\frac{1}{2}} \in R^{3 \times N/2}$$表示的是一半的category-specific keypoints。$$P_{i\frac{1}{2}}$$被$$A_C \in R^{3 \times 3}$$反射一次，再和$$P_{i\frac{1}{2}}$$连起来，获得最终全部的category-specific keypoints。$$B_{C\frac{1}{2}} \in R^{3N/2 \times K}$$来表示对于一半的keypoints的shape basis，注意$$K$$并没有变化，说明shape basis并没有变少，只是我们只表示一半的keypoints。reflection operator $$A_C$$通过一个过原点的unit normal vector $$n_C \in R^3$$来定义。我们从公式2到公式3的变化，既减小了计算量，又将对称性加入了keypoints之中。

*Symmetric deformation space* 

在很多non-rigid objects里，shape instances并不是symmetric的。但是再deformation space里可能会存在symmetry，比如说，human body。假设在$$\mathcal{C}$$里的某一个shape instance $$S_k$$有着关于$$n_C$$的reflective symmetry，这样我们就可以把其分为两部分：$$S_{k\frac{1}{2}}$$和$$S_{k\frac{1}{2}}^{'}$$。而且我们可以认为对于这个类别的所有的shape instances，这个reflective symmetry都是存在的。

**Definition 1(Symmetric deformation space)** 在$$\mathcal{C}$$里的某一个shape instance $$S_k$$有着关于$$n_C$$的reflective symmetry，这样我们就可以把其分为两部分：$$S_{k\frac{1}{2}}$$和$$S_{k\frac{1}{2}}^{'}$$。如果对于任意的一半的shape deformation instance $$S_{i\frac{1}{2}}$$，其都存在一个shape instance $$S_j \in \mathcal{C}$$，使得$$S_{j\frac{1}{2}}^{'}$$和$$S_{i\frac{1}{2}}$$对称，那么称$$\mathcal{C}$$是一个symmetric deformation space。

上述定义对于keypoints shape space $$P$$来说一样成立。instance-wise symmetric space是上述定义的一个特例。但是公式3并不能描述symmetric deformation space里的keypoints instances。我们通过引入可以被非对称的加权的symmetric basis来对这种keypoints进行建模，从而：

$$P_i = R_i \left[mat(B_{C\frac{1}{2}} c_i), mat(B_{C\frac{1}{2}}^{'} c_i^{'}) \right] \tag{4}$$

其中$$B_{C\frac{1}{2}}^{'}$$是将$$B_{C\frac{1}{2}}$$利用$$A_C$$反转而得，而$$c_i^{'}$$构成第二部分basis的权重。尽管公式4增加了计算量，但是其增加了模型能够表示symmetry deformation space的能力。从而我们有了如下的proposition：

**Proposition 1** 如果$$B_{C\frac{1}{2}}^{'}$$和$$B_{C\frac{1}{2}}$$关于某个平面对称，如果系数$$c_i$$和$$c_i^{'}$$满足同一个概率分布，那么上述最新的公式就表示了一个symmetric deformation space。

有了proposition 1的结论，我们就可以利用公式4来表示non-rigid symmetric objects，并且只要我们满足$$c$$和$$c^{'}$$的分布相同这样一个条件，我们仍然可以保持symmetry的性质。


**4. Learning Category-Specific Keypoints**

在这一节里，我们通过对$$\Phi_C$$进行建模来描述使用非监督方法学习category-specific keypoints的过程。更加准确地说，我们希望将函数$$\Pi_C: S_i \longrightarrow P_i$$作为一个参数为$$\theta$$的neural network，使用从$$\Phi_C$$里获得的supervisory signal来学习。关于从point sets上学习keypoints，[USIP: Unsupervised Stable Interest Point Detection From 3D Point Clouds](https://openaccess.thecvf.com/content_ICCV_2019/html/Li_USIP_Unsupervised_Stable_Interest_Point_Detection_From_3D_Point_Clouds_ICCV_2019_paper.html)训练了一个Siamese network来对rigid objects的keypoints进行预测，但keypoints的顺序是不知道的，这个方法对于rotation是稳定的。我们部分的网络结构是受这篇文章的启发的，但它们的源头都是基于[PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)。我们使用的是单个输入，从而避免了Siamese network很复杂的训练过程。fig 1展示了网络结构，它的输入是一个shape $$S_i$$，且在$$SO(2)$$里是misaligned的。

![ModelssStructure]({{ '/assets/images/SYMMETRY-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*fig 1. Network Architecture。pose and coefficient branch和additional learnable parameters生成了category-specific keypoints。node branch预测那些领导训练过程的nodes。第3节是建模，第4节是训练。

>$$SO(2)$$表示的是det=1的orthogonal空间，也就是rotation matrix组成的空间，2表示是2维的，也就是所有的2维旋转操作组成的group称为$$SO(2)$$，其可以被很简单的表示为$$SO(2) = \left[\left[cos \theta, -sin \theta \right], \left[sin \theta, cos \theta \right]\right]$$，其中$$\theta \in R$$。

这种设置是很合理的，因为point clouds一般都是沿着竖直方向对齐，所以其实它们是misaligned的。

我们下面描述一下网络的各个不同的组成部分。

*Node branch* 

这个branch预测了一个稀疏的nodes集合，它们是潜在的category-specific keypoints，但是并没有被排序。我们将其表示为$$X_i = \lbrace x_{i1}, x_{i2}, \cdots, x_{iN} \rbrace$$，其中$$x_{ij} \in R^3, j \in {1,2, \cdots, N}$$。一开始，作者利用Farthest Point Sampling (FPS)算法来从输入的point clouds里采样$$N$$个node，再使用在[So-net: Self-organizing network for point cloud analysis](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_SO-Net_Self-Organizing_Network_CVPR_2018_paper.pdf)和[USIP: Unsupervised Stable Interest Point Detection From 3D Point Clouds](https://openaccess.thecvf.com/content_ICCV_2019/html/Li_USIP_Unsupervised_Stable_Interest_Point_Detection_From_3D_Point_Clouds_ICCV_2019_paper.html)提到的point-to-node grouping来对每个node选取一个包含了若干points的local neighborhood（实际上所有的points都被唯一的分给了某一个node），这样就创建了$$N$$个clusters。point cloud $$S_i$$里的每个点都和上述这些nodes里的某一个建立关联。这个branch有着两个像PointNet的networks，然后再接了一个KNN groupoing层，再之后再用MLP来输出nodes。

*Pose and coefficients branch* 

我们利用这个branch来预测$$R_i$$和$$c_i$$。对于$$R_i$$，我们可以使用一个旋转角来表示它。这个branch包含一个MLP用来预测这些参数。这个branch的输出size对于公式3和公式4描述的两种情况是不同的，后者的输出size翻倍。


*Additional learnable parameters* 

公式3或者公式4里未知的其它的量对于一个object类别$$\mathcal{C}$$来说是常量。这些量不需要对于每个instance都进行预测。我们选择将其作为网络参数来优化。这些量是shape basis $$B_{\mathcal{C}} \in R^{3N \times K}$$和对称平面的法向量$$n_{\mathcal{C}} \in R^{3}$$。我们发现shape basis的$$K$$的选取最好是在5到10之间。实际上所生成的keypoints对于$$K$$值的选取并不敏感，比较大的$$K$$就会导致比较稀疏的shape coefficient $$c_i$$。我们也可以用其它的量来代替对称平面的法向量，比如说Euler角。

在inference的时候，我们使用Non-Maximal Suppression来获得最终的$$N^{'}$$个keypoints。我们对于某个类别的object的不同instances都输出同样数量的keypoints，因为它们具有同一个geometric model。


**Training Losses**

为了符合我们在section1里所定义的category-specific keypoints所需要满足的要求，我们按照如下方式来定义loss functions。

*Chamfer loss with symmetry and non-rigidity* 

公式1表明这个网络$$\Pi_{\mathcal{C}}$$可以被最小化node predictions $$X_i$$和deformation function $$P_i = \Phi_C(R_i, c_i; B_{\mathcal{C}},n_{\mathcal{C}})$$之间的$$l_2$$loss来实现。但是正如[USIP: Unsupervised Stable Interest Point Detection From 3D Point Clouds](https://openaccess.thecvf.com/content_ICCV_2019/html/Li_USIP_Unsupervised_Stable_Interest_Point_Detection_From_3D_Point_Clouds_ICCV_2019_paper.html)所说，因为网络不具备预测keypoints顺序的能力，所以$$l_2$$ loss并不会收敛。但Chamfer loss可以收敛，这个loss会先对于$$X_i$$里的每个$$x_{ij}$$找到其在$$P_i$$里最近的那个点$$p_{ik}$$，然后再最小化这个距离，反之亦然。

$$\mathcal L_{chf} = \Sigma_{k=1}^N \mathop min_{p_{i,j} \in P_i} \left | \left | x_{ik} - p_{i,j} \right | \right | ^2_2 + \Sigma_{j=1}^N \mathop min_{x_{i,k} \in X_i} \left | \left | x_{ik} - p_{i,j} \right | \right | ^2_2 \tag{5}$$

公式5里的Chamfer loss保证所学习到的keypoints满足category-specific property里的generalisability——因为它们是对于这个category定义的共用的shape basis的线性组合。为了对对称性建模，公式3和公式4可以用于计算公式5里的$$P_i$$。


*Coverage and inclusively loss* 

公式5表示的Chamfer loss并不能保证keypoints符合object shape。但是我们可以加入如下限定：a) coverage loss：keypoints能够覆盖整个category shape。b) inclusivity loss：keypoints和point cloud离得不远。coverage loss可以通过计算nodes $$X_i$$和point cloud $$S_i$$的volume之间的Huber loss来获得，需要使用到singular values。但我们这里就直接用3D bounding box来算了，因为简单。

$$\mathcal L_{cov} = \left | \left | vol(X_i) - vol(S_i) \right| \right | \tag{6}$$

inclusivity loss用只有一侧的Chamfer loss就可以表示：

$$\mathcal L_{inc} = \Sigma_{k=1}^N \mathop min_{s_{i,j} \in S_i} \left | \left | x_{ik} - s_{ij} \right | \right | ^2_2 \tag{7}$$

>实际上我们发现，这篇文章里的网络结构分为两部分，一部分用来生成nodes，$$X_i$$，表示的是keypoints。而另一部分利用上述的$$P_i = \Phi_C(R_i, c_i; B_{\mathcal{C}},n_{\mathcal{C}})$$来获取keypoints。之后再将这两部分比较从而来获取非监督的signal。而生成nodes的那部分网络实际上和[USIP: Unsupervised Stable Interest Point Detection From 3D Point Clouds](https://openaccess.thecvf.com/content_ICCV_2019/html/Li_USIP_Unsupervised_Stable_Interest_Point_Detection_From_3D_Point_Clouds_ICCV_2019_paper.html)里的网络结构意义，但是在USIP这篇文章里，其输入是一个point cloud和将这个point cloud经过某种已知的transformation所得到的point cloud构成的一个pair，所以它的非监督label就是这两个point cloud所学习到的keypoints要相对应，应该是处于同一位置。这就是为什么这篇文章不需要使用Siamese结构，也不需要输入一对point cloud的原因，因为它利用shape basis来对keypoints重新提出了一种学习方法。

**5. Experiments**

**6. Conclusion**

这篇文章探究了如何在misaligned 3D point clouds上自动检测到那些能保证对于不同物体之间的shape variation以及同一个物体的deformation都consistent的keypoints的方法。我们发现这个问题可以用一个非监督学习的方式通过用对称的线性basis shapes来表示keypoints这个方式被解决。而且，这些被学习到的category-specific keypoints在不同的输入之间具有1对1的对应关系，并且是semantic consistent的。基于所学习到的keypoints的应用包括registration，generation，shape completion等。我们的实验表明我们的方法可以获得很高质量的keypoints，并且对于更复杂的deformation我们的方法也有潜力。未来的工作方向可以是通过非线性的方式来对更复杂的deformation进行建模（本文里的shape basis是线性组合的）。

### 9. [Skeleton Merger: An Unsupervised Aligned Keypoint Detector](https://openaccess.thecvf.com/content/CVPR2021/html/Shi_Skeleton_Merger_An_Unsupervised_Aligned_Keypoint_Detector_CVPR_2021_paper.html)

*CVPR 2021*

### 3. [USIP: Unsupervised Stable Interest Point Detection from 3D Point Clouds](https://openaccess.thecvf.com/content_ICCV_2019/html/Li_USIP_Unsupervised_Stable_Interest_Point_Detection_From_3D_Point_Clouds_ICCV_2019_paper.html)

[Page](https://github.com/lijx10/USIP)

*Jiaxin Li, Gim Hee Lee*

*ICCV 2019*

3D interest point或者keypoint detection是考虑如何在经过任意的$$SE(3)$$ transformation之后的3D point clouds上找到稳定的并且repeatable的点。

尽管对于2D images来说已经有了很多成功的hand-crafted detectors，但对于3D point clouds来说成功的hand-crafted detectors就几乎没有。这个差异很大程度上是因为对于2D图片来说，我们有具有丰富信息的RGB channels，而对于3D point clouds来说只有点的位置的信息，这对于设计能提取有效信息的hand-crafted keypoints的算法来说是很难的。而且3D point clouds如果在经过任意的transformations之后（也就是说在不同的coordinate frame里），这个难度就更大了。

基于deep learning的3D keypoint detectors很少（实际上现在只有一篇[3DFeat-Net: Weakly Supervised Local 3D Features for Point Cloud Registration](https://openaccess.thecvf.com/content_ECCV_2018/papers/Zi_Jian_Yew_3DFeat-Net_Weakly_Supervised_ECCV_2018_paper.pdf)），但是3D keypoint descriptors的文章却越来越多（[PPF-FoldNet: Unsupervised Learning of Rotation Invariant 3D Local Descriptors]([https://arxiv.org/pdf/1808.10322.pdf](https://openaccess.thecvf.com/content_ECCV_2018/papers/Tolga_Birdal_PPF-FoldNet_Unsupervised_Learning_ECCV_2018_paper.pdf))，[PPFNet: Global Context Aware Local Features for Robust 3D Point Matching](https://openaccess.thecvf.com/content_cvpr_2018/papers/Deng_PPFNet_Global_Context_CVPR_2018_paper.pdf)，[Learning Compact Geometric Features](https://openaccess.thecvf.com/content_ICCV_2017/papers/Khoury_Learning_Compact_Geometric_ICCV_2017_paper.pdf)，[3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zeng_3DMatch_Learning_Local_CVPR_2017_paper.pdf)）（detector仅仅是为了找到keypoint的coordinate，其不关心也不知道keypoints的features或者它和其他数据之间的关系，而keypoint descriptors侧重于获取keypoints的features）。

这篇文章提出USIP detector：一个基于deep learning的非监督的稳定的keypoint detector，其可以在不需要任何监督数据情况下对于做了任意transformation的3D point clouds检测到高度repeatable和精确定位的keypoints。为了达到这个目的，他们提出了一个Feature Proposal Network（FPN）用来从一个输入的3D point cloud上输出一个集合的keypoints以及它们每个点的不确定性。FPN使用的是估计position的方法（也就是说keypoint可能并不是point clouds里任何一个点）来改进了keypoint localization，因为其它的方法（[3DFeat-Net: Weakly Supervised Local 3D Features for Point Cloud Registration](https://openaccess.thecvf.com/content_ECCV_2018/papers/Zi_Jian_Yew_3DFeat-Net_Weakly_Supervised_ECCV_2018_paper.pdf)）使用的是在point cloud里选取keypoints，所以会造成误差。在训练过程中，其使用随机生成的$$SE(3)$$ transformation来处理每个point clouds，从而得到point clouds pairs，用作FPN的输入。他们利用probabilistic chamfer loss来最小化训练数据point cloud pairs的keypoints之间的距离。另外他们们还引入了point-to-point loss来迫使keypoints足够靠近point cloud。某些定性的结果如fig 1所示。

![Mfdaf]({{ '/assets/images/USIP-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*fig 1. USIP detector在四个数据集上找到的keypoints的例子：(a) ModelNet40，object model (b) Oxford RobotCar, outdoor SICK LiDAR (c) KITTI (在Oxford上训练的）,outdoor Velodyne LiDAR。

这篇文章的主要贡献：
* USIP detector完全是非监督的，因此避免了获取那些本来就不可能能有的监督信息（3D point cloud上的keypoints的位置）
* 分析了USIP detector可能会不管用的情况，并且提出了避免这些问题出现的解决办法
* FPN通过估计keypoint的位置而不是从point cloud里选择现有的point改进了keypoint localization
* 提出probabilistic chamfer loss和point-to-point loss来找到高度repeatable和准确定位的keypoints
* 在训练时候使用了随机生成的transformation作用在point clouds上，这使得网络对于rotation来说有很好的效果


**Related Work**

和最近很成功的基于deep learning的3D keypoint descriptors不同，大多数现有的3D keypoint detectors仍然还是hand-crafted的。Local Surface Patches（LSP）和Shape Index（SI）基于一个point的最大和最小principal曲率，如果一个point在预先定义的某个领域内是全局极值点，那么就认为这个点是一个keypoint。Intrinsic Shape Signatures（ISS）和KeyPoint Quality（KPG）选取那些沿着每个主轴都有很大的变化的那些点作为keypoints。MeshDoG和Salient Points（SP）利用类似于SIFT的Difference of Gaussian operator构建了一个曲率的scale space，有着局部最大值的那些点被选为keypoints。Laplace-Beltrami Scale-space（LBSS）通过对每个点使用一个Laplace-Beltrami operator来选取keypoints。更近的工作，LORAX提出将point set投射到一个depth map上，再利用PCA来选择那些具有普遍geometric characteristics的keypoints。所有的hand-crafted方法都是靠着point clouds的局部的几何特征来选择keypoints的。因此，这些detectors的表现会因为扰动，比如说noise，density variations或者transformations，而效果变差。

在这篇文章之前，仅有的基于deep learning的3D keypoint detector就是weakly supervised 3DFeatNet（[3DFeat-Net: Weakly Supervised Local 3D Features for Point Cloud Registration](https://openaccess.thecvf.com/content_ECCV_2018/papers/Zi_Jian_Yew_3DFeat-Net_Weakly_Supervised_ECCV_2018_paper.pdf)），其利用GPS/INS标注的point clouds来训练。然而，3DFeat-Net的训练很大程度上利用Siamese结构来学习具有区分性的descriptors。其并没有确保keypoint detection具有良好的效果。相比较而言，USIP被设计为可以得到高度repeatable和定位准确的keypoints。更进一步的是，USIP的方法是完全非监督的，并不依赖于任何有标注的数据集。


**Our USIP Detector**

Fig 2的(a)解释了USIP detector的pipeline。将一个point cloud记为$$X = \left[ X_0, \cdots, X_N \right] \in R^{3 \times N}$$。一个集合的transformation matrices $$\lbrace T_1, \cdots, T_L \rbrace$$，其中$$T_l \in SE(3)$$是随机生成的，其应用到point cloud $$X$$上就得到了$$L$$对训练inputs，记为$$\lbrace \lbrace X, \tilde{X_1} \rbrace, \cdots, \lbrace X, \tilde{X_L} \rbrace \rbrace$$，其中$$\tilde{X_l} = T_l X \in R^{3 \times N}$$。我们丢掉$$l$$下标，将训练input pair和它们对应的transformation matrix构成的三元组记为$$\lbrace X, \tilde{X}, T \rbrace$$。在训练期间，$$X$$和$$\tilde{X}$$分别喂给FPN，都会输出$$M$$个proposal keypoints和它们对应的不确定性，分别记为$$\lbrace Q = \left[ Q_1, \cdots, Q_M \right], \Sigma = \left[ \sigma_1, \cdots, \sigma_M \right]^T \rbrace$$，和$$\lbrace \tilde{Q} = \left[ \tilde{Q_1}, \cdots, \tilde{Q_M} \right], \tilde{\Sigma} = \left[ \tilde{\sigma_1}, \cdots, \tilde{\sigma_M} \right]^T \rbrace$$。其中$$Q_m \in R^{3}$$，$$\tilde{Q_m} \in R^3$$，$$\sigma_m \in R^{+}$$，$$\tilde{\sigma_m} \in R^{+}$$。为了提升keypoint localization的精度，不需要$$Q_m \in Q$$是$$X$$里的任何点。对于$$\tilde{Q_m}$$也是一样。

对于$$\tilde{Q}$$，我们再计算$$Q^{'} = T^{-1}\tilde{Q} \in R^{3 \times M}$$，而$$Q^{'}$$应该和$$Q$$很像。这里做一个假设，也就是每个keypoint的不确定性，在经过$$T^{-1}$$操作后不会改变，也就是说$$\Sigma^{'} = \tilde{\Sigma}$$。从经历过任意transformation的3D point clouds里检测具有高度repeatble特性以及定位准确的keypoints的任务就可以通过最小化$$Q$$和$$Q^{'}$$来实现。为了实现这个目标，文章提出loss function：$$\mathcal{L} = \mathcal{L_c} + \lambda \mathcal{L_p}$$，其中$$\mathcal{L_c}$$是probabilistic chamfer loss，用于最小化$$Q$$和$$Q^{'}$$里的对应的keypoints的probabilistic距离。$$\mathcal{L_p}$$是point-to-point loss，用来最小化所得到的keypoints和距离它最近的point clouds里的点的距离。$$\lambda$$是个超参数。

*Probabilistic Chamfer Loss*

一个最小化$$Q$$和$$Q^{'}$$之间距离的方法就是使用Chamfer loss：

$$ \Sigma_{i=1}^M min_{Q_j^{'} \in Q^{'}} \left | \left | Q_i - Q_j^{'} \right | \right | + \Sigma_{j=1}^M min_{Q_i \in Q} \left | \left | Q_i - Q_j^{'} \right | \right |  \tag{1}$$

公式1最小化一个point cloud里的点和其在另一个point cloud里最近的那个点之间的距离。然而，$$M$$个proposals并不是等重要的。如果$$Q$$里面的点$$Q_i$$的位置并不好，如果还按照上述的方式来计算，那么其也会导致$$Q_i^{'}$$的结果也不好。

为了解决上述这个问题，FPN同时也学习每个keypoint proposal的不确定性$$\Sigma$$和$$\Sigma^{'}$$，再计算一个probabilistic chamfer loss $$\mathcal{L_c}$$。对于$$Q_i$$和$$Q_j^{'}$$，$$i=1,\cdots,M$$，其上面定义的概率分布为：

$$p(d_{ij} | \sigma_{ij}) = \frac{1}{\sigma_{ij}} exp(-\frac{d_{ij}}{\sigma_{ij}}) \tag{2}$$

其中

$$\sigma_{ij} = \frac{\sigma_i + \sigma_j^{'}}{2}$$

$$d_{ij} = min_{Q_j^{'} \in Q^{'}} \left | \left | Q_i - Q_j^{'} \right | \right | \geq 0$$

$$p(d_{ij} | \sigma_{ij})$$

是一个合规的概率分布。$$d_{ij}$$越小，那么proposal keypoints $$Q_i$$和$$Q_j^{'}$$是高度repeatable以及定位准确的keypoints的概率就越高。

假设对于所有的$$d_{ij} \in D_{ij}$$，$$Q$$和$$Q^{"}$$之间的联合分布是：

$$p(D_{ij} | \Sigma_{ij}) = \Pi_{i=1}^M p(d_{ij} | \sigma_{ij}) \tag{3}$$

因为最近邻的选择不同，所以说$$d_{ij} \neq d_{ji}$$，且$$\sigma_{ij} \neq \sigma_{ji}$$。

$$Q^{"}$$和$$Q$$之间的联合分布是：

$$p(D_{ji} | \Sigma_{ji}) = \Pi_{j=1}^M p(d_{ji} | \sigma_{ji}) \tag{4}$$

其中

$$\sigma_{ji} = \frac{\sigma_i + \sigma_j^{'}}{2}$$

$$d_{ji} = min_{Q_i \in Q} \left | \left | Q_i - Q_j^{'} \right | \right | \geq 0$$

最后概率chamfer loss就定义为：

$$\mathcal{L_c} = \Sigma_{i=1}^M -lnp(d_{ij} | \sigma_{ij}) + \Sigma_{j=1}^M -lnp(d_{ji} | \sigma_{ji})$$

$$ = \Sigma_{i=1}^M (ln \sigma_{ij} + \frac{d_ij}{\sigma_{ij}}) + \Sigma_{j=1}^M (ln\sigma_{ji} + \frac{d_{ji}}{\sigma_{ji}}) \tag{5}$$

通过计算公式2关于$$\sigma_{ij}$$的导数来分析其的物理含义：

$$\frac{\partial p(d_{ij} | \sigma_{ij})}{\partial \sigma_{ij}} = 0$$

可得$$\sigma_{ij} = d_{ij}$$。

这说明，给定一个$$d_{ij} >0$$，$$\sigma_{ij} = d_{ij}$$的时候，上述概率取到最大值。假设我们有三个proposal keypoints，$$(Q_i,Q_j^{'},Q_k^{'})$$，其中$$d_{ij}$$和$$d_{ki}$$是两对keypoints pairs的最近邻距离。当$$d_{ij} \longrightarrow 0$$而且$$d_{kj}$$很大的时候，我们需要$$\sigma_k^{'}$$的值很大。这也就是说，$$\lbrace Q_i, Q_j^{'} \rbrace$$是高度repeatable且精确定位的keypoints，而$$Q_k^{'}$$不是。因此，$$\sigma_k^{'}$$比较大，说明我们的概率chamfer loss是定义正确的。


*Point-to-point loss*

为了减小keypoints localization的错误，不需要keypoints是point cloud里的任何一个点。但是这可能会让keypoints离point cloud太远。通过引入一个loss function $$\mathcal{L_p}$$来解决这个问题。

$$\mathcal{L_{point}} = \Sigma_{i=1}^M  min_{X_j \in X} \left | \left | Q_i - X_j \right | \right | + \Sigma_{i=1}^{M}  min_{\tilde{X_j} \in \tilde{X}} 
\left | \left | \tilde{Q_i } - \tilde{X_j} \right | \right | $$

其中$$X_j \in X$$是$$Q_i$$在point clouds里最近的点。


**Feature Proposal Network**

FPN的结构如fig 2 (b)所示。

**Step1** 从给定的输入point cloud $$X \in R^{3 \times N}$$使用Farthest Point Sampling采样$$M$$个nodes，记为$$S = \left[ S_1, \cdots, S_M \right] \in R^{3 \times M}$$。之后才采用[So-net: Self-organizing network for point cloud analysis](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_SO-Net_Self-Organizing_Network_CVPR_2018_paper.pdf)里的point-to-node grouping的方法（也就是每个points，将其归属于距离它最近的那个nodes）来为每个$$S_m \in S$$构造一个points构成的neighborhood，从而我们获得了

$$\lbrace \lbrace X_1^1 |S_1, \cdots, X_1^{K_1} |S_1 \rbrace, \cdots, \lbrace X_M^1 |S_M, \cdots, X_M^{K_M} |S_M \rbrace \rbrace$$

其中$$K_1, \cdots, K_M$$表示每个nodes的neighborhood points的数量。

>Farthest Point Sampling的解释可以看[这篇博客](https://jskhu.github.io/fps/3d/object/detection/2020/09/20/farthest-point-sampling.html)

>point-to-node方法比node-to-point KNN或者radius-based ball-search要好在以下两个方面：(1) $$X$$里的每个点都唯一的归于了某个node的neighborhood，而另外两个方法可能会有些点没有归属；(2) point-to-node grouping方法对于不同的scale以及point density都可以适应，而KNN search受到density变化以及ball-search受到scale变化的影响很大。

**Step2** 为了使得FPN是translation equivariant的，我们将每个node构成的neighborhood都进行归一化，记归一化之后的结果为

$$\lbrace \hat X_m^1 | S_m, \cdots, \hat X_m^{K_m} | S_m \rbrace$$

其中$$\hat X_m^{k}  = X_m^{k} - S_m$$，$$1 \leq k \leq K_m$$。

**Step3** 将经过上述操作之后的point cloud $$X \in R^{3 \times N}$$（也就是将$$X$$按照选择出来的$$M$$个nodes分为$$M$$个clusters，然后再将每个node对应的cluster除了node以外的点都减去node的坐标实现归一化）被喂给如fig 2 (b)里所示的一个类似PointNet（[PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)）的网络结构，这样$$X$$里的每个点都有一个feature，再将每个cluster里的points的features综合起来获取一个和$$S_m$$相关的局部feature向量$$G_m$$，从而我们就得到了

$$\lbrace G_1 |S_1, \cdots, G_M |S_M \rbrace$$

**Step4** 再之后，对于每一个$$(G_m, S_m)$$，我们使用KNN来找到$$G_m$$的K个最近的向量（是在上述nodes的feature空间里找，而不是在所有的$$X$$里的points经过了之前的PointNet之后所得到的feature空间里找，也就是在

$$\lbrace G_1 | S_1, \cdots, G_M |S_M \rbrace$$

里利用KNN来搜索）。这也就是，利用KNN grouping layer应用在在局部feature向量集合里

$$\lbrace G_1 | S_1, \cdots, G_M |S_M \rbrace$$

来获取层次化的信息综合。对于每个$$(G_m, S_m)$$来说，将其找到的K个最近的node feature向量记为

$$\lbrace (G_m^1|S_m^1)|S_m, \cdots, (G_m^K|S_m^K)|S_m \rbrace$$

这些KNN的局部feature向量再次将$$S_m$$的坐标减去，从而获得了和位置无关的局部feature向量，记为

$$\lbrace (G_m^1|\hat S_m^1)|S_m, \cdots, (G_m^K|\hat S_m^K)|S_m \rbrace$$

其中$$\hat S_m^k = S_m^k - S_m$$，$$1 \leq k \leq K$$

**Step5** 之后再将其送入另一个network来获得feature向量$$\lbrace H_1, \cdots, H_M \rbrace$$。

**Step6** 再利用一个MLP来预测$$M$$个proposal keypoints，

$$\lbrace \hat Q_1|S_1, \cdots, \hat Q_M|S_M \rbrace$$

其中$$\hat Q_m \in R^3$$，以及预测点的不确定性$$\lbrace \sigma_1, \cdots, \sigma_M \rbrace$$。

**Step7** 最后，我们再将$$\hat Q_m$$还原回去，$$Q_m = \hat Q_m + S_m$$来获得最终的keypoint proposals $$\lbrace Q_1, \cdots, Q_M \rbrace$$。

我们需要注意，每个keypoint感受野的大小取决于proposal的数量$$M$$以及KNN算法里的$$K$$的大小。其也决定了每个keypoints feature的描述细节的能力。大的感受野能够使得keypoint具有描述大范围细节的能力。

![dagarf]({{ '/assets/images/USIP-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*fig 2. (a) USIP detector的训练pipeline (b) Feature Proposal Network (FPN)的结构。


**Degeneracy Analysis**

我们将FPN记为：$$f(Y): Y \longrightarrow R^3 \times N$$，其中$$Y = \lbrace Y_1, \cdots, Y_N \rbrace \in R^{3 \times N}$$表示的是网络的输入，也就是point clouds。我们进一步将一个transformation matrix记为$$T \in SE(3)$$，而$$R \in SO(3)$$和$$t \in R^3$$分别表示$$T$$的rotation matrix和translation vector。从而对于经过$$T$$变换后的point clouds，我们记为$$Y^{'}$$，我们就有$$Y^{'} = RY+t$$。如果对于任意的$$R$$和$$t$$，都有$$f(Y^{'}) = Rf(Y)+t$$，那就认为网络是degenerate的。

*Lemma 1* 如果$$f(.)$$输出的是point cloud的centroid，也就是说$$f(Y) = \frac{1}{N} \Sigma_n Y_n$$，那么$$f(Y^{'})$$恒等于$$Rf(Y)+t$$。

*Lemma 2* 如果$$f(.)$$是translational equivariant的，i.e., $$f(x)t = f(xt)$$，而且先计算$$Y$$里所有的点的covariance matrix，再利用SVD解出U，即可得到principal axis的方向（也就是U的三个列表示的向量的方向），如果$$f(Y)$$得出的keypoints都在principal axis上，那么$$f(Y^{'})$$恒等于$$Rf(Y)+t$$。


上述两个lemma可以看出，当点在一些特殊的位置上，比如说point cloud的中心，或者说主轴上（比如说对称物体的对称轴），那么就会导致degenerate的结果。而实际上如果想让网络输出这样的结果，那我们需要网络的每个keypoint的感受野很大才行，因为不管是中心，还是主轴，这都是整个point cloud的性质，如果说keypoint只能注意到局部的特征，那么就不会出现上述的情况，也就避免了degenerate情况的发生。在这个FPN里，则是通过设置$$M$$和$$K$$值来控制感受野。较小的$$M$$和较大的$$K$$都会使得感受野变得很大，从而FPN就会degenerate。

fig 3显示了几个不同的$$K$$的值导致的degenerate的情况，在这三种情况下都有$$M=64$$。实验还表明，对于FPN来说，更加规则的物体，也就是中心或者对称轴更加好定义的物体，其出现degenerate的情况就越容易。

![degenerate]({{ '/assets/images/USIP-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*fig 3. (a) K=9，没有degenerate情况 (b) K=24，出现了keypoints都在主对称轴上的情况 (c) K=64，keypoints全部都跑到中心点去了。


### [SK-Net: Deep Learning on Point Cloud via End-to-End Discovery of Spatial Keypoints](https://arxiv.org/pdf/2003.14014.pdf)

*AAAI 2020*


### [6-DoF Object Pose from Semantic Keypoints](https://arxiv.org/pdf/1703.04670.pdf?ref=https://githubhelp.com)

*Georgios Pavlakos, Xiaowei Zhou, Aaron Chan, Konstantinos G. Derpanis, Kostas Daniilidis*

*ICRA 2017*

这篇文章解决的是从单张image来估计一个object的6自由度姿态，也就是3维空间内的translation和rotation。

我们的方法将描述appearance的statistical model和object的3D shape layout结合起来用于pose estimation。这个方法包括两个stages，首先通过一个集合的2D semantic keypoints来推出3D object投射到2D图片上的shape是什么样的，然后利用这些keypoints来估计3D object的pose。整个过程在fig 1里被详细描述。在第一个stage，我们使用一个CNN来预测一个集合的semantic keypoints。这里，我们利用了CNN的利用层次化设计能够获得很大感受野的信息的特性来获得semantic keypoints。在第二个stage，我们利用这些semantic keypoints来显式推出intra-class shape variability以及用camera model描述的camera pose。pose estimates通过最大化deformable model和2D semantic keypoints之间的geometric consistency来实现。

![Pipeline]({{ '/assets/images/6DOF-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1 方法的pipeline。给定一个object的一张RGB image (a)，我们使用stacked hourglass结构的CNN来预测一系列keypoints。这一步的结果是一系列的heatmaps，在(b)里合起来用作可视化。在(c)里，绿色的点表示我们网络输出的keypoints的位置，蓝色的点表示ground truth keypoints的位置。(d)显示了object的6自由度pose.*

这篇文章的贡献在于：

* 提出了一个高效的方法，将通过一个CNN学习到的semantic keypoints和一个deformable shape model结合起来用来估计一个object的3D pose信息（也就是是6自由度pose，3D translation和rotation）。
* 通过实验证明我们的方法在很复杂的环境中也能有效的学习到准确的6自由度pose。

**Method**

我们的pipeline包括了object detection，keypoint localization和pose optimization。因为object detection已经被研究的很好了，比如Fast R-CNN，所以我们利用现成的方法获取object的detection box，主要聚焦于后面两部分。

*Keypoint localization*

keypoint localization所用的CNN结构是[Stacked Hourglass Network for Human Pose Estimation](https://arxiv.org/pdf/1603.06937.pdf)。

网络结构如fig 2所示，网络的输入是一个image，而输出是一系列的heatmaps，每个heatmap对应一个keypoint，而heatmap每个点的值表示该keypoint出现在这个点的概率。我们有真实的keypoints location作为训练的监督数据，将每个真实的keypoint location也做成heatmap，其是以真实的location为中心，以1为标准差的高斯分布。再将这些真实的heatmaps与网络输出的heatmaps计算$$l_2$$ loss来训练网络。

![Model Structure]({{ '/assets/images/6DOF-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2 stacked hourglass结构示意图。我们这里使用了两个hourglass modules叠起来。hourglass的结构设计可以允许在每个hourglass的前半部分bottom-up处理（从高分辨率到低分辨率），再在每个hourglass的后半部分实现top-down处理（从低分辨率到高分辨率）。这里将两个hourglass叠起来使用，而训练的时候可以同时对第一个hourglass的输出以及第二个hourglass的输出同时进行监督，从而提供更丰富的监督信号。*

*Pose Optimization*

我们对于每个object类别，使用3D CAD构建了具有标注keypoints的deformable shape model。更加准确地说，一个3D object model上的$$p$$个keypoint locations被记为$$S \in R^{3 \times p}$$，以及$$S = B_0 + \Sigma_{i=1}^k c_i B_i$$，其中$$B_0$$是这个3D model的平均shape，$$B_1, \cdots, B_k$$是通过PCA计算出来的几种可能的shape variability。

给定图片中所检测到的2D keypoints，记为$$W \in R^{2 \times p}$$，我们的目标是要估计object和camera frame之间的rotation $$R \in SO(3)$$以及translation $$T \in R^3$$，和shape deformation的参数$$c = \left[c_1, \cdots, c_k\right]^T$$。

上述可以综合为以下的optimization问题：

$$ min_{\theta} \frac{1}{2} \left| \left| \xi (\theta) D^{\frac{1}{2}} \right| \right| + \frac{\lambda}{2} \left| \left| c \right| \right| \tag{1}$$

为了将2D keypoint预测的不确定性考虑进来，我们定义了一个对角权重矩阵$$D \in R^{p \times p}$$：

$$\begin{pmatrix}
d_1 & 0 & \cdots & 0 \\
0 & d_2 & \cdots & 0 \\
\vdots & \vdots & \cdots & \vdots \\
0 & 0 & \cdots & d_p
\end{pmatrix}$$

其中$$d_i$$表示keypoint $$i$$的localization的确信值。我们直接将$$d_i$$的值设置为每个keypoint的heatmap的峰值。

前面的$$\xi (\theta)$$是fitting residual，衡量hourglass网络得出的2D keypoints和3D keypoints之间的差异。我们考虑两种相机模型。

(1) *Weak Perspective Model*

如果camera的intrinsic parameters是未知的，就采用weak perspective model，当camera距离object较远的时候也是对full perspective model的一个较好的近似。在这种情况下：

$$\xi (\theta) = W - s\bar R \left( B_0 + \Sigma_{i=1}^k c_i B_i \right) - \bar T 1^T $$

其中$$s$$是一个scalar，$$\bar R \in R^{2 \times 3}$$和$$\bar T \in R^2$$分别表示$$R$$和$$T$$的前两行，从而我们未知的参数就是$$\theta = \lbrace s, c, \bar R, \bar T \rbrace$$。

(2) *Full Perspective Model*

如果我们知道camera intrinsic parameters，那么我们就可以构建full perspective camera model，从而：

$$ \xi (\theta) = \tilde W Z - R \left( B_0 + \Sigma_{i=1}^k c_i B_i \right) -T 1^T$$

其中$$\tilde W \in R^{3 \times p}$$表示的是2D keypoints的normalized homogeneous coordinates，$$Z$$是一个对角矩阵：

$$\begin{pmatrix}
z_1 & 0 & \cdots & 0 \\
0 & z_2 & \cdots & 0 \\
\vdots & \vdots & \cdots & \vdots \\
0 & 0 & \cdots & z_p
\end{pmatrix}$$

其中$$z_i$$表示3D keypoint $$i$$距离camera plane的距离，也就是depth。从而未知的参数就是$$\theta = \lbrace Z, c, R, T \rbrace$$。


### 5. [UKPGAN: A General Self-Supervised Keypoint Detector](https://arxiv.org/pdf/2011.11974.pdf)

[code](https://github.com/qq456cvb/UKPGAN)

*Yang You, Wenhai Liu, Ynajie Ze, Yong-Lu Li, Weiming Wang, Cewu Lu*

*CVPR 2022*


传统的hand-crafted 3D keypoint detectors包括[Harris 3D: a robust extension of the Harris operator for interest point detection on 3D meshes](http://ivan-sipiran.com/papers/SB11b.pdf)，[A concise and provably informative multi‐scale signature based on heat diffusion](http://ki-www.cvl.iis.u-tokyo.ac.jp/class2013/2013w/paper/correspondingAndRegistration/02_Sun.pdf)，[Sparse points matching by combining 3D mesh saliency with statistical descriptors](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.409.7406&rep=rep1&type=pdf)，[Mesh saliency](https://web.archive.org/web/20170829211638id_/http://www.cs.princeton.edu/courses/archive/fall10/cos526/papers/lee05.pdf)，[Intrinsic shape signatures: A shape descriptor for 3d object recognition](https://www.researchgate.net/profile/Yu-Zhong-23/publication/224135303_Intrinsic_shape_signatures_A_shape_descriptor_for_3D_object_recognition/links/00b4952b8550b1c21c000000/Intrinsic-shape-signatures-A-shape-descriptor-for-3D-object-recognition.pdf)，[Volumetric image registration from invariant keypoints](https://web.stanford.edu/group/rubinlab/pubs/Rister-2017a.pdf)和[Scale-dependent 3D geometric features](https://www.researchgate.net/profile/Ko-Nishino/publication/224297791_Scale-Dependent_3D_Geometric_Features/links/5591f49d08ae1e1f9bb00e4d/Scale-Dependent-3D-Geometric-Features.pdf)。这些方法都过于依赖人为定义的参数，仅仅考虑local geometric information而没有global semantic information，和现在的基于学习的方法效果不能比，和真人更是差得远。

最近有一些基于学习的3D keypoint detectors被提了出来，比如[D3Feat: Joint Learning of Dense Detection and Description of 3D Local Features](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bai_D3Feat_Joint_Learning_of_Dense_Detection_and_Description_of_3D_CVPR_2020_paper.pdf)和[USIP: Unsupervised Stable Interest Point Detection from 3D Point Clouds](https://openaccess.thecvf.com/content_ICCV_2019/html/Li_USIP_Unsupervised_Stable_Interest_Point_Detection_From_3D_Point_Clouds_ICCV_2019_paper.html)。USIP从分割之后的局部clusters里学习到keypoint locations，而且利用了一个probabilistic chamfer loss。其使用了farthest point sampling的方法而且会产生并不和任何point cloud中的点重合的keypoint。D3Feat对于point cloud里的每个点都计算saliency值以及descriptors。USIP和D3Feat都是通过解决预测两个输入的point clouds之间的rotations这样一个代理任务来预测3D keypoints的位置。它们在训练中都需要3D point cloud，而且对于输出的keypoints并没有过多的控制。

这篇文章采用了完全不同的方式来获取3D keypoints，叫做unsupervised keypoint GANeration (UKPGAN)。对于detector network，这个方法使用了一个keypoint saliency distribution，并且使用了一个新的GAN loss来控制这个distribution的稀疏性。之后，为了使得学到的keypoints更加semantic，文章使用了一个salient information distillation的方法从这些检测到的稀疏的keypoints来重构原输入的keypoint cloud，从而构成了一个encoder-decoder的结构。

这篇文章提出的方法可以被看成一个information compression方法，也就是使用最少量的keypoints来保持object的point cloud里最多的信息。这个方法背后的道理很简单有效：我们应该可以从稀疏的keypoints上重构一个object的结构。

和之前的方法相比，UKPGAN有以下几个优势：1) 不需要任何数据增强操作，通过先估计一个local reference frame，这个detector就可以做到rotation invariant，而且keypoint的descriptors也是rotation invariant的。2) 所检测到的keypoints是类内高度consistent的，对于rigid和non-rigid objects都是。3) 我们的模型是在干净的数据集上训练的（比如ModelNet），推广到真实世界的point cloud上仍然可行。


**Related work**

hanf-crafted 3D keypoint detectors前面已经列举了。重点关注learning-based keypoint detectors，对于2D来说，关注那些unsupervised的方法，对于3D来说，都要关注。

对于2D unsupervised keypoints detectors来说，[Unsupervised Learning of Object Landmarks through Conditional Image Generation](https://proceedings.neurips.cc/paper/2018/hash/1f36c15d6a3d18d52e8d493bc8187cb9-Abstract.html)通过将target image通过一个窄的bottleneck来提取object的geometry的方式来学到semantically有意义的keypoints。[Unsupervised Discovery of Object Landmarks as Structural Representations](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Unsupervised_Discovery_of_CVPR_2018_paper.html)使用的是autoencoder框架，并且利用channel-wise softmax来检测keypoints。[Discovery of Latent 3D Keypoints via End-to-end Geometric Reasoning](https://papers.nips.cc/paper/2018/hash/24146db4eb48c718b84cae0a0799dcfc-Abstract.html)通过要求multi-view consistency来从2D images里找到latent 3D keypoints。[End-to-end learning of keypoint detector and descriptor
for pose invariant 3D matching](https://openaccess.thecvf.com/content_cvpr_2018/papers/Georgakis_End-to-End_Learning_of_CVPR_2018_paper.pdf)使用一个Siamese结构再加上一个sampling层和一个score loss来在depth maps上检测keypoints。

在3D keypoints detectors领域，SyncSpecCNN方法[]()以及deep functional dictionaries方法[]()需要ground-truth的keypoints locations作为监督信号。对于unsupervised learning方法，USIP方法[]()利用将point cloud分割为clusters，再在clusters上学习到keypoints，并且结合了probablistic chamfer loss的方法来学习keypoints locations。D3Feat方法[]()为每个point cloud的点都计算一个saliency score和descriptor。USIP和D3Feat都是依赖于一个检测一对point cloud之间的rotation这样一个代理任务来学习keypoints的，其并没有多关心semantic information。而还存在另一条line of research：[Unsupervised Learning of Intrinsic Structural Representation Points](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Unsupervised_Learning_of_Intrinsic_Structural_Representation_Points_CVPR_2020_paper.pdf)，[Unsupervised Learning of Category-Specific Symmetric 3D Keypoints from Point Sets](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700545.pdf)和[KeypointDeformer: Unsupervised 3D Keypoint Discovery for Shape Control](https://openaccess.thecvf.com/content/CVPR2021/papers/Jakab_KeypointDeformer_Unsupervised_3D_Keypoint_Discovery_for_Shape_Control_CVPR_2021_paper.pdf)，它们对rigid transformation不稳定，而且并不能拓展到真实世界的数据上。


**Method**

*Overview*

给定一个point cloud集合

$$X = \lbrace x_n | x_n \in R^3, n = 1,2,\cdots,N \rbrace$$

这些$$x_n$$是从某个流形$$\mathcal M$$上采样得来的。我们想要获得一个子集$$\tilde X \subset X$$，这个子集就是keypoints的集合，

而

$$\left| \tilde X \right|$$

表示的就是keypoints的个数。

>所以这里还是从point cloud中来选择keypoint，和USIP的想法截然相反。

文章采用的是一个unsupervised encoder-decoder的结构。在encoder里，point cloud里的每个点都会被预测一个keypoint probability $$s$$。为了使得所检测到的keypoints是稀疏的，文章还使用了GAN-based keypoint稀疏性控制。decoder同时也是一个reconstruction network，我们在decoder里使用salient information distillation以一种非监督的方式来重构输入的point cloud。其直觉是，一个好的keypoints集合应该含有一个object的point cloud的独特的信息，从而有能力仅仅基于这些keypoints来重构原输入的point cloud。

整个方法的流程示意图如fig 1所示。

![Model Structure]({{ '/assets/images/UKPGAN-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1 keypoint和embedding生成的pipeline。我们先得到每个点的rotation invariant features，然后利用两个MLP分别输出每个点的keypoint probability和每个点的semantic embedding。GAN被用来控制keypoint稀疏性，salient information distillation被用来获取最显著的features。之后再利用一个decoder来基于学到的keypoints集合重构原始输入的point cloud。*

*Rotation Invariant Feature Extraction*

为了在rigid transformation的情况下仍然robust，我们对于每个点$$x$$的球形领域

$$\mathcal S = \lbrace x_i: \left| \left| x_i - x \right| \right| \leq r \rbrace$$

计算covariance eigendecomposition，从而生成一个Local Reference Frame (LRF)。然后在这个球形邻域内的点$$x_i \in \mathcal S$$对于计算的LRF，变换到它们的canonical position $$x_i^{'}$$。然后我们使用PerfectMatch（[The Perfect Match: 3D Point Cloud Matching with Smoothed Densities](https://openaccess.thecvf.com/content_CVPR_2019/papers/Gojcic_The_Perfect_Match_3D_Point_Cloud_Matching_With_Smoothed_Densities_CVPR_2019_paper.pdf)）里的方法将这些点离散化到一个Smoothed Density Value (SDV) grid，以$$x$$为中心，并且与LRF对齐。voxelization是基于高斯光滑kernel的。从而，我们对于每个点$$x$$就获得了一个voxelized descriptor $$\mathcal F(x) \in R^{W \times H \times D}$$。每个点的3D descriptors再组合起来，喂给3D convolution层。因为LRF，这一步可以让我们获得local rotation-invariant features，其让我们的方法对于rotation具有robustness。

*Dual Branches on Estimating Probabilities and Embeddings*

在上一步获取了每个点的rotation invariant feature之后，我们使用两个MLP来分别预测每个点的keypoint salient probability $$\Phi(x) \in \left[0, 1\right]$$和一个高维的embedding $$h(x) \in R^F$$，其会被用来做reconstruction。

(1) **$$\Phi(x)$$的稀疏性**

为了能将整个point cloud压缩到一个keypoints的小集合，$$\Phi(x)$$需要是稀疏的。而如何使得$$\Phi(x)$$稀疏呢？$$L_1$$ regularization可以实现稀疏性。但是它只能做到输出更多一点的0附近的值，却不能控制那些非0的值的大小（也就是说对于那些我们需要它是1，从而表示keypoints的点，这个值会比1要小很多）。为了输出具有辨识度的keypoints，并且压制住那些没有意义的points，我们希望$$\Phi(x)$$只取0和1附近的值。一个最直观的办法就是直接把这个point cloud上的distribution定义为本身就是在0和1附近聚集的distribution（也就是说这个distribution本身就是在0和1附近值很大，在其他位置值很小），然后迫使这个MLP的输出尽可能地去拟合这个distribution。我们采用Beta distribution来作为这样一个prior distribution。在Beta distribution里有两个参数，$$\alpha$$和$$\beta$$，分别控制1和0的聚集程度。

(2) **GAN-based Keypoint Sparsity Control**

一个最直观的控制稀疏性的方法就是计算上面的MLP输出的keypoint probabilities和beta distribution之间的KL divergence。但是因为我们并没有显式的输出$$\Phi(x)$$的参数，只是输出了每个point的keypoint probability，所以$$\Phi(x)$$和beta distribution之间的KL divergence的解析形式并不存在。我们利用一个adversarial loss来替代KL divergence来解决这个问题。

>但实际上，利用$$\Phi(x)$$的输出结果可以构建一个empirical distribution，也就是先将0到1的坐标轴均匀分割为若干份，然后统计落入每个区间内的点与所有的点的比例，作为这个区间的概率值，从而获得0到1范围内$$\Phi(x)$$的一个近似，区间分割的越小，近似的就越准确。之后我们就可以对这个近似的分布与beta distribution之间计算KL divergence了。

我们采用将我们的模型看作GAN来获取MLP生成的keypoint distribution $$\Phi(x)$$和beta distribution $$p(x)$$之间的一个loss。我们认为$$\Phi(x)$$生成假的keypoint distribution，其需要和我们的prior beta distribution很像。还需要一个discriminator网络$$D$$来判断distribution的真假。distriminator $$D$$的输入是一个point cloud经过MLP输出的所有可能的keypoint distribution构成的集合。在实际操作上，我们使用WGAN-GP而不是最初的GAN，因为这个更加的robust：

$$\mathcal L_{GAN} = min_{\Phi} max_{D} E_{\mathcal M} \left[D(\lbrace p(x) | x \in \mathcal{M} \rbrace)\right] - E_{\mathcal M} \left[D(\lbrace \Phi (x) | x \in \mathcal{M} \rbrace)\right] + \lambda (||\nabla D||-1)^2$$

>这篇文章里说，因为MLP输出的是$$\Phi(x)$$，是point cloud里每个点的keypoint probability，而没有显式的得出这个probability的参数，所以没办法和beta distribution计算KL divergence，从而采用的GAN的方法。但实际上，按照之前批注里的说法也是可以计算的。而且在看了代码之后发现，作者对于这里的实现是，利用一个sigmoid将每个点经过了若干层MLP之后的值压缩到0到1之间，从而代表每个点的keypoint probability，即$$\Phi(x)$$，对于每个点都有这样一个值，从而所有的点构成了一个大小为$$batch \times N$$的矩阵，其中$$N$$是输入point cloud里点的个数，而batch指的是batch size。而之后，再从beta distribution里sample一个一样大的矩阵，再将这个矩阵和之前那个矩阵都送给discriminator来判断。因为进行的是这样的操作，所以实际上其并不能控制keypoints会在哪个位置出现，因为采样的beta distribution不能保证位置信息，但$$\Phi(x)$$是有位置信息的，因为其代表的是每个位置的点的keypoint probability。所以说文章中那样的设置，只会控制$$\Phi(x)$$里0和1的数量，而无法控制其出现的位置。其出现的位置是由其它的loss来控制的。

*Reconstruction network*

给定一个point cloud的keypoint probability distribution

$$\lbrace \Phi(x) \in R | x \sim \mathcal M \rbrace$$

以及高维的embeddings

$$\lbrace h(x) \in R^F | x \sim \mathcal M \rbrace$$

使用一个decoder来重构原输入的point cloud。将这个decoder记为：$$\Psi: R^N \times R^{N \times F} \longrightarrow R^3$$，重构loss如下：

$$\mathcal L_{recon} = CD(\Psi(\lbrace \Phi(x) \in R | x \sim \mathcal M \rbrace, \lbrace h(x) \in R^F | x \sim \mathcal M \rbrace), X)  \tag{1}$$

其中CD是Chamfer distance。

(1) Salient Information Distillation

在公式1里，$$\Psi$$的输入是point cloud的keypoint distribution和高维embedding。我们的目的是找到一个具有显著keypoints的稀疏的keypoint集合，而且能够重构原输入point cloud。为了达到这个目标，我们从PointNet（）的max operation里吸取经验，设计了一个salient information distillation模块。这个模块使得网络能够给出可能性大的点的值大的feature：

$$\Psi = TopNet(max_{x \sim \mathcal M} \left[\Phi(x) h(x)\right])$$

因为$$\Phi(x)$$和$$h(x)$$维度不一样，我们上面的乘法用到了broadcasting。上述的max操作也是针对每个element的，从而$$max_{x \sim \mathcal M} \left[\Phi(x) h(x)\right] \in R^F$$。TopNet是decoder的结构，用了类似[TopNet: Structural Point Cloud Decoder](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tchapmi_TopNet_Structural_Point_Cloud_Decoder_CVPR_2019_paper.pdf)里的设计。

而且对于$$h(x)$$，我们考虑其的绝对值（也就是说$$h(x)$$里的非常大的负数不应该被忽略），从而最终的decoder设计为：

$$\Psi = TopNet(max_{x \sim \mathcal M} \left[\Phi(x) max(h(x), 0) \right], max_{x \sim \mathcal M} \left[\Phi(x) max(-h(x), 0) \right])$$

其中$$max_{x \sim \mathcal M} \left[\Phi(x) max(h(x), 0) \right]$$和$$max_{x \sim \mathcal M} \left[\Phi(x) max(-h(x), 0) \right]$$被连起来，再输入TopNet。

直觉上，我们的decoder使得网络去选择那些显著的并且semantic-rich（也就是$$h(x)$$和$$\Phi(x)$$乘积大的分量）的点的feature。也就是说，不显著的点，或者feature的值并不大的点就都被忽略了。


*Symmetric Regularization*

尽管我们一开始就获取了rotation invariant local descriptors，但是它们并不是symmetric invariant的（也就是说对于对称的点，其的descriptor并不是类似的）。对打大多数的objects，我们都可以认为其检测到的keypoints和features都是对称的，从而：

$$\mathcal L_{sym} = \frac{1}{|S|} \Sigma_{(x,x^{'}) \in S} (||\Phi(x) - \Phi(x^{'})|| + ||h(x) - h(x^{'})||)$$

其中$$S$$是所有的对称的点对构成的集合。这个loss仅仅在训练的时候被使用。

从而最终的loss是：

$$\mathcal L = \eta_1 \mathcal L_{recon} + \eta_2 \mathcal L_{GAN} + \eta_3 \mathcal L_{sym}$$


>这篇文章的一大亮点在于可以通过控制beta distribution里的参数$$\alpha$$和$$\beta$$来控制我们想要的keypoints的个数（假设keypoint probability>0.5就认为其是keypoint）。


### 6. [Single Image 3D Interpreter Network](https://arxiv.org/pdf/1604.08685.pdf)

*Jiajun Wu, Tianfan Xue, Joseph J. Lim, Yuandong Tian, Joshua B. Tenenbaum, Antonio Torralba, Willtian T. Freeman*

*ECCV 2016*

从单张2D的RGB图片中学习3D物体的结构是很困难的。以往的方法都是要么给定2D keypoints的位置，然后用某种optimization方法来推测3D信息，要么就直接在有ground truth的3D信息的生成数据上训练。

而这篇文章提出一个end-to-end的框架，叫做3D Interpreter Network (3D-INN)，其既利用2D keypoints标注信息又利用生成的3D数据作为训练数据，来学习预测2D keypoints的位置以及3D的结构信息。

这篇文章提出了两个技术创新。首先，提出了projection layer，将预测到的3D结构project到2D空间里，从而3D-INN能够通过2D keypoints的监督数据来对3D结构进行学习。其次，2D keypoints的heatmaps作为连接真实和生成数据之间的中间表示，使得3D-INN能够使用充足的生成的3D物体来训练。

虽然说现在deep learning在对物体类别进行识别的任务上做的已经很好了，但是这还不够，我们还需要了解每个类别众多物体之间的差别，比如说对于椅子这一类别，我们需要学习到intrinsic性质，比如说椅子高度、腿长、座椅宽度、材质等，以及其extrinsic性质，比如说椅子在图片中摆放的角度。

这篇文章利用3D skeleton（也就是3D keypoints和keypoints之间的连接）来表示一个物体的3D的结构，而不使用3D mesh或者depth map。正如fig1 (c)所示。

![Model Structure]({{ '/assets/images/UKPGAN-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*fig 1. 3D-INN的流程。*

在这篇文章里，他们还假设每个类别的物体，比如说chair，sofa，human等，都有一个定义好了的skeleton model。


### 7. [D3Feat: Joint Learning of Dense Detection and Description of 3D Local Features](https://arxiv.org/pdf/2003.03164.pdf)

*Xuyang Bai, Zixin Luo, Lei Zhou, Hongbo Fu, Long Quan, Chiew-Lan Tai*

*CVPR 2020 Oral*

[code](https://github.com/XuyangBai/D3Feat.pytorch)

一个成功的point cloud registration通常都要依靠具有区分度的3D feature desciptors之间的稀疏的匹配。尽管基于学习的3D feature descriptors发展得很快，但是3D feature detectors（也就是keypoint）发展的很慢，将这两个任务结合起来的研究就更少了。在这篇文章里，作者对于3D point cloud，利用一个3D的CNN，提出一些新颖有效的学习方法，从而能够对于3D point cloud里的每个点都给出一个description feature以及一个detection score。具体来说，文章提出了一个keypoint挑选的策略，克服了point cloud存在的density不同的困难，并且提出了一个由feature matching引导的自监督的detector loss。这篇文章的方法在indoor和outdoor的数据上都进行了测试，包括3DMatch和KITTI数据集。


point cloud registration（点云配准）的目的是在两个部分重合的point cloud fragements之间找到一个最优的transformation，其在SLAM等任务里面是很基础并且重要的。keypoint detection和description对于获取鲁棒的point cloud alignment的结果是最重要的两点。

>point cloud matching，registration和alignment指的是同一个任务。

最近关于3D local feature descriptions的研究开始转向基于学习的方法。因为获取ground truth标注数据的困难，现在绝大多数关于point cloud matching的工作都忽略了keypoint detection learning这个流程，只是随机采样了一个点的集合用来作为feature description。显然这种做法会有一些问题。首先，随机采样的点一般位置都不好，在geometric verification的过程中会导致不准确的transformation的估计。其次，这些随机采样的点很可能位于non-salient的位置（非显著）比如说某些光滑的表面上，这会导致indiscriminative的descriptors，对之后的matching来说效果不好。第三，为了采样的点能够覆盖描述整个场景，往往需要采样很多的点，这导致matching过程的效率变低。实际上，只需要一小部分的keypoints，其就能很好的做好matching任务了，而且定位准确的keypoints还能提高registration的精度。detector和descriptor之间不平衡的发展，让我们想要设计一个模型来联合学习它们。

基于学习的3D keypoint detector并没有得到多少关注。[3DFeat-Net: Weakly Supervised Local 3D Features for Point Cloud Registration](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Zi_Jian_Yew_3DFeat-Net_Weakly_Supervised_ECCV_2018_paper.pdf)预测了patch-wise的detection score，所以并没有考虑point cloud里所有的点的feature信息。[USIP: Unsupervised Stable Interest Point Detection from 3D Point Clouds](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_USIP_Unsupervised_Stable_Interest_Point_Detection_From_3D_Point_Clouds_ICCV_2019_paper.pdf)采用了一种非监督的方式来对经过任意rigid transformation的一对point cloud points输入预测consistent的3D keypoints，但因为其并没有联合训练detector和descriptor，所以说学习到的detector和用其他方式得到的descriptor可能不匹配，从而并不能对于point cloud给出合适的稀疏的3D feature。在这篇文章里，作者采用了一个联合学习的框架，不仅能稠密的预测keypoints，而且还能够将detector与descriptor整合在一起（共享参数），从而做到快速inference。

为了实现这个目的，我们从[D2-Net: A Trainable CNN for Joint Description and Detection of Local Features](https://arxiv.org/pdf/1905.03561.pdf)里吸取了灵感，这篇文章是关于在2D领域的detector和descriptor的联合学习。然而将D2-Net拓展到3D point clouds上并不容易。首先，需要一个能够预测dense feature的网络结构，而不是之前基于patch的结构。在这篇文章里，我们采用[KPConv: Flexible and deformable convolution for point clouds](https://openaccess.thecvf.com/content_ICCV_2019/papers/Thomas_KPConv_Flexible_and_Deformable_Convolution_for_Point_Clouds_ICCV_2019_paper.pdf)的方法，这篇论文提出了一个可以用在3D point cloud上的卷积运算，可以直接在unstructured的3D point cloud上构建一个CNN网络。其次，我们将D2-Net改造的能对于同一类别的不同object的不同point cloud的variation也能提出高度repeatable的3D keypoints。第三，D2-Net里的原始loss并不能保证在我们的问题里收敛，我们提出利用feature matching来指导一个新的self-supervised detector loss，从而使得detection scores和keypoints的可信赖度统一。

本篇文章的贡献有三点：

* 采用了基于KPConv的CNN框架，构建了一个联合学习3D local feature的detection和description的框架，来进行快速的inference。
* 我们提出了一个新颖的density-invariant keypoint选择策略，这是对于3D point clouds能获取高度repeatable keypoints的关键。
* 提出了一个从feature matching那得到信息的self-supervised detector loss，从而使得联合训练detector和descriptor能够收敛。

这是第一篇为3D point cloud同时学习3D local features的dense detection和description的文章，这个框架叫做D3Feat。


**Related Work**

*3D Local Descriptors*

早期的3D local descriptors都是hand-crafted的，对于noise和occlusion都是不鲁棒的。最近的研究方向转移到了基于学习的方法，这也是本文所关注的。绝大多数的基于学习的local descriptors需要输入为point cloud patches。基于学习的方法目前分为两大类，一类是patch-based networks，一类是fully-convolutional networks。对于第一类，几种3D数据表示被提出用来在3D数据里学习local geometric features：[Learning and matching multi-view descriptors for registration of point clouds]()，[The perfect match: 3d point cloud matching with smoothed densities]，[Ppfnet: Global context aware local features for robust 3d point matching]，[Ppf-foldnet: Unsupervised learning of rotation invariant 3d local descriptors]，[3Dmatch: Learning local geometric descriptors from rgb-d reconstructions]。但是基于point cloud patch的方法效率低。对于第二类，将CNN用在3D local descriptor学习并不是很多见。[Fully convolutional geometric features]，其采用了[3d spatio-temporal convnets: Minkowski convolutional neural networks]里的框架。

*3D Keypoint Detector*

目前绝大多数的3D keypoint detectors都是hand-crafted的。其过于依赖point cloud的局部几何特征，从而对于真实数据里的噪音、遮挡等都不具有鲁棒性。[]提出了一个非监督的方式来检测3D keypoints。但是USIP并不能对每个点都输出一个detection scores，而且如果要求输出的keypoint数很小的时候可能会效果不好。

*Joint Learned Descriptor and Detector*

在2D image matching领域，有几篇文章采用了joint learning detection和description的方法：[Lift: Learned invariant feature transform]，[Superpoint: Self-supervised interest point detection and description]，[Geodesc: Learning local descriptors by integrating geometry constraints]，[Lf-net: Learning local features from images]，[Unsuperpoint: End-to-ned unsupervised interest point detector and descriptor]，[R2d2: Repeatable and reliable detector and descriptor]和[Contextdesc: Learning local descriptors by integrating geometry constraints]。但是将这些方法应用到3D领域并不容易，也没有什么研究。[3dfeat-net: Weakly supervised local 3d features for point cloud registration]是唯一一篇对于3D point cloud联合学习detector和descriptor的文章。然而，他们的方法更加侧重于学习feature descriptor，而只用一个attention layer来估计每个point的salience，也就是detection是作为description网络的副产物存在的，所以他们方法里的keypoint detector的效果是不能保证的。另外，他们的方法使用point patches作为输入，其并没有直接只用point cloud作为输入效率高效果好。相反的，我们使用同一个forward pass来检测keypoint locations和每个点的feature。


**Joint Detection and Description Pipeline**

受到D2-Net的启发，不像之前的方法单独训练两个keypoint detection和description的网络，我们设计了一个网络来完成dense feature descriptor和feature detector两项任务。但是将D2-Net的思路用到3D领域并不简单，因为3D point cloud的不规则性和稀疏性。下面我们先介绍如何在3D point cloud上进行feature description和feature detection，之后再解释我们解决3D数据稀疏性的策略。

*Dense Feature Description*

为了解决再point clouds上如何进行convolution以及如何更好的获取局部几何信息的问题，[KPConv: Flexible and deformable convolution for point clouds](https://openaccess.thecvf.com/content_ICCV_2019/papers/Thomas_KPConv_Flexible_and_Deformable_Convolution_for_Point_Clouds_ICCV_2019_paper.pdf)提出了Kernel Point Convolution(KP-Conv)，使用kernel points来表示convolution weights，从而模仿2D convolution里的kernel pixels，从而就可以在原始的3D point cloud上定义convolution操作了。我们这篇文章采用KPConv作为backbone来进行dense feature extraction。我们先来介绍一下KPConv的公式。

给定一个点集，$$P \in R^{N \times 3}$$，以及一个集合的features，$$F_{in} \in R^{N \times D_{in}}$$，表示成一个矩阵的形式，$$x_i$$和$$f_i$$分别表示$$P$$里的第$$i$$个点以及其在$$F_{in}$$里对应的feature。从而点$$x$$位置的kernel $$g$$的convolution计算如下：

$$(F_{in} \ast g) = \Sigma_{x_i \in N_x} g(x_i-x)f_i \tag{1}$$

其中$$N_x$$是点$$x$$的领域，$$x_i$$是这个领域内别的点。kernel function $$g$$定义为：

$$g(x_i - x) = \Sigma_{k=1}^K h(x_i-x, \hat x_k)W_k \tag{2}$$

其中$$h$$是kernel point $$\hat x_k$$和点$$x_i$$之间的correlation function，$$W_k$$是kernel point $$\hat x_k$$的weight matrix，而$$K$$是kernel points的个数。

原论文里的方法对于point cloud里的point density并不是不变的。所以这篇文章加上了一个density normalization term：

$$(F_{in} \ast g) = \frac{1}{\lVert N_x \rVert} \Sigma_{x_i \in N_x} g(x_i-x)f_i \tag{1} \tag{3}$$

基于上述的normalized kernel point convolution，文章使用了UNet的结构（[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)）来构建一个网络，如fig1所示。

![Model Structure]({{ '/assets/images/D3FEAT-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*fig 1. (Left)D3Feat的网络结构。每一个block都是一个ResNet block，其使用KPConv来替代原本的image convolution。除了最后一层，每层都接了一个batch normalization和ReLU。(right)Keypoint Detection。在dense feature extraction之后，我们通过计算saliency score和channel max score来进行keypoint detection score的计算。*

不同于只能获取稀疏feature description的patch-based方法，我们的网络可以获取dense feature description。我们网络的输出是一个dense feature map，$$F \in R^{N \times c}$$，其中$$c$$是feature向量的维数，$$N$$是point cloud里点的数量。点$$x_i$$相关的descriptor记为$$d_i$$：$$d_i = F_{i:}, d_i \in R^{c}$$。每个点的descriptor都被$$L_2$$标准化到单位长度。

*Dense Keypoint Detection*

D2-Net里通过找到feature map的局部的spatial和channel的最大值来检测到2D keypoints，并且使用一个softmax来衡量这个局部最大值点的值。因为CNN的结构以及images本身是一个2维矩阵，而CNN的feature maps也都是高维tensor，所以每个pixel的neighborhood就直接是它相邻的pixels就可以。

为了将D2-Net里的方法拓展到3D上，可以用radius neighborhood来替代来解决point clouds里的点并不是均匀分布的问题。但是每个点的radius neighborhood里的点的数量差别很大。在这种情况下，如果我们直接使用一个softmax函数来在spatial维度衡量局部最大值，那些有着很少的点的布局区域就会有很高的值（比如说对于indoor场景来说的边缘位置，或者对于outdoor场景来说远离Lidar中心的位置）。为了解决这个问题，我们提出一个density-invariant saliency值来衡量一个点和其邻居点相比的saliency。

我们有dense feature map，$$F \in R^{N \times c}$$，我们将$$F$$看作一系列3D responses $$D^k, k=1, \cdots, c$$的集合：

$$D^k = F_{:K}, D^k \in R^N$$

$$x_i$$是一个keypoint的标准就是：

$$x_i$$ is a keypoint $$\iff k=argmax_{t} D_i^t, i = argmax_{j \in N_{x_i}} D_j^k$$

其中$$N_{x_i}$$是$$x_i$$的radius neighborhood。这表明一个点$$x_i$$如果想成为keypoint，首先找到具有最大值的那个通道，然后对于这个通道，$$x_i$$是其邻域内值最大的那个。在实际训练的过程中，我们将上述过程条件放松，从而使得能够训练，引入了两个scores，如fig1右边所示。下面详细介绍这两个scores。

* Density-invariant saliency score

这个值用来衡量每个点相对于其邻域内的其它点有多么的salient。在D2-Net里，衡量局部最大值的score是：

$$\alpha_i^k = \frac{exp(D_i^k)}{\Sigma_{x_j \in N_{x_i}} exp(D_j^k)}$$

这个公式对于稀疏性来说并不是不变的。稀疏的区域比稠密的区域自然的就会有更高的score，因为这个值是被总和归一化过的。从而我们设计一个density-invariant saliency score：

$$\alpha_i^k = ln(1 + exp(D_i^k - \frac{1}{\lVert N_{x_i} \rVert} \Sigma_{x_j \in N_{x_i}} D_j^k))$$

在上述公式里，每个点的saliency是通过这个点和其neighborhood点的feature的均值的差来表示的。而且使用均值而不是总和可以避免值受到neighborhood里的point的数目的影响。

* Channel max score

这个score是用来对于每个point挑选最重要的通道用的：

$$\beta_i^k = \frac{D_i^k}{max_t (D_i^t)}$$

最后，这两个值被综合考虑为最终的keypoint detection score:

$$s_i = max_k (\alpha_i^k \beta_i^k)$$

在我们获取了整个输入的point cloud的keypoint score map之后，我们就可以选取那些有最高值的那些point为keypoints了。



**Joint Optimizating Detection & Description**

设计一个有效的supervision signal是联合学习一个descriptor和一个detector的关键。在这一节里，我们先介绍descriptor的metric learning loss，然后再设计一个自监督的detector loss。

* Descriptor loss

为了优化descriptor网络，很多工作都使用metric learning方法，比如contrastive loss或者triplet loss。我们将会使用contrastive loss因为实验证明效果更好。至于如何找到有效的采样方法来选择训练数据对，我们采用[Working hard to know your neighbor's margins: Local descriptor learning loss](Working hard to know your neighbor's margins: Local descriptor learning loss)里说的hardest in batch策略来使得网络集中注意力于hard pairs。

给定一对部分重合的point cloud fragments，$$P$$和$$Q$$，以及$$n$$个对应的3D keypoints对组成的集合。假设$$(A_i, B_i)$$是两个point cloud中对应的点对，并且他们有相应的descriptors，$$d_{A_i}$$和$$d_{B_i}$$，以及scores $$s_{A_i}$$和$$s_{B_i}$$。一个positive pair之间的距离被定义为它们descriptors之间的欧氏距离：

$$d_{pos}(i) = \lVert d_{A_i} - d_{B_i} \rVert_2$$

一个negative pair的距离为：

$$d_{neg}(i) = min \lbrace \lVert d_{A_i} - d_{B_j} \rVert_2 \rbrace s.t. \lVert B_j - B_i \rVert_2 > R$$

其中$$R$$是safe radius，$$B_j$$是真实的对应关系的safe radius外hardest negative点。contrastive margin loss被定义为：

$$L_{desc} = \frac{1}{n} \Sigma_i \left[ max(0, d_{pos}(i) - M_{pos}) + max(0, M_{neg} - d_{neg}(i))\right]$$

其中$$M_{pos}$$和$$M_{neg}$$分别是positive和negative pairs对应的margin的值。


* Detector loss

为了优化detection的效果，我们找到一个loss，其对于容易匹配的点对具有更高的keypoint detection score，对于难匹配的点对则具有较低的keypoint detection score。在D2-Net里，作者提出了一个triplet margin loss的拓展来用于同时优化detector和descriptor：

$$L_{det} = \Sigma_{i} \frac{s_{A_i}s_{B_i}}{\Sigma_{i} s_{A_i}s_{B_i}}max(0, M + d_pos(i)^2 - d_neg{i}^2)$$

其中$$M$$是triplet margin。D2-Net认为为了最小化loss，网络对于最容易匹配的那些点（也就是$$d_{pos}$$很小，而$$d_{neg}$$很大的那些点）应该具有较高的keypoint detection score。但是本文发现他们的方法对于3D keypoint来说不好使。

因此本文也涉及了一个loss项来显式的引导scores的gradient。从一个自监督的角度来看，我们使用feature matching的结果来衡量每对对应点的辨识度，其将会引导每个keypoint的score的gradient flow。

$$L_{det} = \frac{1}{n} \Sigma_i \left[(d_{pos}(i) - d_{neg}(i))(s_{A_i} + s_{B_i})\right]$$

直觉上，如果$$d_{pos}(i) < d_{neg}(i)$$，这说明使用最近邻搜索就能找到正确的匹配点，上述loss鼓励这样的点的$$s_{A_i}$$和$$s_{B_i}$$尽量大一些，也就是被选为keypoints的可能性大一些。相反的，如果$$d_{pos}(i) > d_{neg}(i)$$，那么对应的点对对于当前的网络来说就不够那么具有辨识度来使得正确的对应关系被建立，所以上述的loss会使得这样的点成为keypoints的可能性变小。为了最小化上述loss，网络需要对那些匹配的点给出高keypoint detection scores，对于那些不匹配的给出低scores。



### 8. [Weakly-Supervised Discovery of Geometry-Aware Representation for 3D Human Pose Estimation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Weakly-Supervised_Discovery_of_Geometry-Aware_Representation_for_3D_Human_Pose_Estimation_CVPR_2019_paper.pdf)
*Xipeng Chen, Kwan-Yee Lin, Wentao Liu, Chen Qian, Liang Lin*

*CVPR 2019*



This work proposed a method to learn 3D coordinates of human body joints in order to do human pose estimation. This model is based on skeleton extracted from the raw RGB images, not an End-to-end framework.

Hyperparameter: Number of Keypoints $$K$$.

**Step1** Inputs are source image $$I_s$$ and target image $$I_t$$, and the rotation matrix are known due to the parameters of cameras. First, they use existing skeleton algorithm to extract skeleton maps of $$I_s$$ and $$I_t$$.

**Step2** Instead of a traditional encoder-decoder framework, they use a novel view synthesis method, i.e., source image $$I_s$$ are encoded and combined with rotation matrix $$R_{s \rightarrow t}$$, target image $$I_t$$ are reconstructed from the decoder. The 3D keypoint coordinates are the output of the encoder, as a geometry-aware representation, as explained in the paper. They also design the bidirectional encoder-decoder framework, which hinges on two encoder-decoder networks with same architecture to perform view synthesis in the two directions simultaneously, i.e., from $$I_s$$ to $$I_t$$ and from $$I_t$$ to $$I_s$$. These two reconstructions will involve two losses.

**Step3** They believe that the 3D keypoints of these two images should be the same. There are two encoders and the outputs should be the same, thus involve a new loss.

![Model Structure1]({{ '/assets/images/weak-supervised.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1 The framework of learning a geometry representation for 3D human pose in a weakly-supervised manner. There are three main components. (a)Image-skeleton mapping module is used to obtain 2D skeleton maps from raw images. (b)View synthesis module is in a position to learn the geometry representation in latent space by generating skeleton map under viewpoint $$j$$ from skeleton map under viewpoint $$i$$. (c) Since there is no explicit constrain to facilitate the representation to be semantic, a representation consistency constrain mechanism is proposed to further refine the representation.*








### 10. [Hand Keypoint Detection in Single Images using Multiview Bootstrapping](https://openaccess.thecvf.com/content_cvpr_2017/papers/Simon_Hand_Keypoint_Detection_CVPR_2017_paper.pdf)
*Tomas Simon, Hanbyul Joo, Iain Matthews, Yaser Sheikh*

*CVPR 2017*

This paper proposed a framework to learn 3D keypoints of hand. 

The input is a keypoint detector $$d_0$$ trained on a small labelled dataset $$T_0$$, a sequence of images $$\left\{I_v^f, v=1,2,...,V, f=1,2,..,F\right\}$$, with $$v$$ denote the camera view and $$f$$ denote the time frame.

Hyperparameter: Number of Keypoints $$K$$.

**Step1** First use $$d_0$$ to calculate the image coordinates (no depth) and confidence of each keypoint $$k$$ of $$I_v^f$$, denoted as $$x_v^{f,k}$$ and $$C_v^{f,k}$$. Then use the random sample consensus to pick inliers out of each set $$\left\{(x_v^{k}, C_v^{k})\right\}$$ for each time frame $$f$$. Then the 3D wolrd coodinates are computed as:

$$X_v^{k} = argmin_X \Sigma_{v \in I_v^{k}} ||P_v(X)-x_v^k||^2_F$$

where $$I_v^k$$ is the inlier set, with $$X_f^k \in R^3$$ the 3D triangulated keypoint $$k$$ in frame $$f$$, and $$P_v(X) \in R^2$$ denotes projection of 3D point $$X$$ into camera view $$v$$. They use calibrated cameras, thus $$P_v$$ are known.

**Step2** Then they use a window through the time frame, and pick the frame with highest score. The score is defined as the sum of $$C_v^k$$, thus the frame that has the biggest confidence of all keypoints detection from all camera views.

**Step3** After picking this frame, they add the labelled images into the orignal training dataset and train a new keypoint detector, $$d_1$$. And so on.



### 11. [Self-Supervised Learning of 3D Human Pose Using Multi-View Geometry](https://openaccess.thecvf.com/content_CVPR_2019/html/Kocabas_Self-Supervised_Learning_of_3D_Human_Pose_Using_Multi-View_Geometry_CVPR_2019_paper.html)

*CVPR 2019*


### 12. [Unsupervised Learning of Probably Symmetric Deformable 3D Objects From Images in the Wild](https://openaccess.thecvf.com/content_CVPR_2020/html/Wu_Unsupervised_Learning_of_Probably_Symmetric_Deformable_3D_Objects_From_Images_CVPR_2020_paper.html)

*CVPR 2020*

### 13. [Weakly-Supervised Discovery of Geometry-Aware Representation for 3D Human Pose Estimation](https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Weakly-Supervised_Discovery_of_Geometry-Aware_Representation_for_3D_Human_Pose_Estimation_CVPR_2019_paper.html)

*CVPR 2019*


### 14. [Self-Supervised 3D Hand Pose Estimation Through Training by Fitting](https://openaccess.thecvf.com/content_CVPR_2019/html/Wan_Self-Supervised_3D_Hand_Pose_Estimation_Through_Training_by_Fitting_CVPR_2019_paper.html)

*CVPR 2019*



### 16. [End-to-End Learning of Multi-category 3D Pose and Shape Estimation](https://arxiv.org/pdf/2112.10196.pdf)

*Yigit Baran Can, Alexander Liniger, Danda Pani Paudel, Luc Van Gool*

*Arxiv 2021*

>这篇文章来自于ETHZ的CVLab，是大佬Luc Van Gool指导的论文。

**Abstract**

在这篇文章里，我们通过keypoints来研究物体的形状和姿态的representation。因此，我们提出了一个端到端的方法来从一张图片里检测2D keypoints，并将其lift到3D keypoints。我们的方法能学习到3D keypoints，但是只有2D keypoints的标注。我们的方法除了是端到端的（输入为2D图片，输出为3D keypoints），还使用了同一个网络来处理多个类别的物体（也就是不同category的images的keypoints可以用同一个网络来学习）。我们使用了基于Transformer的网络来检测keypoints，以及综合图片里的视觉信息。而这些视觉信息之后被用来将2D keypoints lift到3D keypoints.我们的方法可以处理含有遮挡的情况，而且对很大一部分的物体类别都是适用的。在三个benchmarks上的实验证明了我们的方法要比sota的效果好。


**1. Introduction**

基于keypoints的shape和pose的representations是很不错的，因为它足够简单也足够好进行后续的操作。基于keypoints的shape和pose的representations的应用案例包括，3D reconstruction，registration，human body pose analysis，recognition，generation等等。而keypoints通常是作为2D keypoints出现的，因为在图像上标注2D keypoints比较容易。然而，2D keypoints往往不能满足后续任务的需要。在很多应用里，既需要3D shape，也需要3D pose，比如说augmented reality任务。

因此，我们需要来预测3D keypoints，正如这些文章里做的那样：[Augmented autoencoders: implicit 3d orientation learning for 6d object detection]()，[Discovery of latent 3d keypoints via end-to-end geometric reasoning]()，[Learning deep network for detecting 3d object keypoints and 6d poses]()。然而，这些方法都有着以下两个问题之一或者都有：（1）需要3D keypoints、pose或者多个角度的图片来作为监督数据；（2）并没有直接相对于一个固定frame来做pose reasoning。而这篇文章提出的方法，可以直接从单张图片里学习到3D keypoints和pose，从而为一系列下游任务，比如说scene understanding，augmented reality等提供帮助。另一类方法，基于template的从单张图片学习的方法，也可以从2D keypoints里获取3D keypoints和pose：[Optimal pose and shape estimation for category-level 3d object perception]()，[In perfect: Certifiably optimal 3d shape reconstruction from 2d landmarks]()。然而，基于template的方法，除了需要templates之外，还对自遮挡非常敏感，[3d registration for self-occluded objects in context]()给出了解释。因此，我们并不使用基于template的方法，而是使用基于学习的方法，来从单张图片里推断物体的3D keypoints和pose。

在这篇文章里，我们在training和inference的时候，对于每个物体，都只有一张图片。这个设定让我们可以使用十分广泛的数据集，甚至直接从网上下载图片就可以。为了方法具有更好的可扩展性，我们只需要2D keypoints和物体的类别来作为监督数据。也就是说，我们希望我们的模型能够从单张图片里学到物体的3D keypoints和pose，而这些图片里的物体是不同类别，不同个体，甚至我们都不需要同一个物体不同角度的views。

现有的那些能够从单张图片里仅仅通过物体类别这个信息就能学到3D shape和pose的方法叫做deep non-rigid structure-from-motion（NrSfM）。这些方法可以被分为两类：每个模型只能针对一个物体类别的图片：[Deep non-rigid structure from motion]()，[Procrustean regression networks: Learning 3d structure of non-rigid objects from 2d annotations]()，[Procrustean autoencoder for unsupervised lifting]()，[Pr-rrn: Pairwise-regularized residual-recursive networks for non-rigid structure-from-motion]()，以及每个模型可以针对多个物体类别的图片：[C3DPO: canonical 3d pose networks for non-rigid structure from motion]()，也就是single和multi-category两个类别。multi-category方法显然更有意思，因为（1）计算复杂度更低，仅仅需要一个神经网络就可以从含有不同类别物体的图片里学习keypoints和pose；（2）有可能能够建立不同类别物体之间的关联性。后面这个特性不仅可以让我们来衡量不同类别物体之间的相似性，而且使得模型的泛化效果更好。

现有的这些方法，[Deep non-rigid structure from motion]()，[Procrustean regression networks: Learning 3d structure of non-rigid objects from 2d annotations]()，[Procrustean autoencoder for unsupervised lifting]()，[Pr-rrn: Pairwise-regularized residual-recursive networks for non-rigid structure-from-motion]()，[C3DPO: canonical 3d pose networks for non-rigid structure from motion]()，[Unsupervised 3d pose estimation with geometric self-supervision]()，都是由两个stages组成的：2D keypoints extraction和3D shape and pose estimation。这两个stages一般都是独立处理的。但我们认为，这两个stage是相关的，而且可以从互相那里获得帮助，从而所获取的2D keypoints就更加适合下游的3D reasoning了。为了实现这个目标，我们在提取2D keypoints的keypoint extraction network里加入结构，从而增加一部分信息，也就是伴随着2D keypoints的位置输出，还输出visual context information。然后，这个visual context information和2D keypoints一起被用于后续将2D keypoints lift到3D的network里。我们的实验证明了在获取3D pose和shape时使用visual context information的有效性。


类似于这篇文章（[C3DPO: canonical 3d pose networks for non-rigid structure from motion]()）里的做法，我们利用dictionary learning的方法来表示3D shape，其中所使用的shape basis是通过学习得到的（这个shape basis集合包含各个类别的物体的shape basis）。然后，利用shape basis coefficients就可以恢复每个物体的shape了。然而，正如[C3DPO: canonical 3d pose networks for non-rigid structure from motion]()和[Deep non-rigid structure from motion]()所说，shape basis的size需要很仔细地被设置。在multi-category的情况下，所有的object categories共享一个latent space（也就是shape basis构成的space），而每个category可以有不同长度的shape basis size。而且，如果直接使用shape basis coefficients来将shape basis重构为object shape，会导致模型对于细节或者小扰动过于敏感，文章里提出了一个简单的方法来解决这个问题：学习一个threshold vector来cut-off shape basis coefficients。这个简单的方法能达到很好的效果，要比直接使用稀疏字典学习的方法简单很多。


这篇文章的主要贡献如下：

* 单个端到端的网络，能够对多种不同类别的objects重构它们的3D shape和pose。
* 提出使用image context information来辅助3D shape和pose的学习，而不仅仅只有2D keypoints.
* 我们的方法在multi-category的设置下取得了sota的结果，而且效果好了很多。


**2. Related Work**

从单张图片里将可变形物体（deformable obejcts）的2D keypoints lift到3D keypoint这个问题一般在NrSfM任务框架里被研究。而在NrSfM框架里，任务是从物体随着时间变化而获取的若干张图片里获取物体的poses或者viewpoints。已经有非常多的研究工作利用各种方法来改进NrSfM任务的效果，包括使用sparse dictionary learning（[]()，[]()）、low-rank constraints（[]()）、union of local subsapces（[]()）、diffeomorphism（[]()）以及coarse-to-fine low-rank reconstruction（[]()）。如果我们将同一个类别的不同的object的图片当作同一个object的在不同时刻拍下来的图片的话，我们也可以使用NrSfM框架来构造一个category-specific模型来估计pose和viewpoint，比如说这些工作就是这么干的：[]()，[]()，[]()。

从单张图片里获取一个物体的3D结构并没有过多的被研究。在[]()里，instance segmentation数据集被用来训练一个模型，从而能够在给一张图片的情况下输出3D mesh reconstructions。[]()通过建立2D和3D keypoints之间的联系来改进了结果。尽管最近一些结果已经可以预测viewpoint和non-rigid meshes，但它们针对的都是变化比较小的物体，比如说人脸：[]()，[]()，[]()。

跟我们工作最相近的一条line of research考虑的任务是为不同的输入类别构建单个模型。C3DPO（[]()）的方法是学会将object deformation和viewpoint change这两个因素分解开。它们使用一个网络学到物体相对于canonical shape（规范化shape）的rotation。[]()使用Procrustean regression来确定motion和shape。他们还提出了使用CNN从图片输入里输出human的3D keypoints。然而，他们的方法并不能处理multi-category的情况，也不能处理被遮挡住的keypoints。而且，这个方法还需要时序信息。[]()也研究了human pose estimation，他们提出了一个cyclic-loss，使用了GAN并加入了额外数据集来训练这个GAN。但是，他们的方法仅限于human pose estimation。最近，[]()将Procrustean regression方法进一步加上了autoencoders，这样的方法就可以不需要时序信息就能获取3D shapes了。然而，他们的方法在Procrustean alignement的基础上，还需要额外两个autoencoders，从而在测试的时候很慢。上述所有的这些方法都是以2D keypoints作为他们的输入，而不是图片，所以他们使用了一个keypoint detector，比如说stacked hourglass network。


**3. Multi-category from a Single View**

我们通过获取3D keypoints的方式来表示3D structures，而输入仅仅是某个category的object的一张图片。在训练过程中，我们只有2D keypoints的位置信息以及category类别信息。为了简单起见，我们将方法分为两个stages：（1）从图片种获取2D keypoints和category；（2）将2D keypoints lift到3D。下面，我们会先介绍如何将2D keypoints lift到3D。我们的方法是在NrSfM框架下进行描述的。我们先介绍lifter network，然后再介绍2D keypoint extractor以及如何将这两部分连接在一起成为一个端到端的网络的。


**3.1 Preliminaries - NrSfM**

$$\pmb Y_i = \left[ \pmb y_{i1}, \cdots, \pmb y_{ik} \right] \in \mathbb R^{2 \times k}$$表示的是对于第$$i$$个view，由$$k$$个2D keypoints摞起来的矩阵，也就是每一列表示一个keypoint的2D location。从而，我们将第$$i$$个view的3D结构表示为$$\pmb X_i = \alpha_i \pmb S$$，其中$$\pmb S \in \mathbb R^{d \times 3k}$$是shape basis，$$\alpha_i \in \mathbb R^d$$是coefficients。为了简单起见，我们的相机参数里的projection model是标准的，也就是$$\Pi = \left[\pmb I_{2 \times 2}, \pmb 0 \right]$$。给定相机旋转矩阵$$\pmb R_i \in SO(3)$$，从而式子就写为：$$\pmb Y_i = \Pi \pmb R_i (mat(\alpha_i \pmb S_i))$$，其中$$mat(\cdot)$$操作是将一个$$\mathbb^{1 \times 3k}$$的矩阵变成一个$$\mathbb^{3 \times k}$$的矩阵的操作。从而，如果我们有$$n$$张图片，并且有这$$n$$张图片里2D keypoints的location的数据，也就是$$Y_i$$，在NrSfM框架里，目标函数就是：

$$ min_{\alpha_i, \pmb S, \pmb R_i \in SO(3)} \Sigma_{i=1}^n \mathcal L(\pmb Y_i, \Pi \pmb R_i (mat(\alpha_i \pmb S_i))) \tag{1}$$

其中$$\mathcal L$$表示的是某种loss函数。

上述问题是ill-posed的，因此有一系列关于$$\alpha_i$$和$$\pmb S$$的假设就被提了出来。最常见的是：low_rank（[]()）,finite-basis（[]()），以及sparse combination（[]()）。在这篇文章里，我们更感兴趣的是如何通过学习的方法来解决公式(1)。但是和别的NrSfM问题不一样的是，我们的输入是单张图片，而且我们的单个网络需要能够处理不同object的图片。


**3.2 Multi-category Formulation**

我们考虑的是multi-category的情况，这对应着multi-class NrSfM。因此，$$mat(\alpha_i \pmb S) \in \mathbb R^{3 \times k}$$需要能够对于不同的category有不同数量的keypoints，也就是$$k$$不是固定的。$$\pmb Z$$表示object categories的集合，而$$z_i$$表示的是sample $$i$$的category。$$\pmb Z$$里的每个category $$z$$的keypoints数量记为$$k_z$$，因此，一共有$$k = \Sigma_z k_z$$个keypoints。为了对每个类别找到正确的keypoints的数量，我们使用了一个subset selection vector $$\zeta_z \in {0,1}^k$$，用来标记这些keypoints里哪个是category $$z$$的。有了上述这些定义之后，我们可以将公式1重新描述为：

$$ min_{\alpha_i, \pmb S, \pmb R_i \in SO(3)} \Sigma_{i=1}^n \mathcal L(\pmb Y_i \ast \zeta_{z_i}^T, \Pi \pmb R_i(\alpha_i \pmb S) \ast \zeta_{z_i}^T) \tag{2}$$

上述公式里的$$\ast$$表示这个操作是element-wise乘法，而不是矩阵乘法，而且使用了broadcasting。在公式2里，$$\pmb R_i$$和$$\alpha_i$$是依赖于输入的，而$$\pmb S$$是所有图片所有category共用的。为了将公式2变为可学习的，我们让$$\alpha_i$$是某个网络的输出，而这个网络的输入是$$\pmb Y_i$$，也就是$$\alpha(\pmb Y_i)$$。我们再将$$\alpha$$函数分解为两个函数，$$\alpha(\pmb Y_i) = g(f(\pmb Y_i))$$，其中$$g$$是一个affine function，$$g(v) = W_g v + b_g$$，而$$v \in \mathbb R^{F}$$，$$W_g \in \mathbb R^{D \times F}$$，$$b \in \mathbb R^{D}$$。我们对$$f$$不做任何限制，它就是一个输入为$$\pmb Y_i$$，输出为一个维度为$$F$$的向量的函数。同样的，我们也将$$\pmb R_i$$表示为$$\pmb Y_i$$的函数。既然我们使用learning的方法来实现上述目标函数，记网络的参数为$$\theta$$，那么公式2就写为：

$$ min_{\theta} \Sigma_i \mathcal L(\pmb Y_i \ast \zeta_{z_i}^T, \Pi \pmb R_i(\pmb Y_i) mat((W_g f(Y_i) + b_g) \pmb S) \ast \zeta_{z_i}^T) \tag{3}$$

从而，shape basis coefficient $$\alpha$$就隐式的用$$W$$和$$b$$来表示了，而$$W$$和$$b$$是网络参数，所以是所有图片共享的。


**3.3 Cut-off Shape Coefficients**


### 17. [Weakly-Supervised Discovery of Geometry-Aware Representation for 3D Human Pose Estimation](https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Weakly-Supervised_Discovery_of_Geometry-Aware_Representation_for_3D_Human_Pose_Estimation_CVPR_2019_paper.html)

*CVPR 2019*


### 18. [Watch It Move: Unsupervised Discovery of 3D Joints for Re-Posing of Articulated Objects](https://openaccess.thecvf.com/content/CVPR2022/html/Noguchi_Watch_It_Move_Unsupervised_Discovery_of_3D_Joints_for_Re-Posing_CVPR_2022_paper.html)

*CVPR 2022*


### 19. [Unsupervised Temporal Learning on Monocular Videos for 3D Human Pose Estimation](https://arxiv.org/pdf/2012.01511.pdf)

*Arxiv 2022*


### 20. [Weakly-Supervised Discovery of Geometry-Aware Representation for 3D Human Pose Estimation](https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Weakly-Supervised_Discovery_of_Geometry-Aware_Representation_for_3D_Human_Pose_Estimation_CVPR_2019_paper.html)

*CVPR 2019*


### 21. [Unsupervised Learning of 3D Semantic Keypoints with Mutual Reconstruction](https://arxiv.org/pdf/2203.10212.pdf)

*Arxiv 2022*


### 22. [Learning deep network for detecting 3D object keypoints and 6D poses](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhao_Learning_Deep_Network_for_Detecting_3D_Object_Keypoints_and_6D_CVPR_2020_paper.pdf)
*Wanqing Zhao, Shaobo Zhang, Ziyu Guan, Wei Zhao*

*CVPR 2020*


### 23. [SNAKE: Shape-aware Neural 3D Keypoint Field](https://arxiv.org/pdf/2206.01724.pdf)

*Arxiv 2022*


### 24. [Semi-automatic 3D Object Keypoint Annotation and Detection for the Masses](https://arxiv.org/abs/2201.07665)

*Arxiv 2022*

[CODE](https://github.com/ethz-asl/object_keypoints)


### 25. [EPro-PnP: Generalized End-to-End Probabilistic Perspective-n-Points for Monocular Object Pose Estimation](https://github.com/tjiiv-cprg/EPro-PnP)

*CVPR 2022 Best Student Paper*

[CODE](https://github.com/tjiiv-cprg/EPro-PnP)

**Abstract**

通过perspective-n-points（PnP）来从单张RGB图片里定位3D物体是CV领域一个长期未解决的问题。通过端到端的deep learning方法，最近的研究表明可以将PnP看作一个differentiable的layer，从而2D-3D点的correspondence就可以通过关于object pose的反向传播而被部分确定下来。然而，从图片里来直接学习没有任何约束的2D-3D点还是没有被解决，因为pose实际上是non-differentiable的。在这篇文章里，我们提出了EPro-PnP，一个probabilistic PnP层，来进行general的端到端的pose estimation，其会输出一个pose的distribution，这些pose都在$$SE(3)$$这个流形上，也就相当于将Softmax分类层引入了连续域。2D-3D点的坐标以及对应的weights被当作中间变量来学习，其是通过最小化预测和目标pose distribution之间的KL距离来实现的。这个方法深层次的原理统一了现有的这些方法，而且和注意力机制有些相似。EPro-PnP极大的超过了现有的baseline，在LineMOD 6DoF pose estimation和nuScenes 3D object detection这两个benchmark上极大的减小了基于PnP的方法和那些针对特别任务的方法。

**1. Introduction**

从单张RGB图片中估计3D物体的姿态（也就是position和orientation）是CV领域一个重要的任务。这个领域一般会被细分为更细致的任务，比如说机器人的6DoF姿态估计，自动驾驶的3D物体检测等。尽管这些任务都有着同为姿态估计问题的共性，但数据本身的不同使得对于这些不同的任务我们会选择不同的方法来解决。3D object detection领域的benchmarks上最好效果的结果使用了端到端的deep learning模型来直接进行4DoF的估计。而6DoF姿态估计的benchmark上最好的结果则是由geometry-based的方法所获得的。然而，如何将这两种方法结合起来是很具有挑战性的，也就是说如何用端到端的方式来训练一个geometric模型去获取object的姿态。

最近有一些研究提出可以基于perspective-n-points（PnP）方法来设计这样的端到端的模型：[Dsac-differentiable ransac for camera localization]，[Learning less is more: 6d camera localization via 3d surface regression]，[Solving the blind perspective-n-point problem end-to-end with robust differentiable geometric optimization]和[End-to-end learnable geometric vision by backpropagating pnp optimization]。PnP算法将学习姿态这个问题转化为学习object space里的3d points和image space里它们的2d projections之间的对应关系的这个新问题。vanilla的对应关系学习算法会使用geometric priors来建立一些loss function，迫使网络来学习一系列预定义的对应关系。端到端的对应关系学习算法将PnP当作一个differentiable的层并且使用基于姿态的loss function，从而姿态的error就可以被反向传播到2D-3D的对应关系上去。

然而，现有的differentiable PnP工作仅仅学习了一部分这样的对应关系（要么就是只有2D坐标，要么就是只有3D坐标，要么就是只有2D和3D之间对应的权重，也就是上面提到的那几篇论文），而且这些论文会将那些没有被学习的部分当作先验知识来辅助。这就产生了一个问题：为什么不直接用端到端的形式来一起学习这些点的坐标值以及它们之间对应关系的权重呢？这个问题的一个简单的回答是：PnP问题的解天生就在某些点不可微，会导致训练困难以及训练不收敛。更加细致的说就是，一个PnP问题会有多个解，这样就会导致反向传播不稳定。

为了解决上述的问题，我们提出了一个扩展版的端到端的概率的PnP方法（EPro-PnP），来直接从图片里学习以权重表示的2D-3D点的对应关系，如fig1所示。

![EPro-PnP-1]({{ '/assets/images/EPRO-PNP-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. EPro-PnP是端到端2D-3D对应关系学习的一个general的解。在这篇文章里，我们将会使用EPro-PnP来训练两个不同的网络：（a）一个现成的dense correspondence网络；（b）一个新的deformable correspondence网络用来发掘新的2D-3D之间的对应关系。*

主要的想法很直接：deterministic pose是non-differentiable的，但是pose的概率分布是differentiable的（就像分类任务里softmax给出的不同类别的概率分布一样，所以分类任务对于softmax层是differentiable的）。因为，我们将PnP问题的输出理解为一个由2D-3D对应关系来表示的概率分布（这个对应关系是可学习的）。

>这个想法其实很有意思，因为其可以拓展到几乎所有的问题上，因为很多机器学习任务本质上都是一个非凸优化问题，或者说有很多个最优解，那么如果希望误差反向传播到我想要的那个解上，本质上是ill posed的，也就是上面说的non differetiable，而如果将这所有的解都用概率表示出来，那就变得和softmax的结果一样了，那么误差就会反向传播到所有的解上，当然是可微的了，而这些解的重要性就可以通过不同的权重来表示。

在训练中，预测的pose分布和目标的pose分布之间的KL散度将作为loss function，其可以用蒙地卡罗算法来计算。

作为一个general的方法，EPro-PnP本质上统一了现有的correspondence的学习方法（见3.1节）。更进一步的是，正如注意力机制一样，对应关系权重可以被自动训练为更加关注那些重要的点对，我们可以从注意力机制的设计方法上来获取设计网络的灵感。

总结来说，我们的主要贡献如下：
* 我们提出了EPro-PnP，一个probabilistic PnP层用来通过可学习的2D-3D对应关系来进行general的端到端的姿态估计
* 我们展示了EPro-PnP可以轻易的达到现有的6DoF pose estimation的sota效果
* 我们通过提出deformable correspondence learning来进行3D object detection展示了EPro-PnP的灵活性


**2. Related Work**

*Geometry-Based Object Pose Estimation*

一般来说，geometry-based的方法所学习到的是points，edges或者其它那些受到相机投射约束的representations。然后，姿态可以通过优化过程来获得。一大批工作预测的都是points，其可以被分为两类：sparse keypoints和dense correspondences。[Rtm3d: Real-time monocular 3d detection from object keypoints for autonomous driving]和[BB8: A scable, accurate, robust to partial occlusion method for predicting the 3d poses of challenging objects without using depth]将3D bounding box的角作为keypoints，[Deep manta: A coarse-to-fine many-task network for joint 2d and 3d vehicle analysis from monocular image]用的是handcrafted的模板，而[Pvnet: Pixel-wise voting network for 6dof pose estimation]使用的是farthest point sampling。[Monorun: Monocular 3d object detection by reconstruction and uncertainty propagation]，[Cdpn: Coordinates-based disentangled pose network for real-time rgb-based 6-dof object pose estimation]，[Pix2Pose: Pixel-wise coordinate regression of objects for 6d pose estimation]，[Normalized object coordinate space for category-level 6d object pose and size estimation]，[Dpod: 6d pose object detector and refiner]，这些都是dense coorepsondence方法。绝大多数的geometry-based的方法都是一个two-stage的方法，中间的representations（也就是2D-3D对应关系）是通过一个surrogate loss function来学习的，这相对于端到端的方法来说就是sub-optimal的。

*End-to-End Correspondence Learning*

为了解决用surrogate loss function所得到的2D-3D correspondence的效果不好的这个问题，很多端到端的方法被提了出来，其会将pose上的error直接反向传播到这些中间的representations上（也就是correspondences上）。[Learning less is more: 6d camera localizsation via 3d surface regression]提出了一个dense correspondence网络，其中3D points是可被学习的，[End-to-end learnable geometric vision by backpropagating pnp optimization]预测的是2D keypoints的位置，[Solving the blind perspective-n-point problem end-to-end with robust differentiable geometric optimization]学习一批没有顺序的2D/3D点集的correspondence weight矩阵，[Repose: Fast 6d object pose refinement via deep texture rendering]除了学习点之间的对应关系，还学习了feature和metric之间的对应关系。上述的这些所有的方法都使用了某种regularization loss，要么因为pose的non differentiable特性，上述方法的训练是不能收敛的。但在这篇文章所提出的概率框架下，上述的这些方法可以被看作一个Laplace approximation过程（见3.1）或者一个local regularization方法（见3.4）。

*Probabilistic Deep Learning*

最常见的一种引入概率框架从而使得端到端学习变得可能的就是为离散的分类问题，引入了Softmax层，这样one-hot的arg max结果就变成了连续的概率分布结果，这样就可以端到端的训练了（变得differentiable了）。


**3. Generalized End-to-End Probabilistic PnP**

**3.1 Overview**

给定一个object，我们的目标是来预测一个集合$$X = \lbrace x_i^{3D}, x_i^{2D}, w_i^{2D} \vert i=1,2,\cdots, N \rbrace$$，包含$$N$$个点，其中3D object坐标$$x_i^{3D} \in \mathbb{R}^3$$，2D图像坐标$$x_i^{2D} \in \mathbb{R}^2$$以及2D权重$$w_i^{2D} \in \mathbb{R}^2_{+}$$，然后一个PnP问题就可以建立在$$X$$上来预测这个相机角度下物体的姿态了。

一个PnP层的关键在于寻找到一个最优的姿态$$y$$（也就是一个rotation matrix $$R$$和一个translation vector $$t$$）来使得如下的cumulative squared weighted reprojection error最小：

$$\begin{equation}
\mathop{argmin}\limits_{y} \frac{1}{2} \sum_{i=1}^N \lvert w_i^2d \circ (\pi (R x_i^{3D} + t) - x_i^{2D} ) \rvert^2
\end{equation}

其中将$$w_i^2d \circ (\pi (R x_i^{3D} + t) - x_i^{2D} )$$定义为$$y_i(y) \in \mathbb{R}^2$$，$$\pi$$是由相机内参定义的projection function，$$\circ$$表示element-wise乘法，$$f_i(y)$$表示的是加权了的reprojection error。


### 26. [Self-Supervised Viewpoint Learning From Image Collections](https://openaccess.thecvf.com/content_CVPR_2020/papers/Mustikovela_Self-Supervised_Viewpoint_Learning_From_Image_Collections_CVPR_2020_paper.pdf)

[CODE](https://github.com/NVlabs/SSV)

*CVPR 2020*

**Abstract**

训练神经网络来预测object的viewpoint需要大量的标注数据。然而，手动标记viewpoints是很困难的，而且容易出错。而且，从网上来下载某个类别的object的图片是很容易的，比如说cars或者human faces。本文致力于研究是否可以直接从这些无标注的某个类别的object的图片来无监督的预测object的viewpoint。作者提出了一个新的学习框架，使用analysis-by-synthesis的思路来使用viewpoint aware的方式reconstruct输入的图片。作者表明对于多种物体类别，比如cars，human faces，buses，trains等，本文的方法和那些监督方法的效果差不多。

**1. Introduction**

从2D图片里学习3D信息是CV领域一个很基本的问题。object viewpoint（也就是azimuth，elevation和tilt角）的估计给3D geometry理解和2D图片之间建立了很重要的联系。在这篇文章里，作者尝试解决从单张图片估计object的viewpoint的问题。viewpoint的估计在3D geometry理解里扮演着核心角色，其对于object manipulation，3D reconstruction，image synthesis等都有很重要的影响。但是从单张2D图片估计object的viewpoint是一个很难的问题。基于学习的方法使用大量的有标注的训练数据来训练神经网络解决这个问题。但是这样的方法需要大量人工标注的数据，这是很难获取而且容易标错的。所以构造这样的一个大型数据集是很难的，而且费时费力，还容易出错。现存的基于学习的viewpoint估计方法要么就是基于这样大量的标注数据，要么就是基于合成的数据。

在本文里，作者提出了一个自监督的方法来解决这个问题，如fig1所示。

![1]({{ '/assets/images/SSV-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1.*

在本文的任务里，作者假设含有object的图片已经被检测到了，而且是tight bounded的（也就是说图片里基本上只有这样一个object，不受背景影响）。

本文所使用的方法仍然是analysis-by-synthesis（也就是还是auto-encoder），intermediate features是viewpoint。作者使用了一些约束来使得这个网络能够学习到可靠的viewpoint，包括encoder和decoder之间的cycle-consistency，viewpoint和appearance之间的disentanglement loss，object-specific symmetry prior等。据作者的认知，本文是第一篇使用自监督的方式从image collections in-the-wild来估计object的3D viewpoint的文章。

本文的贡献如下：

* 1. 提出了一个新的analysis-by-synthesis的框架来以自监督的方式进行viewpoint estimation。
* 2. 作者提出了generative，symmetric和adversarial约束来辅助这个auto-encoder能够获取viewpoint。
* 3. 作者在BIWI数据集上对head pose estimation进行了估计，并且在PASCAL3D+数据集上对cars，buses和trains的viewpoint进行了估计，和那些监督方法的效果差不多。


**3. Self-Supervised Viewpoint Learning**

**Problem Setup**

作者从某一类别object的in-the-wild的image collection $$\lbrace I \rbrace$$里学习到一个viewpoint estimation network $$\mathcal{V}$$，而不需要任何标注数据。因为viewpoint estimation需要有检测框的图片，所以作者假设每张图片里的物体已经被检测并分离好了，也就是说背景信息很少。在inference的时候，viewpoint network $$\mathcal{V}$$的输入是一个单张object image $$I$$，输出是object的3D viewpoint $$\hat{v}$$。


**Viewpoint representation**

作者使用三个Euler angles来表示一个object的viewpoint，也就是azimuth $$\hat{a}$$，elevation $$\hat{e}$$和in-plane rotation $$\hat{t}$$。为了表示这些Euler angle，作者使用的方法是将它们用一个2D coordinate来表示：比如说对于azimuth，$$a \in \left[0, 2 \pi \right]$$，表示为单位圆上的一个2D坐标：$$(cos(a),sin(a))$$。也就是说，对于每个角度，作者在第一象限内预测一个单位向量：$$\lvert \hat{a} \rvert = (\lvert cos(\hat{a}) \rvert, \lvert sin(\hat{a}) \rvert)$$，然后还预测它们的符号，$$sign(\hat{a})$$，一共有四种可能：$$(+,+),(+,-),(-,+),(-,-)$$。最后，通过计算$$tanh(sin(a)/cos(a))$$来获取这个角$$a$$。简而言之，这个viewpoint esitmation网络对于每个角，用regression来预测一个单位长度的向量，用classification来进行一个四分类。

**Approach overview and motivation**

作者使用一系列self-supervised loss来对这个viewpoint estimation network $$\mathcal{V}$$进行训练，如fig2所示。

![1]({{ '/assets/images/SSV-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. 使用的loss包括generative consistency，symmetry和discriminator losses。*

作者使用了三个不同的约束，分别叫做generative consistency，symmetry constraint和discriminator loss。generative consistency是本文的核心约束。这篇文章从generative model的角度来看decoder，将其称为synthesis function，也就是输入是viewpoint，输出是image，确实是一个生成模型。本篇文章使用的synthesis networks是从两个输入来生成图片的（和一般的auto-encoder还不一样）：$$v$$，用来控制object的viewpoint；和$$z$$，用来控制object的style。将encoder，viewpoint estimation network $$\mathcal{V}$$和decoder，synthesis network $$\mathcal{S}$$使用cycle-consistency约束（section3.1里说）进行联合训练。如果synthesis network $$\mathcal{S}$$能够在$$v$$和$$z$$的输入下得到拟真的图片，那么该图片就可以再作为$$\mathcal{V}$$的输入来得到viewpoint。而对于真实的图片，如果$$\mathcal{V}$$能给出正确的viewpoint $$v$$，那$$\mathcal{S}$$就可以产生和输入相类似的图片。作者还加入了一个discriminator来判定图片的真实性。同样的，object的symmetry被用来加强对网络的约束。


### 1. [DenseFusion: 6D Object Pose Estimation by Iterative Dense Fusion](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_DenseFusion_6D_Object_Pose_Estimation_by_Iterative_Dense_Fusion_CVPR_2019_paper.html)

*CVPR 2019*

### 2. [PVN3D: A Deep Point-Wise 3D Keypoints Voting Network for 6DoF Pose Estimation](https://openaccess.thecvf.com/content_CVPR_2020/html/He_PVN3D_A_Deep_Point-Wise_3D_Keypoints_Voting_Network_for_6DoF_CVPR_2020_paper.html)

*CVPR 2020*

### 3. [KeyPose: Multi-View 3D Labeling and Keypoint Estimation for Transparent Objects](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_KeyPose_Multi-View_3D_Labeling_and_Keypoint_Estimation_for_Transparent_Objects_CVPR_2020_paper.html)

*CVPR 2020*

### 4. [Learning Deep Network for Detecting 3D Object Keypoints and 6D Poses](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhao_Learning_Deep_Network_for_Detecting_3D_Object_Keypoints_and_6D_CVPR_2020_paper.html)

*CVPR 2020*

### 5. [Photo-Geometric Autoencoding to Learn 3D Objects from Unlabelled Images](https://arxiv.org/pdf/1906.01568.pdf)

*Arxiv 2019*

### 6. [Structured Domain Adaptation for 3D Keypoint Estimation](https://ieeexplore.ieee.org/abstract/document/8885979)

*3DV 2019*


### 7. [Gen6D: Generalizable Model-Free 6-DoF Object Pose Estimation from RGB Images](https://arxiv.org/pdf/2204.10776.pdf)

*Arxiv 2022*


### 8. [3D Sketch-Aware Semantic Scene Completion via Semi-Supervised Structure Prior](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_3D_Sketch-Aware_Semantic_Scene_Completion_via_Semi-Supervised_Structure_Prior_CVPR_2020_paper.html)

*CVPR 2020*




---
