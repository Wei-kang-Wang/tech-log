---
layout: post
comments: false
title: "NrSfM"
date: 2020-01-01 01:09:00
tags: paper-reading
---

> This post is a summary of Non-rigid Structure from Motion papers.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

## 1. [Piecewise Planar Hulls for Semi-supervised Learning of 3D Shape and Pose from 2D Images](https://arxiv.org/pdf/2211.07491.pdf)

*Arxiv 2022*

**Abstract**

作者研究了如何从单张2D图片，通过keypoints来估计3D shape和pose的问题。3D shape和pose是直接从按类划分的图片以及图片中物体的部分2D keypoints标注上获取的。在这篇文章里，作者首先提出了一个端到端的训练框架来进行intermediate 2D keypoints的获取以及最终的3D shape和pose的估计。再之后，上述这个框架再由这些intermediate 2D keypoints作为监督信号来训练。而且，作者还设计了一种半监督的训练方法，其可以从有标注和无标注的数据中都获益。为了能够使用无标住的数据，作者使用了piece-wise planar hull prior来为标准的object shape提供约束。这些planar hulls是借由keypoints来为每一类物体定义的（不是每一个）。一方面，所提出的框架学习如何从有标注的数据里来获取每类物体的planar hull。另一方面，框架也要求对于无标住的数据，其keypoints和hulls需要和有标注的那些保持consistent。这种consistency约束使得我们可以充分利用那些无标注的数据。本文所提出的方法和那些sota的监督方法所得到的结果效果差不多，但只需要使用一半的标注。



**1. Introduction**

使用预定义的keypoints来预测一个物体的shape和pose是一个很火的方向，也有着很广泛的应用，包括registration，recognition，generation等。除了其在人体姿态估计方面的应用以外，基于keypoints的representations在非人类的物体类别上也经常被使用，比如很多机器人和虚拟现实的应用需要3D shape和pose。

现有的估计3D shape和pose的方法（[Augmented autoencoders: Implicit 3d orientation learning for 6d object detection]()，[Discovery of latent 3d keypoints via end-to-end geometric reasoning]()，[Viewpoints and keypoints]()，[Learning deep network for detecting
3d object keypoints and 6d poses]()）使用了各种不同种类的监督信号，包括3D keypoints，pose或者multiple views。也有一些方法（[Optimal pose
and shape estimation for category-level 3d object perception]()，[In perfect shape: Certifiably optimal 3d shape reconstruction from 2d landmarks]()）对于每一类物体，使用某种3D template-matching来match 2D keypoints，但是它们对于有遮挡的情况效果很差。另一类方法（[Indoor Scene Understanding Using Non-conventional Cameras]()，[Pose estimation for augmented reality: a hands-on survey]()）直接从单张2D图片上来预测keypoints的3D位置，因此它们可以有着更宽的应用（相对于之前那种使用shape template的那些方法）。这些learning-based的方法的一个分支所使用的方法是对于每一类2D images的collections，其上标注有2D的keypoints annotations，然后训练一个模型，在inference的时候，对于每张输入的图片，输出其3D shape以及pose（也就是3D keypoints），这些方法叫做deep non-rigid structure-from-motion（NrSfM），因为输入是同一类object的不同相机角度的图片，而且因为是同一类而不是同一个物体，所以也是non-rigid transformation。

NrSfM方法可以被分为处理单一种类物体的（[Unsupervised 3d pose estimation with geometric self-supervision]()，[Deep non-rigid structure from motion]()，[Procrustean regression networks: Learning 3d structure of non-rigid objects from 2d annotations]()，[Pr-rrn: Pariwise-regularized residual-recursive networks for non-rigid structure-from-motion]()），和处理多个种类物体的（[C3dpo: Canonical 3d pose networks for non-rigid structure from motion]()，[Procrustean autoencoder for unsupervised lifting]()）。处理单一种类物体的方法会对于每个object种类训练一个模型，而处理多个种类物体的方法设计的模型对于多个object种类都有用，这样提高了训练和测试的效率。因此，本文的作者致力于研究如何开发一个deep NrSfM来进行端到端的直接从2D图片里得到3D shape的模型。目前现有的大多数方法认为2D keypoints extraction和将它们lift到3D的这两个过程是不同的（除了[Procrustean regression networks: Learning 3d structure of non-rigid objects from 2d annotations]()，但这篇文章只研究了human pose estimation这一个object类别）。

多种类的shape，pose估计任务需要大量的有2D keypoints标注的图片。绝大多数现有的NrSfM方法都会使用一个成熟的预训练好的2D keypoints detector（比如stacked-hourglass网络）来直接从2D keypoints上获取2D keypoints标注，作为输入数据使用。但这样的话，对于新的没有见过的物体种类，就需要更多的数据来训练这个2D keypoints detector，从而影响和阻碍了这些方法的应用范围。因此，semi-supervised方法就显得很有用了。现有的一些semi-supervised的NrSfM方法，要么就需要3D annotations（[Neural view synthesis and matching for semi-supervised few-shot learning of 3d pose]()），要么就需要pose annotations（[FS6D: few-shot 6d pose estimation of novel objects]()）。作者发现，现在deep NrSfM领域还没有semi-supervised的方法。在这篇文章里，作者提出了第一个deep NrSfM semi-supervised的框架，使用piece-wise planar hulls prior来对于每类物体，从输入的2D图片中，获取3D shape和pose信息。框架图如fig 1所示。

作者直接在shape template上定义piecewise planar hulls，这会被用来在图片上获取semantic regions。简单来说，planar hulls就是一个list，用来表示哪些keypoints应该被归为一个小团体来表示一个semantic region。这些semantic regions应该要能表示物体的3D surface上一块有意义的区域。这些semantic regions对于不同的图片，相对应的semantic region应该是consistent的。需要注意的是，对于每一类物体，planar hulls只需要定义一次（也就是不会对于每个物体都定义，只对于这一类物体定义），这样的标注相对来说比较省力。

这篇文章所提出的weak semi-supervised方法探索了（i）只使用部分有2D keypoints标注的图片来进行3D shape学习的问题；（ii）同一张图片的keypoints和planar hulls表示的semantic region之间的consistency作为一个重要的约束来帮助训练。

本篇文章的贡献总结如下：
* 作者提出了piecewise planar hulls的概念，其可以仅仅用keypoints就被定义。这些planar hulls表示的是物体的3D surface的semantic的区域。
* 作者提出了第一个deep NrSfM半监督框架，利用了planar hulls所表示的semantic regions和keypoints之间的consistency来提供约束。
* 本文的方法和监督方法的效果是差不多的，但仅仅需要在PASCAL3D+数据集上一半的数据有标注。


**2. Related Works**

NrSfM领域解决的问题是对于一类物体的2D图片以及2D keypoints，将这些keypoints lift到3D上，并且获取每张图片的viewpoints信息（参考[Trajectory space: A dual representation for nonrigid structure from motion]()，这是NrSfM的创始文章）。这个问题已经被广泛的进行过研究，包括：[Prior-less compressible structure from motion]()，[Local non-rigid structure-from-motion from diffeomorphic mappings]()，[Sparseness meets deepness: 3d human pose estimation from monocular video]()，[Complex non-rigid motion 3d reconstruction by union of subspaces]()。通过将同一个种类的不同物体的不同角度的照片当作同一个物体的不同角度的照片，deep NrSfM可以被用来从单张图片获取3D pose和shape（[Unsupervised 3d reconstruction networks]()，[Deep non-rigid structure from motoin]()，[Structure from category: A generic and prior-less approach]()）。

有一些NrSfM方法以meshes的方法表示3D shape，输入仍然是图片，输出是meshes，比如[Learning category-specific mesh reconstruction from image collections]()和[To the point: Correspondence-driven monocular 3d category reconstruction]()。尽管最近的NrSfM方法可以从同一类物体的多个角度的照片来获取non-rigid meshes，但它们所能够处理的物体的种类很有限，比如人脸（[Self-supervised multi-view synchronization learning for 3d pose estimation]()，[Lifting autoencoders: Unsupervised learning of a fully-disentangled 3d morphable model using deep non-rigid structure from motion]()，[Unsupervised learning of probably symmetric deformable 3d objects from images in the wild]()）。C3DPO（[C3DPO: canonical 3d pose networks for non-rigid structure from motion]()）通过将一类物体的canonical shape和viewpoint解耦来学习很大一类物体的3D shape和pose。[Procrustean regression networks: Learning 3d structure of non-rigid objects from 2d annotations]()使用procrustean regression（[Procrustean regression: A flexible alignment-based framework for nonrigid structrue estimation]()）来约束学习shape和pose的过程。尽管这些方法也可以从2D图片端到端的直接输出3D keypoints，但它们处理的物体类别都是人脸，而且不能解决occlusion的问题。[Unsupervised 3d pose estimation with geometric self-supervision]()利用cycle-consistency同样也研究了human pose estimation的问题。最近，[Procrustean autoencoder for unsupervised lifting]()将Procrustean regression方法和autoencoders相结合提出了新的方法，但是其输入是图片的2D keypoints的位置而不是图片，从而需要一个预训练好的2D keypoint detector，比如说Hourglass network。

其它的一些相关工作包括segmentation领域的半监督方法（和NrSfM关系不大，就不说了）。


**3. Keypoints and Planar Hulls for Weak and Semi-supervision**

本文的目标是，仅仅从一类物体的一张图片，获取其3D structure（这个structure是由3D keypoints所表示的，也就是获取3D keypoints）。这个任务在NrSfM这个setting里已经被研究了很久了。现有的方法（[C3dpo: Canonical 3d pose networks for non-rigid structure from motion]()，[Procrustean regression: A flexible alignment-based framework fro nonrigid structure from motion]()，[Procrustean autoencoder for unsupervised lifting]()）需要2D keypoints标注作为weak supervision。然而，获取这些2D keypoints标注费时费力，从而限制了将这些方法扩展到新的object类别的能力。

本文的作者提出了一个新的框架，只需要利用一小部分的有2D keypoints标注的图片就可以和那些有着全部2D keypoints标注的方法差不多的效果。本文的方法依赖于一种precewise planar hulls的prior，其是对于每一类物体进行定义的。planar hulls prior直接在shape template上定义，之后会被用来在图片上获取semantic region（也就是和segmentation也有了联系）。这些semantic regions和keypoints之间的consistency被用作一种监督信号来辅助网络的训练。

本文的weak，semi-supervised框架使用了（i）部分有标注的图片（ii）keypoints和planar hulls表示的semantic regions之间的consistency。

**3.1 Preliminaries - NrSfM**

在NrSfM的设定里，一个物体是用一系列keypoints来表示的（3D的keypoints）。给定这些keypoints在$$n$$个viewpoints下的2D projections，问题的目标是获取这些viewpoints下的2D keypoints的3D信息（也就是将其lift到3D）。以$$Y_i = \left[ y_{i1}, \cdots, y_{ik} \right] \in \mathbb{R}^{2 \times k}$$表示第$$i$$个viewpoint的$$k$$个2D keypoints的坐标，其是一个matrix，每列表示一个keypoints坐标。第$$i$$个viewpoint下的物体的structure是$$X_i = \alpha_i^T B$$，其中shape basis $$B \in \mathbb{R}^{d \times 3k}$$，coefficients $$\alpha_i \in \mathbb{R}^d$$。我们假设keypoints是centered以及normalized的，并且相机模型是orthographic projection model，也就是$$\Pi = \left[ I_{2 \times 2}, 0 \right]$$。给定相机的旋转矩阵$$R_i \in SO(3)$$，以及已经centered和normalized的keypoints（也就是$$X_i$$），我们就有：$$Y_i = \Pi R_i (I_3 \odot X_i) = \Pi R_i (I_3 \odot \alpha_i^T B)$$，其中$$I_3 \odot b$$表示将一个行向量$$b \in \mathbb{R}^{1 \times 3k}$$转换为一个大小为$$\mathbb{R}^{3 \times k}$$的矩阵。从而，如下的loss就可以被定义了：

$$\min\limits_{\alpha_i, B, R_i \in SO(3)} \sum\limits_{i=1}^{n} L(Y_i, \Pi R_i (I_3 \odot \alpha_i^T B))$$

其中$$L(a,b)$$是某种表示距离的loss。在multi-class NrSfM的方法里，其需要具有能够表达多种类别的物体的structure的能力，所以$$I_3 \odot \alpha_i^T B$$应该要能够表示具有不同数量的keypoints的不同物体的3D structure的能力。用$$Z$$表示物体类别集，$$z_i \in Z$$表示物体$$i$$所属的类别。对于类别$$z \in Z$$来说，我们用$$k_z$$个keypoints来表示这一类的物体，从而我们就一共需要定义$$k = \Sigma_z k_z$$个keypoints。为了对于每一类物体，找到其对应的那些keypoints，我们就需要一个subset selection向量$$\zeta_z \in \lbrace 0, 1 \rbrace^k$$来表示哪些keypoints是类别$$z$$对应的。从而，我们上述的NrSfM的目标函数在multi-class下就变成了：

$$\min\limits_{\alpha_i, B, R_i \in SO(3)} \sum\limits_{i=1}^{n} L(Y_i \circ \zeta_{z_i}, \Pi R_i (I_3 \odot \alpha_i^T B) \circ \zeta_{z_i})$$

在这片文章里，作者使用的是deep NrSfM框架，从而$$\alpha, B, R$$都是神经网络的输出。因此，就将$$R(\alpha^T B)$$简写为$$\hat{X}$$，也就是在相机坐标系下的3D shape。


**3.2 Motivation**

在本文的设定下，我们有$$N_L$$张有2D keypoints标注的图片，$$N_U$$张没有2D keypoints标注的图片。将有标注的和无标注的图片分别叫做$$D_L$$和$$D_U$$。ground truth的2D keypoints叫做$$\bar{Y_{D_L}}$$。本文的目标是直接从图片来预测3D keypoints，将有标注的图片的3D keypoints预测结果记为$$\hat{X_{D_L}}$$，将无标注的图片的3D keypoints的预测结果记为
$$\hat{X_{D_U}}$$，基于上述的公式(2)，我们的损失函数就是：

$$\min\limits_{\alpha_i, B, R_i \in SO(3)} \sum\limits_{i=1}^{N_L} L(\bar{Y_{D_L}^{i}} \circ \zeta_{z_i}, \Pi \hat{X_{D_L}^{i}} \circ \zeta_{z_i}) + \sum\limits_{i=1}^{N_U} L(\bar{Y_{D_U}^{i}} \circ \zeta_{z_i}, \Pi \hat{X_{D_U}^{i}} \circ \zeta_{z_i})$$

但上述公式存在的问题是，我们并没有$$\bar{Y_{D_U}}$$。因此，我们就需要想办法对于无标注的那些图片，获取2D keypoints的假标签来代替$$\bar{Y_{D_L}}$$。为了达成这个目标，作者提出了一个代理任务：预测semantic planar hulls。

总体来说，本文所提出的模型，以图片作为输出，输出有三部分：（i）一个原图片输入的segmentation；（ii）2D keypoints的locations；（iii）2D keypoints的对应的3D信息。将上述模型表示为：$$T(\textit{I}) = (S,Y,X)$$，其中$$\textit{I} \in \mathbb{R}^{H \times W \times 3}$$是输入的图片，$$S \in \mathbb{R}^{H \times W \times s}$$是segmentation mask的logits（也就是分成$$s$$个部分），$$Y \in \mathbb{R}^{k \times 2}$$是2D keypoints的坐标，$$X \in \mathbb{R}^{k \times 3}$$是3D keypoints的坐标。模型里的segmentation branch的输入是3D piecewise planar hulls的2D projections，而不是原图片。segmentation的结果会和2D以及3D keypoints预测结果结合起来用来预测$$\bar{Y_{D_U}}$$。这样就可以让我们利用起来那些无标注的数据，进行这种半监督的学习。

**3.3 Piecewise Planar Hulls**

本文基于2D keypoints提出了一种新的结构，Piecewise Planar Hulls（PPH），其数据格式是一个list，里面的每个元素都是一系列能够联合表示某种语义信息的2D keypoints的集合。也就是说，PPH里的每个元素都是若干个keypoints表示的一个plane。这些planes的union就构成了一个包含了所有keypoints的3D hull。因为这些planes是通过semantics定义的，PPH只需要对每一类进行定义（而不是每一类的每一个物体），如fig 3所示。

>所以还是需要对每一类先手动定义这样的一个PPH

可以有多种方法来为每一类物体定义PPH，但是需要满足：（i）每个keypoint至少要在一个plane里出现；（ii）除了可能的公共边以外，任意两个plane都不应该有别的交集（注意这里指的是3D空间内的交集，它们的2D projections当然可以有交集）。PPH理想状态下应该表示这类物体的每个有语义信息的平面。PPH定义的好坏对于后续segmentation network的效果好坏有着重要的影响。对于大多数类别来说，PPH的定义是不困难的，因为keypoints本身就包含了语义信息了，对于绝大多数物体类别来说，PPH的定义实际上已经被keypoints的语义信息所基本确定了。在本文里，除了要包含语义信息，PPH还需要是满足条件的volume最小的那些平面。

另一个重点是对称的planes。比如说，car的左右两侧planes就是对称的，而且是相同的。但是，理论上我们是需要区分这两个平面的，比如通过汽车的前挡风平面，我们就可以区分左右两个车身平面。为了让模型能够学习到这点，作者使用了Coordiante convolutions。

PPH可以被用来产生segmentation。给定了2D keypoints坐标之后，就可以按照预先设定的这一类的PPH来找到这张图片的PPH结果。对于有标注的图片，既然有了2D keypoints的位置了，那么根据这些keypoints位置就可以定义PPH里的planes，根据网络预测的3D keypoints结果，就可以根据这些3D keypoints也找到PPH的那些planes。对于无标注的图片，3D的PPH planes是一样的，而2D的PPHD planes则根据网络给出的2D keypoints结果来获得。从而，对于任意一张图片（不管有无标注），现在我们都有了两个PPH：2D的和3D的。因为每一类物体的PPH是预先设定的，segmentation结果也可以是预先设定好的，从而就可以用上述的结果来训练这个segmentation分支（segmentation要分成多少部分，也就是PPH有多少个平面，这是个超参数）。将物体类别$$z$$的PPH的planes个数表示为$$s_z$$，那么segmentation分支的分割部分数量就是$$s = \Sigma_{z} s_z + 1$$，其中多出来的那个表示的是背景。


**3.4 Cross Consistency between Keypoints and Planar Hulls**

为了能够使用那些无标注的数据，作者研究了keypoints和planar hulls的语义之间的consistency。这是通过从网络的输出结果中交替的获取segmentation假标签$$\bar{S_{D_U}}$$以及2D keypoints假标签$$\bar{Y_{D_U}}$$来实现的。作者提出了两个模块：（i）2D keypoints假标签的生成；（ii）语义假标签的生成（也就是segmentation假标签）。这些假标签就可以用来self-supervise网络的训练。具体这个流程在下面介绍。


**4. Psudo-label Generation and Semi-supervised Learning**

**4.1 Semantic Psedo-label Generation**

假设模型$$T$$是由有标签的数据来训练的。那么segmentation mask branch就可以用由2D keypoints位置得到的ground truth planes来约束，2D keypoint detection branch就可以用ground truth的2D keypoint locations来约束，3D keypoint branch就可以用之前所说的公式2的reprojection loss来约束。因此，整个网络在有了ground truth 2d keypoint locations之后就可以进行训练了。

但如果我们需要利用无标签的数据也参与训练的话，就需要在segmentation mask branch的训练上使用假标签。具体来说，每个像素点都有$$s+1$$种分类可能（$$s+1$$类），表示的是$$s$$个planes和背景。

* Monte Carlo Dropout
Monte Carlo Dropout是一种被广泛使用的度量不确定性的方法。具体来说，在模型$$T$$里，我们在segmentation branch里使用dropout，然后运行整个网络$$N_D$$次，dropout的概率是$$p_D$$。从而对于segmentation branch的输出，我们就获得了一个大小为$$N_D \times H \times W \times s$$的矩阵，叫做logits matrix，记为$$R_D$$。然后使用Welch's t-test对于每个像素点都进行一波处理，最终得到每个像素点的假标签。

* Visibility
网络预测的3D keypoint可以用来建立plane visibility。

* Plane Estimation Agreement
网络预测的2D keypoint也可以用来建立planes。



## 2. [C3DPO: Canonical 3D Pose Networks for Non-Rigid Structure From Motion](https://openaccess.thecvf.com/content_ICCV_2019/papers/Novotny_C3DPO_Canonical_3D_Pose_Networks_for_Non-Rigid_Structure_From_Motion_ICCV_2019_paper.pdf)

*ICCV 2019*

[[CODE](https://github.com/facebookresearch/c3dpo_nrsfm)]
[[PAGE](https://research.facebook.com/publications/c3dpo-canonical-3d-pose-networks-for-non-rigid-structure-from-motion/)]

这篇文章解决的问题是NrSfM，输入是同一个物体的不同角度的views的2D keypoints annotations，也就是一个$$2 \times K$$的矩阵，$$K$$是超参数，keypoints的数量，输出是该物体的3D keypoints，也就是3D shape，大小为$$3 \times K$$。输入并不是RGB图片。

这篇文章的亮点在于，作为2019年的文章，还处于使用deep learning解决NrSfM问题的中期，在现在来看方法并不复杂，但好的效果和好的可视化整体来看是不错的。文章的主要想法是要将由物体的rigid motion transformation导致的2D keypoints不同，与物体形变（比如人体视频不同帧因为动作变化导致的物体形变）导致的不同区分开。文章是通过引入两个loss来解决这个问题。

第一个loss很显然，网络在以2D keypoints matrix作为输出后，并不是直接输出3D keypoint matrix，而是输出一个basis $$S \in \mathbb{R}^{3D \times K}$$，一个依赖于输入的coefficient vector $$\alpha \in \mathbb{R}^{1 \times D}$$，然后将3D matrix用这个basis的线性组合来表示：$$X = (\alpha \bigotimes I_3)S$$，其中$$\bigotimes$$是Kronecker product，而$$D$$是超参数。再然后，将这个3D matrix $$X$$ 经过rotation $$R$$之后再project到2D上，和输入的2D ground truth进行比较，计算loss。

上述有几个技术性细节：
* rotation matrix $$R$$和coefficient $$\alpha$$均为网络的输出，basis $$S$$是网络参数
* 在预处理数据的时候就将2D keypoint的x和y维度分别进行了zero-center处理，并且整体乘上了一个scalar，使得variance较大的那个轴（x或y）的数值范围大概位于-1到1之间。这样的话，就可以在计算transformation的时候不用考虑translation了。
* 在计算loss的时候，计算的是两个matrix或者vector之间的loss，用的不是一般的loss，而是humber loss，暂时还不知道为什么要这样。
* 输入不仅仅有$$Y$$，实际上还有每个keypoint是否visible的flag vector $$v$$，在计算loss的时候，这些$$v$$就乘以每个keypoints，也就是说，不可见的就不算在内。

第二个loss是作者为了使得网络认为所有的只经过rigid body transformation后的shape都应该等价所加上的。具体做法是，再设计一个网络$$\Psi$$，前一个网络叫做$$\Phi$$，对于任何一个2D keypoint matrix $$Y$$输入，$$\Phi$$输出了$$\alpha$$和$$R$$，以及$$S$$，从而计算出了3D matrix $$X$$。对于网络$$\Psi$$，先随机采样一个rotation matrix $$R^{'}$$，然后将$$R^{'}X$$输入$$\Psi$$，输出$$\alpha^{'}$$（注意，并不是直接输出3D shape）。然后，结合$$S$$，得到了一个新的3D shape $$X^{'} = (\alpha^{'} \bigotimes I_3) S$$，新的loss就是$$X^{'}$$和$$X$$之间的距离。

最后还有个可有可无的loss，也就是还可以在plane内加上rotation，也就是说不是对于$$X$$加上3D rotation matrix，而是直接对于$$Y$$加上2D rotation matrix，这是用来使得网络$$\Phi$$更robust的，可以理解为一种data augmentation。

流程图如下：

![C3DPO-1]({{ '/assets/images/C3DPO-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}


**技术细节**

* 网络并不是直接输出的rotation matrix，而是输出了一个长度为3的向量，然后经过hat operator和matrix exponential计算，得到了rotation matrix
* hat operator: https://en.wikipedia.org/wiki/Hat_operator
* matrix exponential: https://en.wikipedia.org/wiki/Matrix_exponential



## 3. [Procrustean Regression Networks: Learning 3D Structure of Non-Rigid Objects from 2D Annotations](https://arxiv.org/pdf/2007.10961.pdf)

*ECCV 2020*

[[CODE](https://github.com/sungheonpark/PRN)]

**line of research**

[C3DPO](https://openaccess.thecvf.com/content_ICCV_2019/papers/Novotny_C3DPO_Canonical_3D_Pose_Networks_for_Non-Rigid_Structure_From_Motion_ICCV_2019_paper.pdf) $$\longrightarrow$$ [PRN](https://arxiv.org/pdf/2007.10961.pdf)

[EM-PND](https://openaccess.thecvf.com/content_cvpr_2013/papers/Lee_Procrustean_Normal_Distribution_2013_CVPR_paper.pdf) $$\longrightarrow$$ [PR](https://ieeexplore.ieee.org/document/8052164) $$\longrightarrow$$ [PRN](https://arxiv.org/pdf/2007.10961.pdf)

其中，PRN是基于C3DPO的想法，也想要将rigid motion和object本身形变造成的2D keypoints不同给分离开，C3DPO专门设计了一个新的网络$$\Psi$$来解决这个，而PRN则是通过将每个shape都用Generalized procrustes analysis align到一起叠成一个matrix，然后对这个matrix进行约束实现的（比如说进行nuclear norm约束其rank等）。而PRN的这个使用GPA进行align的想法则是源于PR这篇论文，而PR的想法则是源于EM-PND。EM-PND是希望使用procrustean distribution来表示这些aligned的shapes，而使用EM算法来对参数进行学习。

所以说，PRN的核心就是使用某种在aligned shapes上的regularization term来替代C3DPO里的canonicalization网络，而这些aligned shapes则是通过GPA计算得来。在PRN里，procrustean analysis的reference shape是aligned的shape的平均值，而在PR里这也是个可学习的参数。

**技术细节**

* 对于任意一个3D shape $$X_i \in \mathbb{R}^{3 \times n_p}$$，和reference shape，$$\bar{X}$$，aligned所使用的rotation matrix是这样计算得来的：$$R_i = \mathop{argmin}\limits_{R} \lVert RX_iT - \bar{X} \rVert$$，其中$$R_i^T R = I$$，$$T = I_{n_p} - \frac{1}{n_p} 1_{n_p} 1_{n_p}^T$$是translation matrix，用于将shape $$X_i$$center到origin上。这里的$$T$$的用法可以被借鉴。而aligned的shape就是$$\tilde{X_i} = R_i X_i T$$。
* PRN和PR这两篇文章都花了大量的篇幅证明上述网络设计的每个部分都是differentiable的（计算出来了loss对于$$X_i$$和reference shape $$\bar{X}$$的导数），所以说GPA也可以被放在可学习的框架内。
* PRN相对于C3DPO还有个创新就是，其的输入既可以是和C3DPO一样，是2D keypoint matrix，也可以是RGB图片，分别使用MLP和CNN来作为网络框架。













## 5. [Deep Non-Rigid Structure from Motion](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kong_Deep_Non-Rigid_Structure_From_Motion_ICCV_2019_paper.pdf)

*ICCV 2019*


## 6. [Lifting autoencoders: Unsupervised learning of a fully-disentangled 3d morphable model using deep non-rigid structure from motion](https://openaccess.thecvf.com/content_ICCVW_2019/papers/GMDL/Sahasrabudhe_Lifting_AutoEncoders_Unsupervised_Learning_of_a_Fully-Disentangled_3D_Morphable_Model_ICCVW_2019_paper.pdf)

*ICCV Workshop 2019*

[[POST](https://msahasrabudhe.github.io/projects/lae/)]














