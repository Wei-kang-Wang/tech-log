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


## \[**ICCV 2019**\] [C3DPO: Canonical 3D Pose Networks for Non-Rigid Structure From Motion](https://openaccess.thecvf.com/content_ICCV_2019/papers/Novotny_C3DPO_Canonical_3D_Pose_Networks_for_Non-Rigid_Structure_From_Motion_ICCV_2019_paper.pdf)

[[CODE](https://github.com/facebookresearch/c3dpo_nrsfm)]

这篇文章解决的问题是NrSfM，输入是同一个物体的不同角度的views的2D keypoints annotations，也就是一个$$2 \times K$$的矩阵和一个$$1 \times K$$的0，1矩阵用来表示keypoints的visibility，$$K$$是超参数，keypoints的数量，输出是该物体的3D keypoints，也就是3D shape，大小为$$3 \times K$$。输入并不是RGB图片。

> 注意，visibility矩阵非常重要，其在两个地方都有作用：1）其是作为factorization network $$\phi$$的输入的。具体来说，网络首先将keypoint矩阵和visibility矩阵相乘（从而invisible的那些列就是0了），再将乘积后的矩阵和visibility矩阵align，得到$$3 \times K$$的矩阵输入给$$\phi$$得到feature，再进行后续的shape coefficient和viewpoint的预测。2）在reprojection loss里，只计算那些可见keypoint的projected 2d keypoints和gt keypoints之间的差异。

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

![C3DPO-1]({{ '/assets/images/C3DPO-1.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}


> 注意，网络并不是直接输出的rotation matrix，而是输出了一个长度为3的向量，然后经过hat operator和matrix exponential计算，得到了rotation matrix。参考[hat operator](https://en.wikipedia.org/wiki/Hat_operator)，[matrix exponential](https://en.wikipedia.org/wiki/Matrix_exponential)



## \[**ICCV 2019**\] [Deep Non-Rigid Structure from Motion](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kong_Deep_Non-Rigid_Structure_From_Motion_ICCV_2019_paper.pdf)



## \[**ECCV 2020**\] [Procrustean Regression Networks: Learning 3D Structure of Non-Rigid Objects from 2D Annotations](https://arxiv.org/pdf/2007.10961.pdf)

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



