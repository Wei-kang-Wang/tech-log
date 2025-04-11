---
layout: post
comments: false
title: "Viewpoint Estimation"
date: 2020-01-01 01:09:00
tags: paper-reading
---

> 这个post将会介绍各种viewpoint estimation方法，和pose estimation是不同的，前者相当于是对相机外参的估计，而后者是对skeleton的估计，比如human pose estimation就是对人的3D/2D关键点进行估计。


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

* 在NeRF--这篇文章里，基于NeRF，其做出的改进就是，NeRF需要知道相机内参，以及每张view对应的相机pose，也就是相机外参，包括相机的位置（translation决定）以及角度（rotation决定），而NeRF--认为这个要求太苛刻了（确实要求过高），于是将rotation和translation都综合到网络里来学习，其办法就是直接让网络以图片为输入，学习translation和rotation，具体来说，对于translation，直接学习一个长度为3的vector就行了，而对于rotation matrix，因为rotation matrix自身要求位于$$SO(3)$$空间内，所以作者使用Rodrigues公式来用向量表示rotation matrix，具体来说就是让网络输出一个长度为3的向量即可，其与rotation matrix一一对应。最后，对于相机内参，作者假设$$f_x = H/2, f_y=W/2$$，其中$$H,W$$分别是图片的长宽，从而就只需要估计相机的focal length就行了，这也是网络的输出，作者还要求所有的输入图片都是同一个相机拍摄的。这个方法可以得到不错的结果，但是有一个约束就是，输入图片只能有很小的transformation，也就是说rotation和translation都被限定在了一个很小的范围内，这是符合认知的，因为大角度的旋转或者大的translation是很难直接通过这种方式学到的，unsuper3d那篇文章其实也只对那些拥有小的rotation和translation的数据集进行了实验，它也解决不了大transformation的问题，比如说Human3.6m这种数据集。实际上如何进行变化很大的viewpoint的估计还是个开放性问题，magicpony也有提到。

* 在C3DPO这篇文章里，其解决的问题是NrSfM，输入是同一个物体的不同角度的views的2D keypoints annotations，也就是一个$$2 \times K$$的矩阵，$$K$$是超参数，keypoints的数量，输出是该物体的3D keypoints，也就是3D shape，大小为$$3 \times K$$。输入并不是RGB图片。作为2019年的文章，还处于使用deep learning解决NrSfM问题的中期，在现在来看方法并不复杂，但好的效果和好的可视化整体来看是不错的。文章的主要想法是要将由物体的rigid motion transformation导致的2D keypoints不同，与物体形变（比如人体视频不同帧因为动作变化导致的物体形变）导致的不同区分开。文章是通过引入两个loss来解决这个问题。

第一个loss很显然，网络在以2D keypoints matrix作为输出后，并不是直接输出3D keypoint matrix，而是输出一个全局的basis $$S \in \mathbb{R}^{3D \times K}$$，一个依赖于输入的coefficient vector $$\alpha \in \mathbb{R}^{1 \times D}$$，然后将3D matrix用这个basis的线性组合来表示：$$X = (\alpha \bigotimes I_3)S$$，其中$$\bigotimes$$是Kronecker product，而$$D$$是超参数。再然后，将这个3D matrix $$X$$ 经过rotation $$R$$之后再project到2D上，和输入的2D ground truth进行比较，计算loss。

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

技术细节
* 网络并不是直接输出的rotation matrix，而是输出了一个长度为3的向量，然后经过hat operator和matrix exponential计算，得到了rotation matrix
* hat operator: https://en.wikipedia.org/wiki/Hat_operator
* matrix exponential: https://en.wikipedia.org/wiki/Matrix_exponential
