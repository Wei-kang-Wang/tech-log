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

* 在C3DPO这篇文章里，其
