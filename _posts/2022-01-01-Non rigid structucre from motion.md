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

## Abstract

作者研究了如何从单张2D图片，通过keypoints来估计3D shape和pose的问题。3D shape和pose是直接从按类划分的图片以及图片中物体的部分2D keypoints标注上获取的。在这篇文章里，作者首先提出了一个端到端的训练框架来进行intermediate 2D keypoints的获取以及最终的3D shape和pose的估计。再之后，上述这个框架再由这些intermediate 2D keypoints作为监督信号来训练。而且，作者还设计了一种半监督的训练方法，其可以从有标注和无标注的数据中都获益。为了能够使用无标住的数据，作者使用了piece-wise planar hull prior来为标准的object shape提供约束。这些planar hulls是借由keypoints来为每一类物体定义的（不是每一个）。一方面，所提出的框架学习如何从有标注的数据里来获取每类物体的planar hull。另一方面，框架也要求对于无标住的数据，其keypoints和hulls需要和有标注的那些保持consistent。这种consistency约束使得我们可以充分利用那些无标注的数据。本文所提出的方法和那些sota的监督方法所得到的结果效果差不多，但只需要使用一半的标注。



## 1. Introduction

使用预定义的keypoints来预测一个物体的shape和pose是一个很火的方向，也有着很广泛的应用，包括registration，recognition，generation等。除了其在人体姿态估计方面的应用以外，基于keypoints的representations在非人类的物体类别上也经常被使用，比如很多机器人和虚拟现实的应用需要3D shape和pose。

现有的估计3D shape和pose的方法（[Augmented autoencoders: Implicit 3d orientation learning for 6d object detection]，[Discovery of latent 3d keypoints via end-to-end geometric reasoning]，[Viewpoints and keypoints]，[Learning deep network for detecting
3d object keypoints and 6d poses]）使用了各种不同种类的监督信号，包括3D keypoints，pose或者multiple views。也有一些方法（[Optimal pose
and shape estimation for category-level 3d object perception]，[In perfect shape: Certifiably optimal 3d shape reconstruction from 2d landmarks]）对于每一类物体，使用某种3D template-matching来match 2D keypoints，但是它们对于有遮挡的情况效果很差。另一类方法（[Indoor Scene Understanding Using Non-conventional Cameras]，[Pose estimation for augmented reality: a hands-on survey]）直接从单张2D图片上来预测keypoints的3D位置，因此它们可以有着更宽的应用（相对于之前那种使用shape template的那些方法）。这些learning-based的方法的一个分支所使用的方法是对于每一类2D images的collections，其上标注有2D的keypoints annotations，然后训练一个模型，在inference的时候，对于每张输入的图片，输出其3D shape以及pose（也就是3D keypoints），这些方法叫做deep non-rigid structure-from-motion（NrSfM），因为输入是同一类object的不同相机角度的图片，而且因为是同一类而不是同一个物体，所以也是non-rigid transformation。

NrSfM方法可以被分为处理单一种类物体的（[Unsupervised 3d pose estimation with geometric self-supervision]，[Deep non-rigid structure from motion]，[Procrustean regression networks: Learning 3d structure of non-rigid objects from 2d annotations]，[Pr-rrn: Pariwise-regularized residual-recursive networks for non-rigid structure-from-motion]），和处理多个种类物体的（[C3dpo: Canonical 3d pose networks for non-rigid structure from motion]，[Procrustean autoencoder for unsupervised lifting]）。处理单一种类物体的方法会对于每个object种类训练一个模型，而处理多个种类物体的方法设计的模型对于多个object种类都有用，这样提高了训练和测试的效率。因此，本文的作者致力于研究如何开发一个deep NrSfM来进行端到端的直接从2D图片里得到3D shape的模型。目前现有的大多数方法认为2D keypoints extraction和将它们lift到3D的这两个过程是不同的（除了[Procrustean regression networks: Learning 3d structure of non-rigid objects from 2d annotations]，但这篇文章只研究了human pose estimation这一个object类别）。

多种类的shape，pose估计任务需要大量的有2D keypoints标注的图片。绝大多数现有的NrSfM方法都会使用一个成熟的预训练好的2D keypoints detector（比如stacked-hourglass网络）来直接从2D keypoints上获取2D keypoints标注，作为输入数据使用。但这样的话，对于新的没有见过的物体种类，就需要更多的数据来训练这个2D keypoints detector，从而影响和阻碍了这些方法的应用范围。因此，semi-supervised方法就显得很有用了。现有的一些semi-supervised的NrSfM方法，要么就需要3D annotations（[Neural view synthesis and matching for semi-supervised few-shot learning of 3d pose]），要么就需要pose annotations（[FS6D: few-shot 6d pose estimation of novel objects]）。作者发现，现在deep NrSfM领域还没有semi-supervised的方法。在这篇文章里，作者提出了第一个deep NrSfM semi-supervised的框架，使用piece-wise planar hulls prior来对于每类物体，从输入的2D图片中，获取3D shape和pose信息。框架图如fig 1所示。
