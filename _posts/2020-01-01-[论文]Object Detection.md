---
layout: post
comments: false
title: "[论文]Object Detection"
date: 2021-11-29 01:09:00
tags: paper-reading
---

> This post is a summary of object detection papers.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

## [Object Detection in 20 years: A survey](https://arxiv.org/pdf/1905.05055.pdf?fbclid=IwAR0pD4GD6zW2n3G_lbYUDFn9wFL1ECuZHJfZc3-6-Xq_gYSO5xx2mgpKueA)

*Zhengxia Zou, Zhenwei Shi, Yuhong Guo, Jieping Ye*

*Arxiv 2019*

### Abstract

目标检测，作为CV领域最基础也是最具有挑战性的一个任务，在过去的很多年里受到了非常多的关注。其在过去二十年的发展可以被看作是CV历史的缩影。如果我们把现在的object detection任务看作是利用强大的deep learning的技术美学，那么将眼光放到二十年前，我们将会看到冷兵器时代的智慧。这篇文章十分广泛的包含了超过400篇object detection的文章，时间跨度从1990到2009，有1/4个世纪。这篇文章会包含很多的话题，包括历史上的milestone detectors，detection datasets，metrics，detection systems的基础的building blocks，speed up techniques，以及最近sota的detection methods。这篇文章同时也包含了一些重要的detection应用，比如说pedestrian detection，face detection，text detection等等，并且对它们的难点以及近些年技术上的突破进行了深度分析。


### 1. Introduction

object detection是一个重要的CV任务，其解决的是在图片里对于某个特定的类别的视觉目标的实例进行检测，类别包括humans，animals，cars等。object detection的目标是开发出模型来为CV应用提供一个最为基本的信息：what objects are where?

作为CV领域一个最基本的问题之一，object detection为很多其它的CV任务提供了基础，比如说instance segmentation，image captioning，object tracking等。从应用的角度来说，object detection可以被分为两种研究发现，general object detection和detection applications，前一种致力于模仿人类的视觉和意识来用一个普适的模型对不同的objects实例都可以进行检测，而后者则是致力于在某种特定的应用场景下的检测比如pedestrian detection，face detection，text detection等。在最近一些年里，deep learning的快速发展为object detection注入了新鲜的血液，带来了瞩目的成果并且将object detection推到了研究的热点。object detection现在在很多现实生活的应用中都得到了使用，比如说autonomous driving，robot vision，video surveillance等。Fig 1显示了在过去二十年间有关object detection的论文的数量，可以看到明显的增长趋势。


![zz]({{ '/assets/images/SURVEY-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. 从1998年到2018年object detection论文的数量趋势。*

**Difficulties and Challenges in Object Detection**

尽管人们经常问，object detection的难点在哪？实际上这个问题并不好回答，并且可能被过于宽泛化了。不同的检测任务可能有完全不同的目标和约束条件，它们的难点也会不同。除了CV任务的常见的难点，比如说在不同角度下的objects，光照，以及类内变换，object detection的难点包括但不限于以下几方面：object rotation和scale在变化（比如说，很小的objects），精确的object localization很难，dense以及occluded object detection，speed up of detection等。在第4和第5章里，我们将会给这些问题更详细的描述。


### 2. Object Detection in 20 years

在这一章里，我们将会从多个角度回顾object detection的发展历程，包括milestone detectors，object detection datasets，metrics以及关键techniques的发展。

#### 2.1 A Road Map of Object Detection

在过去的20年里，大家都普遍同意object detection的发展主要可以分为两个历史阶段，traditional object detection阶段（2014年以前）和deep learning based detection阶段（2014年以后），正如Fig 2所示。

![important]({{ '/assets/images/SURVEY-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. object detection发展的历程图。milestone detectors也被标注在图里：VJ Detectors，HOG Detectors，DPM，RCNN，SPPNet，Fast RCNN，Faster RCNN，YOLO，SSD，Pyramid Network，Retina-Net。*


##### 2.1.1 Milestones: Traditional Detectors

如果你认为今天的object detection是deep learning强大能力下的技术美学，那么回到20年前我们就可以看到冷兵器时代的设计智慧。绝大多数早期的object detection算法都是基于handcrafted features来设计的。因为那个时代没有有效的image representation，研究者们只有设计复杂的feature representations，以及利用各种加速计算的手段来利用有限的计算资源。

* **Viola Jones Detectors**

18年前，P.Viola和M.Jones第一次实现了不需要任何约束的人脸的real-time detection。运行在当时的CPU下，这个detector在效果差不多的情况下比当时其它的算法要快个几百倍。这个detection算法之后被命名为了VJ detector。

VJ detector使用了最直接的detection的方式：sliding Windows。也就是让窗口滑过图片上所有的位置并且尝试所有大小的窗口，因此检测图片上任何位置任何大小的人脸。尽管这个算法看起来不复杂，但所涉及到的计算对当时的电脑来说是不行的。VJ detector利用三个重要的技术极大的加速了它的检测速度：integral image，feature selection和detection cascade。

1）Integral image：integral image是一种用来加速box filtering或者说convolution过程的计算方式。正如当时其他的object detection算法，VJ detector使用Haar wavelet作为一张图片的feature representation。integral image使得VJ detector里的每个window的计算量与window的大小无关。

2）Feature selection：作者并没有用一系列手动设计的Haar basis filters，而是使用了Adaboost算法来从很多的features里选取对face detection最有用的一小部分features。

3）Detection cascades：VJ detector里使用了一种multi-stage detection paradigm（也就是detection cascades）通过减少在背景上的windows和增加在face上的windows来减少运算量。


* **HOG Detector**

Histogram of Oriented Gradients（HOG）feature descriptor在2005年由N.Dalal和B.Triggs提出。HOG可以被认为是scale-invariant feature transform和shape contexts的一个重要的改进。为了平衡feature invariance（包括translation，scale，illumination等）和nonlinearity（区分不同种类的objects），也就是说既想detectors能够包容同类objects图片内的变化，有希望objects抓住不同类objects之间的差异，HOG descriptor被设计用来在一个稠密的grid上进行计算。尽管HOG可以被用来检测一系列不同的object类别，它实际上是因为pedestrian detection任务而开发的。为了能够检测不同大小的objects，HOG detector将原输入图片rescale了几次并且保持detection window的大小不变。HOG detector在很长一段时间内都是很多重要的object detectors的基础，并且在CV应用领域用了很多年。

* **Deformable Part-based Model (DPM)**

DPM，作为VOC-07, 08以及09 detection比赛的冠军，是传统object detection方法的巅峰。DPM是由P.Felzenszwalb在2008年作为HOG detector的扩展而提出，之后由R.Girshick做了一系列的重要改进。

DPM遵循divide and conquer原则来做object detection的任务，也就是训练可以被简单看成学习分解一个object的恰当的方式，inference可以被看成对检测到的不同的object parts的组装的过程。比如说，detect一个car可以被认为是detection它的window，body和wheels。

一个典型的DPM detector包含一个root-filter和一些part filters。DPM模型并不需要手动标记part filters的参数（比如说size和location），而是采用了一个weakly supervised的learning method，其中每个part filter的参数都可以被当作latent variables来被学习到。

尽管现在的object detectors已经在detection精度上远超过了DPM，但是很多还在被DPM的思想影响，比如说mixture models，hard negative mining，bounding box regression等。在2010年，P.Felzenszwalb和R.Girshick被PASCAL VOC授予lifetime achievement。


##### 2.1.2 Milestones: CNN based Two-stage Detectors

随着hand-crafted features的表现日趋饱和，object detection在2010年之后到达了它的顶峰。在2012年，全世界都目睹了CNN的重生。因为一个deep CNN可以学习到一张图片robust和high-level的representation，一个自然的问题就是我们是否可以将CNN应用到object detection里。R.Girshick在2014年第一个尝试，他提出[regions with CNN features（RCNN）](https://ieeexplore.ieee.org/ielaam/34/7346524/7112511-aam.pdf)用于object detection。从此，object detection开启了飞速发展的时代。

在deep learning时代，object detection可以被分为两种方式：two-stage detection和one-stage detection，前者将detection描述为coarse-to-fine的过程，而后者直接一步到位。

* **RCNN**

RCNN背后的想法很简单：它从利用selective search来寻找一系列object proposals（object candidate boxes）开始。之后每个proposal会被rescale到一个固定大小的image然后喂给一个在ImageNet上预训练好了的CNN模型来获取features。最后linear SVM classifiers被用来预测在每个位置是否有object以及这个object的类别。

RCNN在VOC07上获得了显著的性能提升，在mean Average precision（mAP）上从DPM-V5的33.7%提升到58.5%。

尽管RCNN取得了巨大的成功，其缺点也是很明显的：在数量很多的互相覆盖的proposals上进行大量的冗余的feature计算（每张照片有超过2000个proposals），这导致了极低的detection速度（14秒一张）。在同一年里，[SPPNet](http://datascienceassn.org/sites/default/files/Spatial%20Pyramid%20Pooling%20in%20Deep%20Convolutional%20Networks%20for%20Visual%20Recognition.pdf)被提了出来，用来解决了这个问题。

* **SPPNet**

在2014年，何凯明提出了Spatial Pyramid Pooling Networks（SPPNet）。之前的CNN模型需要一个固定大小的输入，比如说$$224 \times 224$$。SPPNet的最主要的贡献在于提出了一个spatial pyramid pooling (SPP)层，其使得一个CNN可以不管image或者region的大小也不用去resale它就能得到一个固定长度的representation。当使用SPPNet用于object detection时，feature maps可以从整张图片一次性计算而得，之后任意region的固定长度的representations都可以被生成用来训练detectors，从而就不需要重复计算convolutional features了。SPPNet比R-CNN快了20倍，而并没有牺牲任何的detection精度（VOC07 mAP=59.2%）。

尽管SPPNet很大程度的提高了detection速度，但是它仍然还有很多缺点：首先，训练仍然是multi-stage的；其次，SPPNet仅仅finetune它的fully connected layers而忽略了前面的那些layers。在下一年里，[fast RCNN](https://openaccess.thecvf.com/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)被提了出来用于解决这些问题。


* **Fast RCNN**









## CNN based two-staged detectors

### [RCNN: Region-based convolutional networks for accurate object detection and segmentation](https://ieeexplore.ieee.org/ielaam/34/7346524/7112511-aam.pdf)

*TPAMI 2016*


### [Spatial pyramid pooling in deep convolutional networks for visual recognition](http://datascienceassn.org/sites/default/files/Spatial%20Pyramid%20Pooling%20in%20Deep%20Convolutional%20Networks%20for%20Visual%20Recognition.pdf)

*ECCV 2014*

### [Fast R-CNN](https://openaccess.thecvf.com/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)

*ICCV 2015*

### [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://proceedings.neurips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html)

*NeurIPS 2015*


### [FPN: Feature Pyramid Networks for Object Detection](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf)

*CVPR 2017*


## CNN based one-staged detectors

### [YOLO v1: You Only Look Once: Unified, Real-Time Object Detection](https://www.cvfoundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)

*CVPR 2016*

### [YOLO v2: YOLO9000: Better, Faster, Stronger](https://openaccess.thecvf.com/content_cvpr_2017/papers/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.pdf)

*CVPR 2017*

### [YOLOv3: An Incremental Improvement](https://arxiv.org/pdf/1804.02767.pdf)

*Arxiv 2018*

### [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/pdf/2004.10934.pdf)

*Arxiv 2020*

### [SSD: Single Shot MultiBox Detector](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_2)

*ECCV 2016*

### [Retina-Net: Focal Loss for Dense Object Detection](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)

*ICCV 2017*


## Transformer-based detectors

### [End-to-end object detection with Transformers](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460205.pdf)

*ECCV 2020*


## Generative model-based detectors

### [Self-Supervised Object Detection via Generative Image Synthesis](https://openaccess.thecvf.com/content/ICCV2021/html/Mustikovela_Self-Supervised_Object_Detection_via_Generative_Image_Synthesis_ICCV_2021_paper.html)

*ICCV 2021*



---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*

