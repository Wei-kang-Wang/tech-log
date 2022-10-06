---
layout: post
comments: false
title: "Object Detection"
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

[CODE](https://github.com/facebookresearch/detr)

*Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko*

**Abstract**

作者提出了一个新的方法来解决object detection问题，其将这个问题看作一个直接的set prediction问题。作者的方法使得detection pipeline得以实现自动流水化，去除了很多需要手动操作的部分，比如非极大抑制（non-maximum suppression）、结合先验知识的anchor generation等。作者提出的方法的最主要的部分（也就是模型部分）叫做detection transformer（简称DETR），是一个基于transformer架构的encoder-decoder结构，使用的loss是一个基于set的二分匹配的loss。给一个固定的已经学习到了的物体类别构成的集合作为query，DETR会推测这些给定的objects与图片内容之间的relations，然后直接并行的输出图片里包含的objects。DETR在概念上很简单，而且和目前那些detectors不一样，并不需要指定一个库（也就是指定objects的类别范围）。DETR在COCO object detection数据集上和Faster R-CNN这个baseline效果相似。而且，DETR还可以很轻易的应用于图片分割，作者用实验展示了效果要远超其它baselines。


**1. Introduction**

object detection的目的是预测由bounding box构成的set以及每个object的category。现代的detectors用一种非直接的方式来解决这个set prediction任务：将其描述为在一个由proposals、anchors或者window centers构成的非常大的集合上所进行的surrogate regression以及classification任务。这些detectors的效果会受到后续一些操作的很大的影响，包括如何消除重复的挨得很近的预测结果，如何设计anchor sets，以及如何将target box匹配到anchors上等等。为了简化这些流程，作者提出了一个直接的预测方法，而不必在使用这些surrogate任务。这样的端到端的设计理念在一些很复杂的结构预测任务上都取得了很大的进步，比如说machine translation或者speech recognition，但是还并没有被用在object detection这个任务上。

>实际上也有这样的尝试，只不过要么就是还使用了一些先验知识，要么就是并没有在数据集上的效果能比得上sota的detectors（比如说faster R-CNN）。而这篇文章将会解决这个问题，也就是提出一种端到端的object detection模型，而且效果和sota的差不多。

作者将训练流程实现了自动流水化，他们将object detection任务看作一个直接的set detection任务。作者使用了以基于transformer架构的encoder-decoder模型（transformer对于sequence prediction是有优势的）。transformer架构里的自注意力机制（显式的对于sequence里的任意两个元素都计算其相关性）特别适合用来进行object detection任务（因为其可以很自然的解决重复的挨得很近的prediction的问题）。

作者提出的detection transformer（DETR），由fig1所示，同时预测一张图片里的所有的objects，而且是一个端到端的模型，其使用的是一个set loss，会在预测的objects和ground-truth objects之间计算一个bipartite matching。DETR通过丢弃那些结合了先验知识的需要手动设计的模块来大大简化了detectors的流程，比如说spatial anchors，或者non-maximal suppression。和现有的大多数detectors不一样，DETR并不需要任何特殊设计的layers，所以说内置了ResNet和Transformer架构的框架都可以很轻易地实现DETR模型的构建（比如PyTorch，TensorFlow等）。

![detr1]({{ '/assets/images/DETR-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. DETR结合了一个CNN和一个Transformer架构来直接并行输出detection的集合。在训练过程中，bipartite matching会对每个ground-truth box唯一指定一个预测的结果来计算loss。那些没有匹配结果的预测结果就会被给定一个no object的类别预测。*

之前也不是没有工作尝试进行直接的set prediction，和这些工作相比，DETR的最主要的特点是使用了bipartite matching loss以及使用了并行输出的decoding方式的transformer（在最初的transformer架构里，decoder的输出是一种auto-regressive方式的，也就是一个个的输出，而且后一个输出依赖于之前的输出，但这里是并行输出，non-autoregressive）。而之前的那些工作是使用了autoregressive decoding方式的RNN：[End-to-end instance segmentation with recurrent attention](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ren_End-To-End_Instance_Segmentation_CVPR_2017_paper.pdf)，[Recurrent neural networks for semantic instance segmentation](https://arxiv.org/pdf/1712.00617.pdf)。本文使用的matching loss会对每个ground truth object匹配唯一一个预测的值，而且打乱预测的object的顺序也不影响匹配计算，所以说完全可以并行输出。

作者在COCO object detection数据集上测试了DETR的效果，并和非常强大的对手Faster R-CNN进行了对比。Faster R-CNN已经经过了数代更新，现在非常的强大。作者的实验证明本文所提出的方法和Faster R-CNN在COCO数据集上的效果相当。更准确的说，DETR在一些更大的objects的检测上效果更好，这可能是得益于transformer的全局注意力机制。但是DETR在很小的objects上的检测效果就会差一些。实际上R-CNN也有这样的问题，但后来FPN解决了R-CNN在很小的物体上检测效果不好的问题，作者希望将来也有类似的工作来改进DETR的这个问题。

DETR的训练和其它的object detectors的训练有多处不同。DETR的训练需要很长的训练周期，而且会得益于一些额外的transformer的decoder模块里新增的一些loss。作者也详细阐述了哪些组件对于模型效果来说是至关重要的。

DETR的设计理念可以很容易的被扩展到更复杂的任务上。在作者的实验里，他们展示了是需要在训练好的DETR上加上一个segmentation头就可以在panoptic segmentation任务上比baseline的效果好。

>panoptic segmentation是最近新兴的一个具有挑战性的基于像素的recognition任务。


**2. Related Work**

本文的工作所基于的先前的工作可以分为以下几类：set prediction的bipartite matching loss，基于transformer的encoder-decoder架构，parallel decoding以及object detection method。

**2.1 Set Prediction**

现在还并没有一个标准的深度学习模型来直接预测sets。最简单最基础的set prediction任务是multilabel classification（和multiclass classification不一样，那个就是基础的分类任务，也就是每个输入只能被唯一分配一个类别，而multilabel classification里的每个输入可以被分配好几个label，它们之间互相不会排斥）。预测sets的任务的第一个难点是如何避免很相近的重复结果。绝大多数现在的detectors算法都是使用某种后验方法（在模型得出结果后使用的方法），比如说非极大抑制，来解决这个问题，但是直接的set prediction方法应该是不需要后验的。它们需要某种全局的方法来为所预测到的结果之间计算关系从而避免这种冗余信息。对于固定大小的set prediction任务，dense fully connected networks是可以很好解决的，但是计算量太大。一个general的方法是使用自回归序列模型，比如说RNN。但是在所有的情况下，loss function都需要对于预测结果的顺序是不变的。常见的方法是基于Hungarian algorithm来设计loss function，在ground-truth和prediction之间找一个bipartite matching。这样就可以确保loss对于预测结果顺序不变，而且每个预测结果有个唯一的match。这篇文章也采用了这种bipartite matching loss的方法。和之前绝大多数工作相比，作者并没有使用自回归模型，而是使用了并行输出结构的decoder的transformer架构，下面将会说。

**2.2 Transformers and Parallel Decoding**

Transformers是作为一个用来解决机器翻译任务而提出的基于注意力机制的模型。注意力机制是可以从整个输入的序列里来综合信息的。transformers里介绍的自注意力层，会扫过一个序列里所有的元素，然后以一种从整个序列里综合得到的信息的方式来更新每个元素对应的特征。基于注意力的模型的一个优势在于，其可以进行全局计算，这让它比RNN对于很长的序列来说效果更好。现在在NLP，CV领域，Transformers大有取代RNN的趋势。

transformers最一开始提出是被用在自回归模型里的，这和早期的sequence-to-sequence模型是一样的，以一个一个元素的形式来输出。然而现在也有很多任务使用了并行输出的decoding模式的transformers。本文使用后者来进行set prediction。

**2.3 Object detection**

很多现代的object detectors是基于某些初始化的“猜测”来做预测的。two-stage detectors基于proposals来预测boxes，而single-stage detectors基于anchors或者一系列可能的物体中心点来预测。现在有工作表明，最后detectors效果的好坏很大程度上决定于早期这些猜测的效果。在本文中，作者可以丢弃这些早期的猜测，直接从输入图片或者视频里学习到box预测结果。

*set-based loss*

有些object detectors已经使用了bipartite matching loss。然而，在这些早期的深度学习模型里，不同的预测结果之间的关系是用卷积或者全连接层来建模的，然后还需要一个手动设置的非极大抑制后验步骤来提升效果。

*recurrent detectors*


**3. The DETR model**

直接预测set的detection算法有两个关键的组件：（1）需要一个set prediction loss来迫使ground truth和预测结果之间有唯一的对应关系；（2）一个架构能一次性预测一系列物体和它们之间的关系。fig2详细描述了模型的细节。


![detr2]({{ '/assets/images/DETR-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. DETR使用一个经典的CNN框架来学习一张输入图片的2D特征。接着将其展平，再加上positional encoding，然后送给一个transformer的encoder。transformer的decoder会使用encoder的输出结合一些固定数量的可学习的positional embeddings（叫做object queries）来输出结果。输出的每一个结果都通过一个feed forward network（FFN）（共用的）来预测一个结果，要么是一个物体类别和其的bounding box，要么是一个no class类别。*

**3.1 Object detection set prediction loss**

DETR的输出是一个固定的数值，也就是$$N$$个预测结果，而且是并行得到的，$$N$$是一个超参数，要比图片里可能存在的物体的数量大很多。训练的一个主要的难点在于如何根据ground truth来给预测结果打分（预测结果包括类别，位置）。本文使用的loss会在预测结果和ground truth结果之间产生一个bipartite matching，然后再进行bounding box loss的计算。

用$$y$$表示物体set的ground truth，$$\hat y = \lbrace \hat y_i \rbrace_{i=1}^N$$表示的是$$N$$个预测结果。假设$$N$$要比一张图片里的物体数量多很多，作者将$$y$$也使用padding的方式加了很多$$\emptyset$$来表示no object这个类别。

**3.2 DETR architecture**


## Generative model-based detectors

### [Self-Supervised Object Detection via Generative Image Synthesis](https://openaccess.thecvf.com/content/ICCV2021/html/Mustikovela_Self-Supervised_Object_Detection_via_Generative_Image_Synthesis_ICCV_2021_paper.html)

*ICCV 2021*



---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*

