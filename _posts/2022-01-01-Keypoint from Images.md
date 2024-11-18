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


#### \[**CVPR 2019**\] [D2-Net: A Trainable CNN for Joint Description and Detection of Local Features](https://openaccess.thecvf.com/content_CVPR_2019/papers/Dusmanu_D2-Net_A_Trainable_CNN_for_Joint_Description_and_Detection_of_CVPR_2019_paper.pdf)

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



### \[**WACV 2021**\] [Conditional Link Prediction of Category-Implicit Keypoint Detection](https://openaccess.thecvf.com/content/WACV2021/html/Yi-Ge_Conditional_Link_Prediction_of_Category-Implicit_Keypoint_Detection_WACV_2021_paper.html)


### \[**ICCV 2021**\] [On Equivariant and Invariant Learning of Object Landmark Representations](https://openaccess.thecvf.com/content/ICCV2021/html/Cheng_On_Equivariant_and_Invariant_Learning_of_Object_Landmark_Representations_ICCV_2021_paper.html)


### \[**WACV 2021**\] [Learning of Low-Level Feature Keypoints for Accurate and Robust Detection](https://openaccess.thecvf.com/content/WACV2021/html/Suwanwimolkul_Learning_of_Low-Level_Feature_Keypoints_for_Accurate_and_Robust_Detection_WACV_2021_paper.html)



## 2D keypoints from images (Unsupervised)






### \[**3DV 2019**\] [Structured Domain Adaptation for 3D Keypoint Estimation](https://ieeexplore.ieee.org/abstract/document/8885979)





  
### \[**NeurIPS 2019**\] [Unsupervised Learning of Object Keypoints for Perception and Control](https://proceedings.neurips.cc/paper/2019/hash/dae3312c4c6c7000a37ecfb7b0aeb0e4-Abstract.html)


### \[**NeurIPS 2019**\] [Object landmark discovery through unsupervised adaptation](https://proceedings.neurips.cc/paper_files/paper/2019/file/97c99dd2a042908aabc0bafc64ddc028-Paper.pdf)


### \[**WACV 2022**\] [LEAD: Self-Supervised Landmark Estimation by Aligning Distributions of Feature Similarity](https://openaccess.thecvf.com/content/WACV2022/html/Karmali_LEAD_Self-Supervised_Landmark_Estimation_by_Aligning_Distributions_of_Feature_Similarity_WACV_2022_paper.html)




### \[**NeurIPS 2019**\] [Unsupervised Learning of Object Keypoints for Perception and Control](https://proceedings.neurips.cc/paper/2019/hash/dae3312c4c6c7000a37ecfb7b0aeb0e4-Abstract.html)



### \[**CVPR 2020**\] [Self-Supervised Learning of Interpretable Keypoints From Unlabelled Videos](https://openaccess.thecvf.com/content_CVPR_2020/html/Jakab_Self-Supervised_Learning_of_Interpretable_Keypoints_From_Unlabelled_Videos_CVPR_2020_paper.html)


### \[**NeurIPS 2019**\] [Unsupervised Keypoint Learning for Guiding Class-Conditional Video Prediction](https://proceedings.neurips.cc/paper/2019/hash/801272ee79cfde7fa5960571fee36b9b-Abstract.html)


### \[**CVPR 2020**\] [Unsupervised Learning of Intrinsic Structural Representation Points](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Unsupervised_Learning_of_Intrinsic_Structural_Representation_Points_CVPR_2020_paper.html)


### \[**CVPR 2021**\] [Unsupervised Human Pose Estimation Through Transforming Shape Templates](https://openaccess.thecvf.com/content/CVPR2021/html/Schmidtke_Unsupervised_Human_Pose_Estimation_Through_Transforming_Shape_Templates_CVPR_2021_paper.html)

[POST](https://infantmotion.github.io)



### \[**ICLR 2021**\] [Unsupervised Object Keypoint Learning using Local Spatial Predictability](https://openreview.net/forum?id=GJwMHetHc73)


### \[**ICLR 2021**\] [Semi-supervised Keypoint Localization](https://openreview.net/forum?id=yFJ67zTeI2)


### \[**CVPR 2022**\] [Self-Supervised Keypoint Discovery in Behavioral Videos](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_Self-Supervised_Keypoint_Discovery_in_Behavioral_Videos_CVPR_2022_paper.pdf)


### \[**Arxiv 2021**\] [Weakly Supervised Keypoint Discovery](https://arxiv.org/pdf/2109.13423.pdf)


### \[**Arxiv 2021**\] [Pretrained equivariant features improve unsupervised landmark discovery](https://arxiv.org/pdf/2104.02925.pdf)


### \[**Arxiv 2019**\] [Learning Landmarks from Unaligned Data using Image Translation](https://openreview.net/pdf?id=xz3XULBWFE)



### \[**WACV 2022**\] [LEAD: Self-Supervised Landmark Estimation by Aligning Distributions of Feature Similarity](https://openaccess.thecvf.com/content/WACV2022/html/Karmali_LEAD_Self-Supervised_Landmark_Estimation_by_Aligning_Distributions_of_Feature_Similarity_WACV_2022_paper.html)


### \[**Arxiv 2020**\] [Unsupervised Learning of Facial Landmarks based on Inter-Intra Subject Consistencies](https://arxiv.org/pdf/2004.07936.pdf)


### \[**Arxiv 2020**\] [Unsupervised Landmark Learning from Unpaired Data](https://arxiv.org/pdf/2007.01053.pdf)

[CODE](https://github.com/justimyhxu/ULTRA)


### \[**Arxiv 2021**\] [LatentKeypointGAN: Controlling GANs via Latent Keypoints](https://xingzhehe.github.io/LatentKeypointGAN/)


### \[**NeurIPS 2021**\] [Unsupervised Part Discovery from Contrastive Reconstruction](https://proceedings.neurips.cc/paper/2021/hash/ec8ce6abb3e952a85b8551ba726a1227-Abstract.html)


### \[**Arxiv 2019**\] [Video Interpolation and Prediction with Unsupervised Landmarks](https://arxiv.org/pdf/1909.02749.pdf)


### \[**CVPR 2022**\] [Few-shot Keypoint Detection with Uncertainty Learning for Unseen Species](https://openaccess.thecvf.com/content/CVPR2022/papers/Lu_Few-Shot_Keypoint_Detection_With_Uncertainty_Learning_for_Unseen_Species_CVPR_2022_paper.pdf)


### \[**Arxiv 2021**\] [TACK: Few-Shot Keypoint Detection as Task Adaptation via Latent Embeddings](https://sites.google.com/view/2021-tack)


