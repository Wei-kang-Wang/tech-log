---
layout: post
comments: false
title: "Mesh-based 3D Reconstruction from Images"
date: 2024-05-27 01:09:00
tags: paper-reading
---

> This post is a summary of 3D reconstruction from images papers.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

## [3DFauna: Learning the 3D Fauna of the Web](https://kyleleey.github.io/3DFauna/)

*CVPR 2024*

*Zizhang Li, Dor Litvak, Ruining Li, Yunzhi Zhang, Tomas Jakab, Christian Rupprecht, Shangzhe Wu, Andrea Vedaldi, Jiajun Wu*

*Stanford University, UT Austin, University of Oxford*

![general]({{ '/assets/images/fauna_1.png' | relative_url }})
{: style="width: 1200px; max-width: 100%;"}

这篇文章是基于MagicPony那篇文章做的，作者里也有MagicPony那篇文章的一作，Shangzhe Wu。这篇文章最大的novelty就是，MagicPony，包括之前的3D mesh-based reconstruction from 2D inhomogeneous images都是针对的一个category，也就是说，针对同一种类不同个体的images，来学习一个3D model。而这篇文章是针对**多个具有结构/几何相似性的种类**的不同个体的images set，来学习统一的3D model。

这篇文章的3D model仍然基于skinned model，只不过不同于以往的$$S_{base}+S_{instance}+S_{posing}$$的模式（$$S_{base}$$是所有的instance公用的，$$S_{instance}$$是每个instance独有的小的deformation，$$S_{posing}$$值得是由于articulation和viewpoint改变造成的影响），这篇文章里的3D model对skinned model改进了一点，现在的$$S_{base}$$不再是对于所有的instance都一样了，而是对于每一个都不一样，但这种不一样的flexibility又不能太大了，否则难以学习，以及难以解决viewpoint和deformation之间的ambiguity，从而作者的做法是建立一个basis shape的字典，key就是一些表示feature的vector，value就是一个shape basis，从而对于任意一个输入图片，其也有一个feature vector，将其与字典里的所有的key之间计算cosine similarity，再将这个similarity作为系数，乘上每个shape basis，加在一起，就成为了最后这张图片对应的$$S_{base}$$，也就是说是字典里所有的shape basis的一个线性组合。

注意，这个字典里的key和value都是end-to-end learnable的，没有做任何设计。

上述的这个设计叫做Semantic Bank of Skinned Model (SBSM)，是本文的核心。

而这种设计只能用于那些本就具有相似性的categories间，本文就是针对的四足类动物quadruped animals（如果是差异过大的类别，比如bird和horse，那本文的方法肯定是不行的）。

**1. 本文的动机**

从2D图片来学习动物的3D model一直以来都在做。人类也是一种动物，对于人类3D model的研究做得很透彻，比如说经典的skinned model：[SMPL: A skinned multiperson linear model](https://files.is.tue.mpg.de/black/papers/SMPL2015.pdf)。但是训练一个好的skinned model，需要很多人工标注，还需要好的pose predictor等等，工作量很大。不可能将其推广到所有动物上去。

最近有一系列的方法可以直接从2D图片，不需要标注的情况下来学习动物的3D model，比如说MagicPony，LaSSIE，DOVE等，他们也是使用skinned model来设计3D model，但使用了其他的techniques来绕开了标注的困难，所以是unsupervised的方法。但他们都是一个模型只能建模一个category的动物。而本文致力于一个模型可以建模多个具有类似结构的categories的动物（比如四足类）。这样不仅省力，而且对于那些图片很少的珍稀动物，也能更好的建模。

**2. 方法**

![method]({{ '/assets/images/fauna_2.jpg' | relative_url }})
{: style="width: 1200px; max-width: 100%;"}

3D-Fauna只需要：（1）一系列categories的不同instances的2D RGB图片；（2）一个预训练好的2D image segmentation model，用来将instance从background里分开；（3）一个预训练好的feature extractor，用来提取图片的features（比如DINO-ViT）。（4）一个用于进行articulation的skeleton的设计，比如说将一个skeleton分为头、身体、四肢、尾巴七部分，每部分看作一个整体，可以相对于其他部分做rotation，从而skinned model里的articulation（或者叫做posing）就是要预测这些relative的rotation以及一个整体的rigid body transformation（一个rotation和translation，等价于camera的位姿）。

文章里有些话很精髓，直接引用

> The most important is to develop a 3D representation that is sufficiently expressive to model the diverse shape variations of the animals, and at the same time tight enough to be learned from single-view images without overfitting individual views.

> Non-Rigid Structure-from-Motion shows that reconstruction is still possible, but only if one makes the space of possible 3D shapes sufficiently tight to remove the reconstruction ambiguity. At the same time, the space must be sufficiently expressive to capture all animals.

要想从2D图片里直接学习一个3D model，这个3D model既得足够flexible/expressive，因为categories之间不同，同一个category里的instances之间也不同，但同时这个3D model也不能过于flexible了，因为需要同时学习deformation和viewpoint，而如果deformation过于flexible的话，就没法学到正确的viewpoint了（比如说一个极端情况，rigid body，也就是假设没有deformation，那这样的话，理论上有两张图片的对应点，就能学到viewpoint）。我们需要**设计一个flexibility合适的3D model，来在deformation和viewpoint之间找到一个平衡**，这是3D reconstruction from 2D images的精髓（对于现在的那些deformable NeRF来说也是这样）。

而对于MagicPony等方法，他们实现这个平衡的办法就是使用skinned model。本文也使用skinned model，但由于本文想要一个模型建模多个具有相似性的categories，所以3D model理论上需要更大一点的flexibility，而这就是Semantic Bank Skinned Model的用武之地，也就是在skinned model的basis shape上做了更改，增加了flexibility。

3D-Fauna还做了一个创新。作者发现网络上的图片，尤其是动物图片，很多都是摄影师拍的，它们并不是为了3D reconstruction而设计的，大多数是为了展览等目的，这样的话就会有photographer bias，也就是说，大部分的照片都是从正面拍的，导致数据在viewpoint这个维度上分布不均。作者提出的解决方案就是，在模型里，加入一个模块，其先随机采样viewpoint，再从这个viewpoint对3D model的silhouettes进行rendering，也就是说出来的也就是2D的silhouette，再让一个discriminator来判断这个silhouette图片是生成的还是原图片本身的silhouette，从而让随机角度生成的silhouette和真实图片的silhouette呆在同一个distribution里。

之所以要将skinned model扩展为semantic bank skinned model（也就是扩展$$S_{base}$$），而不是直接让$$S_{instance}$$来model不同categories之间的差异是因为这样的话，这个$$S_{instance}$$就会太大了，这就导致这个模型不够tight了，就会导致deformation和viewpoint prediction之间的ambiguity。为了避免这个问题，为了让$$S_{instance}$$还是足够的小，就选择对$$S_{base}$$进行了改变。

注意在实现细节上，SBSM字典里的每个key对应的value并不直接由shape来表示（即mesh里的vertices和faces），而是表示为类似于keys的vectors，然后对于每张图片，得到一个这些表示values的vectors的linear combination，将这个combination作为输入给一个可学习的网络，输出$$V$$和$$F$$，分别表示base的vertices的位置，以及faces覆盖情况，即$$S_{base}$$。而后续的$$S_{instance}$$以及$$S_{posing}$$就不再学习faces了，只对vertices的位置进行调整。

在具体的实现细节上，这个输出$$S_{base}$$的网络，由一个MLP实现，而这个$$S_{base}$$实际上是表示为了一个SDF。输出$$S_{instance}$$的网络，输出的仅仅是每个vertices的一个位移，加在$$S_{base}$$上，就实现了deformations。而对于$$S_{posing}$$，本文使用了和MagicPony一样的假设，设计了一个quadrupedal skeleton，并预测了一个rigid body transformation以及多个relative rotations，用来进行camera的位姿调整，以及身体各个部分的articulation，这些预测的transformation/rotations被用在$$S_{base}+S_{instance}$$的vertices上（具体做法叫做linear blend skinning equation，参考SMPL论文），从而实现了最终vertices位置的确定，再加上之间预测的faces，就有了完整的mesh了。

在有了mesh之后，就剩下了texturing。本文还设计了网络用来预测albedo和lighting，用标准的Lambertian illumination model来进行上色（和Shangzhe Wu之间的很多论文所用的方法一致）。

















