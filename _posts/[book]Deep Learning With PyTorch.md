---
layout: post
comments: false
title: "[book]Deep Learning with PyTorch"
date: 2021-11-29 01:09:00
tags: book-reading
---

> 这是Deep Learning with PyTorch这本书的翻译版，加上了个人一些拙略的理解。


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

![Whole Process]({{ '/assets/images/DLP-0-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. The Whole Process of Deep Learning.*

## 0.1 序

当我们于2016年中旬开始PyTorch项目的时候，我们还是网上相识的一群搞开源代码的hackers，都想着写一个更好的深度学习软件。这本书的两个作者，Luca Antiga和Thomas Viehmann，在开发PyTorch时贡献了很大的力量并使得它变成今天这样成功。

我们开发PyTorch的目标是建立一个最flexible的框架来实现深度学习算法。PyTorch从Torch7（由Ronan Colobert等人于2007年开始开发）那里继承了很大一部分代码库，并且借用了Lush语言（由Yann LeCun和Leon Bottou开发）的基础概念。这些本就已经有的基础让我们可以更加专注于改进，而不是从头开始造轮子。

PyTorch的成功很难归咎于一个单纯的原因。这个项目提供了很好的用户体验，良好的debug环境以及很强的可塑性，使得用户能够很高效的编程。PyTorch的大量使用造成了一个良好的社区环境，从而促使其更进一步的发展，形成一个良性循环。

这本书Deep Learning with PyTorch提供了一个让我们原作者介绍PyTorch的机会。它涵盖了很多基础的细节，包括tensor，neural network如何实现的细节等等。而且它还涵盖了一些高级的项目，以及如何将PyTorch高效部署等高级知识。

而且这本书还包括了很多应用实例，这样可以让读者更好的理解PyTorch而且在应用中更加熟练。Eli Stevens在软件方面的技巧加上Thomas对PyTorch核心的理解，可以让应用的实现部分更加易懂且深刻。


## 0.2 关于这本书

这本书目的在于介绍PyTorch的基础以及它在日常项目中的应用。我们立志于提供深度学习的关键知识并且展现PyTorch如何使得用户将这些知识应用到现实。在这本书里，我们试图提供支持后续研究的直觉，并且说明代码背后的原理。

Deep Learning with PyTorch这本书并不是想要做成字典类的参考书，而是做成一本介绍概念的书，使得读者能够独立开发自己的项目。所以，我们集中于介绍PyTorch某一方面的知识。最显著的没有介绍的内容就是recurrent neural network。

### 0.2.1 谁该看这本书

这本书是为了想要成为深度学习开发者或者想要了解PyTorch的人所准备的。我们认为这本书典型的读者包括计算机科学家，数据学家，或者软件工程师，或者相关专业的学生。

我们认为读者对命令式编程和面向对象编程有一些基本的了解。因为这本书用的是Python，读者需要对其语法和操作环境有所理解。了解如何安装Python包以及运行脚本是所需要掌握的前行知识。了解Numpy将会很有帮助。同时了解一些linear algebra的基础也是有必要的。

### 0.2.2 这本书是怎么组织的：线路图

Deep Learning With PyTorch包括了三个部分。Part I涵盖了基础知识；Part II从头到尾介绍了一个项目，运用了Part I里的基础知识并加上了一些高级的内容；短小的Part III简单介绍了PyTorch如何部署在设备上作为结尾。

#### 0.2.2.1 Part I

在Part I里，我们开始了认识PyTorch的第一步，建立起理解PyTorch项目的基础知识并为以后写自己的项目做准备。我们介绍的内容涵盖了PyTorch API以及实现PyTorch库的内部机理，并且介绍了如何训练一个分类网络。在Part I结束的时候，我们就已经做好了研究现实项目的能力。

Chapter1 介绍PyTorch是一个库，并介绍了其在深度学习革命中的作用，并且介绍了PyTorch之所以区分于其他深度学习框架的原因。

Chapter2 通过运行pretrained example代码实例来真正接触PyTorch；这章显示了如何在PyTorch Hub里下载以及运行模型。

Chapter3 介绍了PyTorch一个基本的模块：tensor（张量），介绍了它的API以及它背后的实现细节。

Chapter4 阐述了不同类型的数据怎么用tensor来表示，以及各种深度学习模型需要什么格式的tensor。

Chapter5 介绍了通过gradient descent学习的理论，并且介绍了PyTorch如何用automatic differentiation（自动微分）实现它。

Chapter6 用PyTorch里的nn和optim模块介绍了如何构建和训练一个用于回归的模型。

Chapter7 基于之前的知识建立了一个用于图像分类的全连接网络，并且扩展了对PyTorch API的介绍。

Chapter8 介绍了convolutional neural network（卷积神经网络）并介绍了一点构建神经网络模型的高级知识以及它们的PyTorch实现。


#### 0.2.2.2 Part II

在Part II里，每一chapter都让我们对于解决如何自动检测肺癌这个问题更进一步。我们用这个困难的问题作为解释现实中如何解决类似检测肺癌这种复杂问题的动机。

Chapter9 描述了基于CT图像的end-to-end的自动肺癌检测的方法。

Chapter10 加载了具有人工标注的CT照片，并使用标准PyTorch API将相关的信息转化为tensor。

Chapter11 介绍了第一个利用Chapter10中所生成的数据作为输入的分类模型。我们训练这个模型，并记录一些基本的performance metrics。我们同时也介绍了如何用TensorBoard来监控训练过程。

Chapter12 研究并落实了标准performance metrics，并利用这些metrics来鉴别之前训练所体现出的问题。之后我们用一个经过了data balancing and augmentation（数据平衡和扩充）的更好的训练集来解决这些训练设计上的漏洞。

Chapter13 描述了segmentation（分割）任务，利用一个pixel-to-pixel的模型结构来生成一个表示瘤子可能出现位置的p大小和原输入图片一样大的heatmap。这个heatmap可以用来在没有人工标注的CT图像上找到瘤子。

Chapter14 实现了最后的end-to-end项目：在分类任务后接分割模型用来诊断癌症病人。

#### 0.2.2.3 Part III

Part III只有一个章节，用来解释PyTorch的部署。Chapter15 提供了怎么把PyTorch模型部署在网络服务上、将其嵌入一个C++项目或将其部署在移动设备上的一个overview。


### 0.2.3 关于代码

这本书所有的代码都用Python3.6或之后的版本来写。

这本书大量使用Jupyter Notebook来写代码。

下面这段在所有的Jupyter Notebook代码里都被当成第一个模块。

```python
# In[1]:
%matplotlib inline
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_printoptions(edgeitems=2)
torch.manual_seed(123)
```


## Part I Core PyTorch

这是本书的第一部分，从此我们开始正式介绍PyTorch的内容，学习理解PyTorch的机理并理解PyTorch项目的机制。

在Chapter1里我们将会第一次接触PyTorch，理解它是什么，它用来解决什么问题，以及它和别的深度学习框架是什么联系的。Chapter2将会带领读者盘一些已经pretrained的model。Chapter3进一步接触深入的内容，并说明了PyTorch项目里所用的最基础的数据结构：tensor。Chapter4 教会读者如何将不同领域问题的数据表达成PyTorch tensor的形式。Chapter5 展示了深度学习模型是如何从实例中学习到知识的，并说明PyTorch是怎么实现这个过程的。Chapter6 介绍了神经网络组成的基础知识，以及如何用PyTorch构建一个神经网络。Chapter7 介绍了一个利用神经网络实现简单的图像分类任务。Chapter8 介绍了Chapter7里同样的任务如何用卷积神经网络更聪明的实现。

在Part I学习结束之后，在part II里我们就可以用PyTorch应对真实的问题了。

## Chapter 1 Introducing deep learning and the PyTorch Library

>这章包括的内容
>* 深度学习是怎么改变我们处理机器学习任务的方式的。
>* 理解为什么PyTorch很适合深度学习。
>* 检查一个典型的深度学习项目。
>* PyTorch所需的硬件要求。


人工智能是个定义的很差的词，联系到很多的研究，检查，疑惑，fancy的宣传，以及科幻性的制造恐慌。现实当然是更加令人安心的。很难说现在的机器已经可以达到人类的思考水平，更多的是我们发现了一系列通用的算法，它们可以很高效近似复杂的、非线性的过程，从而替代人类自动解决之前只有人类可以解决的问题。

在某种程度上，我们认识到intelligence（智能）是一个和self-awareness（自我意识）相同的概念，而自我意识对于能否解决这些任务来说肯定是不必要的。到了后来，计算机智能这个问题甚至已经变得不重要了。

>Edsger W. Dijkstra认为机器是否能思考这个问题就像是问潜水艇是否会游泳。

我们这本书所要讨论的一大类通用的算法是AI下面一个叫做深度学习的子类，它旨在利用具有指导性的样本来训练深度神经网络（一个数学上的概念）。深度学习利用大量的数据来近似复杂的函数，该函数的input（输入）和output（输出）可以差的很远，比如输入可以是一张图片，而输出是一行描述该输入的文字。这种学习能力让我们可以构建具有处理人类才能解决的问题的代码。

### 1.1 深度学习革命

为了理解深度学习造成了什么样的改变，我们先往前退一步来看。直到上一个十年（2010年左右），绝大部分的机器学习系统还很依赖feature engineering（特征工程）。特征是为了下游任务而对输入数据做的transformation（变形或转换），下游任务比如说一个可以给出正确输出的分类器。特征工程核心在于如何选择一个能使得下游任务能被解决的合适的变形。比如，为了能够区分手写数字0和1的图片，我们可以利用一系列filters（过滤器）来估计图像中边缘的方向，然后训练一个分类器利用边缘方向的分布规律来给出预测结果。另一种可能有效的特征是记录图中围成的洞的个数（0有一个，1有0个）。

相对比来说，深度学习为了成功解决一个任务，直接从原始数据中自动的寻找这样的representation（特征表示）。在判断输入手写数字图片是0还是1的任务里，过滤器在训练的过程中被成对的带有标签的数据反复迭代的更新。这并不是说feature engineering在深度学习中就没有用了；我们经常需要在learning system里插入一些prior knowledge。然而，neural networks基于数据实例摄取数据而提取有用的representation的能力是深度学习最有力的地方。深度学习实践者所关注的重点并不是如何handcrafting这些representations，而是如何使得neural network能基于训练数据来自动的找到这样的representations。而这些自动生成的特征比那些手动的特征往往要更好！因为更多其它技术上的革新，这个事实使得整个机器学习community的想法和观点都有所改变了。


![Revolution]({{ '/assets/images/DLP-1-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Deep Learning exchanges the need to handcraft features for an increase in data and computational requirements.*

如上图右边所示，一个操作员忙碌的定义features并将它们给一个learning algorithm作为输入；而这个任务效果的好坏取决于这个操作员所定义的features的好坏。再上图左侧，利用深度学习，原始的数据直接给了一个algorithm，从而自动得到一系列hierarchical的特征，而这个过程由优化这个任务本身的performance来指导；这个任务效果的好坏将会取决于操作员如何设计这个algorithm。

从上图左侧我们已经初步看到了成功执行一个深度学习任务所需要的东西：
* 我们需要一个方法来ingest手中所拥有的data。
* 我们需要定义一个deep learning machine。
* 我们必须有一个自动的训练过程，用来获得有用的representations以及使得算法输出所需要的输出。

我们更进一步的来看一下训练的过程。在训练过程中，我们需要用一个criterion（尺度，标准），这是一个以模型输出和训练数据为输入的实值函数，我们需要用它来对所想要的输出和实际的输出之前的差值给出一个数值上的数据（一般来说这个值越低越好）。训练就是使得这个值在不断的调整模型参数的过程中越来越低，甚至在训练中没看到的数据上仍然很低（这样就达到了学习的效果）。


### 1.2 用于深度学习的PyTorch

PyTorch是一个用于构建深度学习项目的Python program的库。它强调了flexibility并且允许以Python风格来编写深度学习模型。这种易接近性和易于实用性在研究者团体中找到了早期的追随者，然后在第一版本面世过去的几年里，发展成为了一个具有广泛应用的主流深度学习框架。

正如Python对于programming，PyTorch给深度学习提供了一个很好的介绍。同时，PyTorch也被证明够格用于真实世界里的专业任务。我们相信PyTorch的清晰的语法，合理的API，以及容易上手的debugging使得它是深度学习框架的不二之选。我们十分建议将PyTorch作为第一个深度学习代码库来学习。至于这是不是你将来最后选择使用的库这完全取决于你自己。

在figure1左侧的deep learning machine是一个很复杂的函数。为了能表达这个函数，PyTorch提供了一个核心的数据结构，tensor。tensor是一个和Numpy ndarray很类似的multi-dimensional array。在这个基础数据结构（tensor）周围，PyTorch提供了可以部署在复杂硬件上进行各种加速数学运算的features，使得我们可以很容易的设计neural network的架构以及在单个机器或者并行计算资源上进行训练。

这本书立志于为软件工程师、数据科学家以及熟练掌握Python的学生提供一个学会利用PyTorch构建深度学习项目的起点。我们希望这本书可以既清晰易懂由足够有用，而且我们希望读者能理解书中概念并应用于实际项目中。为了达到这个目标，我们用了一种hands-on的方法并且希望读者已经准备好时刻在电脑上将书中的实例实现。在这本书的基础上，再加上优秀的official documentation，我们希望读者能够自己设计并实现各种项目。

为了能更好学习这本书的内容，你需要做到以下两点：
* 熟悉Python编程。
* 愿意在自己电脑上写代码，get your hands dirty。

这本书由三个不同的部分构成。Part I涵盖了基础知识，将PyTorch如何用代码实现figure1左侧的流程的细节进行了解释。Part II带读者从头到尾过了一个end-to-end的项目：找到并分类CT扫描图片里的肿瘤。此项目基于Part I里的基础知识而构建，并加了一些更多的高级的内容。短暂的Part III简短的介绍了PyTorch怎么把深度学习模型部署在设备上。

深度学习是一个很大的话题。在这本书里，我们仅仅包含了其中很小的一部分：用PyTorch以图片作为输入实现小规模的分类和分割任务。这本书重点在于PyTorch的实践，目的在于提供足够读者能解决实际问题的知识，比如cv，nlp，等。


### 1.3 为什么要选择PyTorch

正如之前所说，深度学习通过学习数据让我们能够解决一系列复杂的问题，比如机器翻译、玩策略类游戏等。为了能在实践中实现问题的解决，我们需要flexible的工具，所以他们可以被应用于一系列范围很广的问题里；而且可以利用大量的数据进行很多次的训练，而且我们还需要训练的网络对于各种各样的输入都能够表现得很好。基于上面这些目的，让我们来看看为什么我们需要用PyTorch。

PyTorch足够简单。很多研究员和程序员觉得它很容易学而且上手。PyTorch是Pythonic的，而且对于复杂的领域，他也有着充足的文档，对错误有所警告，也给出了实例。更具体地说，在PyTorch里编码深度学习框架是十分自然地。PyTorch给了我们一种data type（数据类型），Tensor，来处理numbers，vectors，matrices，或者最广义的arraies。而且，PyTorch还提供了能够操作Tensor的函数。我们可以就像在Python里一样应用这些函数。如果读者了解Numpy，那这些就显得很自然。

PyTorch还提供了两个特征使得其非常适合深度学习：首先，它支持利用GPU进行加速计算，经常比在CPU上进行同样的计算要快几十倍。其次，PyTorch支持对一般数学表达式进行数值优化，这被用于深度学习的训练。我们可以注意到，这两个特征对于一般的scientific computing都很有用，不仅仅只是针对于深度学习问题。实际上，我们完全可以将PyTorch描述为一个支持优化和科学计算的高性能的Python库。

一个PyTorch设计的驱动力是它的表达性，允许开发者实现足够复杂的模型，而不需要因为库本身的复杂性而困扰。PyTorch毫无疑问的提供了提供了一种在深度学习的范围内，将idea用Python代码实现的手段。

PyTorch在从研究和开发转向工业界的路上仍有一些很有意思的故事。因为它起初是为了研究而开发的，PyTorch一开始是基于C++而开发的，这样它就可以利用C++的快速性而且不需要依赖Python库，而且训练和部署都是用C++做的。

#### 1.3.1 十分具有竞争力的深度学习框架领域

在PyTorch最初beta版本发行的时候：
* Theano和TensorFlow已经成为了low-level库的翘楚，它们都是由用户定义一个computational graph来定义模型。
* Lasagne和Keras是Theano high-level的wrapper，而且Keras还是wrapper了TensorFlow和CNTK。
* Caffe，Chainer，DyNet，Torch（基于Lua，是PyTorch的先驱），MXNet，CNTK，DL4J都是当时流行的框架。

在接下来的两年里，这个领域发生了很大的变化。PyTorch和TensorFlow成为了仅有的两大巨头，其它的框架仅仅在特定的领域才会被使用到。
* Theano：第一个深度学习框架，已经停止了开发。
* TensorFlow：完全吸收了Keras，并将其作为TensorFlow最高层的API；提供了一个中间层，eager mode，类似于PyTorch的计算逻辑；发行了TensorFlow 2.0。
* JAX：也是Google开发的库，完全独立于TensorFlow，可以被认为是具有autograd，JIT和GPU版本的Numpy三种功能的库。
* PyTorch：吸收了Caffe2；更换了绝大多数用Lua写的Torch项目的底层代码；加上了一个graph mode，叫做TorchScript。

TensorFlow拥有一个健壮的可用于生产的pipeline，一个广泛的工业界的环境，以及大量的市场占有率。PyTorch因为易于上手获得了很多研究员、学生老师的青睐，并且随着他们进入工业界从而开始流行。TorchScript和TensorFlow的eager mode可以认为是它们双方互相向对方学习的点。


### 1.4 综述：PyTorch如何支持deep learning项目

我们已经暗示了一些PyTorch的building blocks。现在我们正式的构建一个PyTorch high-level的map。我们可以从一个深度学习项目需要从PyTorch中得到什么来研究这个问题。

首先，PyTorch以从Python中继承的Py作为开头，但是仍然有很多底层代码并不是由Python所写。实际上，为了performance，绝大多数PyTorch的底层代码都是用C++和CUDA（一个由NVIDIA开发的类似于C++的语言，被编译后可以进行GPU上大型的并行计算）写的。PyTorch提供了用C++直接运行的方式，将在Chapter15里说。提供这种能力的一个原因是它使得在设备上部署模型成为可能。然而绝大多数时间我们还是从Python的角度来研究PyTorch，包括构建模型，训练，以及用训练好的模型来解决实际问题。

PyTorch的Python API是解释PyTorch作用以及将其与Python生态系统结合的地方。

正如我们已经了解的，在它的核心，PyTorch提供了multi-dimensional arrays（或者在PyTorch的语言里，tensors）的库（我们将会在Chapter3里仔细研究细节），也提供了tensors上各种不同的operations的库，在torch模块里。tensors和他们的operations都可以在CPU和GPU上进行运算。将计算从CPU迁移到GPU上用PyTorch实现只需要一到两行代码。PyTorch另一个核心就是它给tensor提供了一种能力，这种能力可以让每个tensor追踪定义在其上的运算，并且给出任何一个output关于其任意一个input的derivative。这个将会被用于数值优化，这些derivative由tensor来提供，但实际上由PyTorch的autograd模块来进行计算，并提供给tensor。

通过拥有tensor和autograd-enabled tensor标准库，PyTorch可以被用于物理，渲染，优化，仿真，建模，以及更多。但是PyTorch最重要的就是它是一个深度学习库，所以它提供了所有构建以及训练一个neural networks所需要的building blocks。下图展示了一个标准的过程，包括了数据loading，模型训练，以及部署模型。

![Framework]({{ '/assets/images/DLP-1-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. Basic, high-level structure of a PyTorch project, with data loading, training, and deployment to production.*

构建neural networks的最核心的PyTorch module被放在torch.nn里，这个module提供了常见的neural networks layer以及其它的架构组成部分。fully connected layer，convolutional layer，activation function，loss function也可以在这个module里被找到。这些组成部分可以被用于构建以及初始化未被训练的模型（figure2里中间部门的untrained model）。为了训练我们的模型，我们需要额外几件事情：一个source of training data，一个optimizer用来将这个模型适用于这些training data，以及将模型和数据部署到硬件设备上的方法（这样才能真正的开始训练和计算）。

在figure2的左侧，我们可以看到在training data到达模型之前，需要进行一些data processing。这种data processing并不是preprocessing，而是对数据的处理，从而使得能够适应PyTorch的训练模型，这在真实的项目中占有十分重要的位置（原始数据必须变成PyTorch能操作的tensor才可以）。首先，我们需要physically获取数据，大多数情况下是从某种存储设备中获取的。然后我们需要将数据里的每一个sample都转换成PyTorch能够实际上操作的东西：tensor。这个在用户定义的数据（可以是任意形式的）和标准的PyTorch tensor中充当桥梁作用的是定义在torch.utils.data module里的Dataset类。这个过程对于每个任务都不太一样，我们需要自己来实现。本书将会在Chapter4里展示各种不同类型的数据该如何表示为tensor的形式。

因为数据存储通常都很慢，特别是由于access latency，我们通常希望并行化处理data loading。但是Python并不包含并行处理的函数，所以我们需要额外的步骤来操作，将数据整合为batches（批次）：包含多个samples的tensor。这样做很细致，但同样也足够一般化，所以PyTorch在DataLoader类中提供了该方法。这个类的实例可以大量产生子过程用来从dataset里load data，之后就可以被training loop使用了。

Dataset和DataLoader的内容将会在Chapter7中介绍。

>DataLoader会用在Dataset处理之后。Dataset负责将各种类型的数据转化为tensor，DataLoader负责将转化后的tensor整合为batch，仍然是个tensor。

当可以生成sample batches的部分定义好之后，我们就可以来处理training loop本身了。一般情况下，这个training loop都是一个标准的Python for循环。在最简单的情况下，模型在本地CPU或单个GPU上完成计算。这种情况也是我们这本书里所假设的情况。

在training loop的每一步，我们都要用从data loader里获取的sample来评价我们的模型。我们可以用某种criterion或者loss function来比较模型的输出和想要的输出之间的差别。就像PyTorch为我们提供了构建model的函数，他也给我们提供了很多loss function函数或类。这些同样在torch.nn模块里。在我们比较完模型的输出和想要的输出后，我们需要将模型调整一点从而使得它的输出更加靠近所想要的输出。正如之前所提到的，这就是PyTorch autograd所起作用的地方；但我们仍然需要一个optimizer来做更新（因为autograd只是整合了gradient），这些optimizer被放在了torch.optim里。我们将会在Chapter5里开始用loss functions和optimizers，在Chapter6-8里进一步熟悉这些内容，然后在Part II里正式接触大的项目。

现在用更复杂精细的硬件系统变得越来越常见了，比如多块GPU，或者多个设备共同训练一个大的模型，如figure2下面所显示的那样。在这些情况里，torch.nn.parallel.Distributed-DataParallel和torch.distributed子模块可以被用来处理复杂硬件的部署。

training loop可能是整个深度学习任务里最无聊但耗时最长的部分。在训练结束后，我们将会得到一个模型，它的参数已经在我们的任务中优化好了：如figure2里的trained model。为了让这个模型变得有用，我们需要将它合理的放置部署。这个过程在figure2的右边进行了描述，一般涉及将模型放在服务器或上传到云端，或者将其与一个更大的项目相整合，或者集成到手机等等。

放置部署训练好的模型里特殊的一种是将训练好的模型导出（export the model）。正如之前所提到的，PyTorch默认导出到一个立即可以执行的模型（eager mode）。每当一个有关PyTorch的指令在Python interpreter里出现了，相对应的操作就会通过C++或者CUDA立刻执行。随着更多的tensor上的指令提出，更多的底层的操作将会被执行。

PyTorch还提供了一种方法可以预先编译模型：TorchScipt。用TorchScript，PyTorch可以将模型序列化成一系列指令，这些指令可以在与Python解释器完全独立的情况下被调用：比如在C++代码里或者在移动设备上。我们可以将其想象成一种拥有有限指令集的虚拟机，特别用于tensor operations。这可以让我们将模型正常的导出，也可以导出为一种叫ONNX的标准形式。这些内容是PyTorch在生产部署的基础，将在Chapter15里说。





## Chapter 2 Pretrained networks

>这章包括的内容：
>* 运行预训练好的图像识别模型
>* GAN和CycleGAN的一个介绍
>* 生成图片文字描述的Captioning模型
>* 通过Torch Hub分享模型

computer vision毋庸置疑是一个被深度学习深深影响的领域，其中有很多的原因：分类或者从原始图片中理解图像内容的需求是存在的，大量的数据变得可以获得，以及新的网络结构比如convolutional layer被发明出来，而且GPU被应用于加速计算。

我们将会学习如何下载和运行其他的研究者已经在大规模数据库上训练好的模型。我们可以把训练好的模型理解成为一段program，能够输入input并输出相应的output。这段program的行为被网络结构、训练数据以及输出所需要满足的条件所精心设计。

在这章里，我们将会三种流行的预训练模型：一个可以根据图像内容给图像贴上标签的模型，一个可以从一张真实图片生成新图片的模型，以及一个可以用句子描述图像内容的模型。我们将会学习如何在PyTorch里下载并运行这些模型。我们也会介绍PyTorch Hub，它是一个工具集，PyTorch模型可以在这个标准化的用户界面上被处理。沿着这个方向，我们将会讨论data source，定义如label这种的术语。

学习如何用PyTorch来运行一个训练好的模型是很有用的，尤其是当这个模型在一个很大的数据集上被训练过。我们将会习惯于给真实世界的数据设计并运行neural networks，不论这个networks是否经过了训练。

### 2.1 一个能够识别图像主体内容的预训练模型

作为我们对deep learning的第一个了解，我们将会运行一个deep neural network，这个网络被object recognition任务预训练过。有很多预训练的网络可以被源代码仓库所获取。研究者们随着论文的发表而公布自己的源代码是很常见的，而有些研究团队还会提供已经预训练好的网络。用这些预训练好的模型可以让我们直接利用这些模型完成一些任务，或者进行进一步的设计。

我们这里所用的预训练模型是在ImageNet这个数据集上预训练的。ImageNet是Stanford University做的一个超过14 000 000张图片的数据集，它的每张图片都有分层级的名词标签，这些标签来自于WordNet数据集，这是一个英文的大型词汇数据集。

ImageNet数据集，和其它的公共数据集一样，是在academic competition里发展出来的。这些competitions传统上是很多研究院和公司里的研究员打比赛的地方。ImageNet Large Scale Visual Recognition Challege（ILSVRC）从2010年开赛以来就获得了很广泛的关注。这个competition基于一些任务，这些任务每年都不一样，比如image classification（告诉图像所含的物体是什么类别的），object localization（给出图片中物体的位置），object detection（鉴别图片中的物体并给出它们的类别），scene classification（给整张图片分类），以及scene parsing（将图片分割为有语义的区域，比如牛，房子，奶酪，帽子等）。我们细说image classification，这个任务是以一张图片作为输入，并在1000个类别中给出5个类别，按置信度高低排序，来描述图像中的内容。

ILSVVRC的训练集包括12 000 000张图片，这些图片标记有唯一的名词（1000个类别），这个名词叫做图像的class。在这个任务里，我们将label和class混用，表达同一个意思。figure 1给出了ImageNet里一些图片。

![ImageNet]({{ '/assets/images/DLP-2-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. A small sample of ImageNet images.*

我们这个任务（image classification）的最终的效果是，给训练好的模型一张新的图片，它能给这张图片返回一个预测的label的列表（在这个例子里列表里包含五个类别），从而我们可以通过这个label list来了解图像的内容。如figure 2所示。

![Inference]({{ '/assets/images/DLP-2-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. The inference process.*

输入的图片将会先被处理为一个multidimensional array class torch.Tensor的实例。这是一张标有高度和宽度的RGB图片，所以这个tensor将会有三个维度：RGB三个通道，以及两个有指定大小的图片spatial dimensions（tensor将会在下一章Chapter 3里详说）。我们的模型将会以上述的处理后的图片作为输入，并给出每个class的分数。之后每个class被一一对应到class label上。输出被存在一个拥有1000个element的torch.Tensor里，每个element表达该class的分数。

在我们正式开始之前，我们先了解一下这个网络本身，看看它的结构，以及学习一下在模型使用前该如何处理数据。

#### 2.1.1 为image recognition任务获取一个预训练的模型

我们现在去找一个在ImageNet上预训练好的网络。[TorchVision](https://github.com/pytorch/vision)项目包括了一些computer vision领域表现最好的模型，比如AlexNet，ResNet，Inception V3等等。在Pytorch里同样很容易获取如ImageNet这样的数据集。之后我们将会介绍更多这样的PyTorch里已经包括或者容易获得的外部资源。现在，我们下载两个网络：AlexNet和ResNet。

预训练好的模型可以在torchvision.models里获得：

```python
# In [1]:
from torchvision import models

# In [2]:
dir(models)

# Out [2]:
['AlexNet',
 'DenseNet',
 'InceptionV3',
 'ResNet',
 'SqueezeNet',
 'VGG',
...
 'alexnet',
 'densenet',
 'densenet121',
...
 'resnet',
 'resnet101',
 'resnet152',
...
]
```

上面首字母大写的那些名字都是Python的class，每一个class都对应着一系列模型，这些模型的内部结构都有着细小的区别。而小写字母的名字是function，它们会返回大写字母对应的那些class里的实例，这样有时候会更方便。比如，resnet101会返回一个ResNet的实例，其有101层，resnet18则会返回一个有18层的ResNet的实例。我们现在将注意力放在AlexNet上。


#### 2.1.2 AlexNet

AlexNet赢了2012 ILSVRC比赛，比第二名高了很多，top-5 error rate是15.4%，第二名是26.2%（第二名不是基于deep Learning的）。这是computer vision发展史上值得记住的时刻：人们意识到了deep learning对于vision任务的巨大潜力。从此开启了对这个比赛持续性的基于deep learning的改进，最低能使得top-5达到3%。

以今天的衡量标准来看，AlexNet是个相当小的网络。但是这是对刚了解neural network以及想上手操作的人一个非常好的例子。

AlexNet的结构如figure 3所示。在图中，输入image从最左侧开始，经过了五个filters，每个给出一系列输出images（我们可以把这些filters理解成一系列加法乘法并最后加上一个activation函数的filter，以一系列图片为输入，一系列图片为输出）。在经过每个filter之后，images的大小被缩小了。最后一个filter输出的image被展平成4096个element的一维向量，并再加上几层全连接层来产生1000个输出概率，每个对应着一个class。

![Architecture]({{ '/assets/images/DLP-2-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 3. The AlexNet Architecture.*

为了运行AlexNet，我们可以构建AlexNet class的一个instance：

```python
# In [3]:
alexnet = models.AlexNet()
```

现在alexnet就是一个具有着上图中AlexNet结构的object。通过给alexnet一些准确sized的输入数据，我们可以运行forward pass，将这个输入数据通过alexnet。从实际操作来说，就是我们有input object，这样可以运行forward pass：output = alexnet(intput)。

但是如果我们这样做的话，是没有意义的。因为整个网络参数是没有被学习过的。我们需要训练这个网络或者从预训练的网络上加载学习好的参数。我们回到models模块。我们知道大写字母开头的对应着computer vision领域流行的网络结构，它们每个都是一个class。小写字母的那些是functions，输出的是某个class的instance，这个instance有给定好的网络层数，并且可以将预训练好的参数加载到这个instance里。注意到，用这些小写字母代表的函数没有什么特别的，他只是使得生成一个满足条件的instance变得更容易而已（要不然还需要挨个定义层数等等）。


#### 2.1.3 ResNet

用resnet101 function，我们可以实例化一个101层的convolutional neural network。在2015年residual networks发明之前，训练一个很深的网络是十分困难的。residual networks克服了训练很深的网络的困难，并在提出的那一年在多项比赛上打败了其它的网络结构。

我们来构建一个ResNet的实例。我们给resnet101 function传递一个argument，来告诉这个function去下载已经在ImageNet上预训练好的参数：

```python
# In [4]:
My_resnet = models.resnet101(pretrained=True)
```


#### 2.1.4 Ready, Set, Almost run

我们来看一下一个resnet101到底是什么样的。我们可以通过printing所返回的实例来看。结果是一个文字化的表述，描述了网络结构的细节。

```python
# In [5]:
My_resnet

# Out [5]
ResNet(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                    bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
                       track_running_stats=True)
    (relu): ReLU(inplace)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1,
                        ceil_mode=False)
    (layer1): Sequential(
       (0): Bottleneck(
...
       )
    )
    (avgpool): AvgPool2d(kernel_size=7, stride=1, padding=0)
    (fc): Linear(in_features=2048, out_features=1000, bias=True)
)
```

上面所示的每一行都是一个模块。这个模块和Python里的module是不同的概念，这里每个模块代表着一种运算，是neural networks的组成部分。这些模块也成为neural networks的layers。

resnet101 function所返回的实例里，有很多重复的Bottleneck的模块（实际上有101个），每个都包含了convolutions以及其它的运算。resnet101是个典型的深度神经网络结构：由众多的层堆积而成，并以fully connected layer收尾，从而输出一个向量来代表每个种类的分数。

My_resnet变量可以被当作一个function来调用，以一张或多张图片作为输入并输出1000个类别的分数。在我们用它之前，我们还需要对输入图片做一些preprocess，从而使得这些图片有正确的size以及它们的pixel value大致落在一个合理的区间内。为了实现这样的preprocessing，torchvision module提供了transforms，可以允许我们很快定义包含基础preprocessing function的pipeline：

```python
# In [6]:
from torchvision import transforms
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
```

在这种情况下，我们定义了一个preprocess function，这个function将输入图片scale到$$256 \times 256$$的大小，按中心区域crop一块$$224 \times 224$$的区域，transform这个图片成一个tensor（一个PyTorch multidimensional array：在这种情况下，是一个3D array，维度分别代表color，height和width），并沿着color维度normalize，从而使得RGB维度的tensor都具有指定的mean和standard devariations。在Chapter7的7.1.3里我们将会更加深入的介绍transforms。

我们现在可以找一张图片，preprocess它，再将它给到My_resnet，来看看ResNet会给出什么样的结果。我们可以用Python的Pillow module来上传一张本地文件系统里的照片，Pillow是个很好用的图片处理module：

```python
# In [7]:
from PIL import Image
img = Image.open("../data/p1ch2/bobby.jpg")
```

如果我们用的是Jupyter Notebook，那我们还可以inline的看一下照片是什么样的（在实际中，照片会显示出来，替代下面<PIL.JPegImagePlugin.JpegImageFile image mode=RGB size=1280x720 at 0x1B1601360B8>的部分）：

```python
# In [8]:
img

# Out [8]:
<PIL.JPegImagePlugin.JpegImageFile image mode=RGB size=1280x720 at 0x1B1601360B8>
```

或者我们还可以用show() method，这样会弹出一个带有viewer的窗口，就可以直接看到图片：

```python
>>> img.show()
```

![Bobby]({{ '/assets/images/DLP-2-4.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 4. The picture of Bobby, the output of img.show().*

然后，我们就可以把输入的图片通过我们的preprocessing pipeline函数：

```python
# In [9]:
img_t = preprocess(img)
```

我们还可以reshape，crop，normalize这个input tensor，在后两章会详细说。

```python
# In [10]:
import torch
batch_t = torch.unsqueeze(img_t)
```

#### 2.1.5 Run!

对输入的新数据运行一个已经训练好的模型，这个过程叫做inference。为了inference，我们需要将这个network的mode变成eval：

```python
# In [11]:
My_resnet.eval()

# Out [11]:
ResNet(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                    bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
                       track_running_stats=True)
    (relu): ReLU(inplace)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1,
                        ceil_mode=False)
    (layer1): Sequential(
       (0): Bottleneck(
...
       )
    )
    (avgpool): AvgPool2d(kernel_size=7, stride=1, padding=0)
    (fc): Linear(in_features=2048, out_features=1000, bias=True)
)
```

如果我们忘记将其mode改成eval，一些内部的模块，比如batch_normalization和dropout，将会给出没有意义的结果，因为它们内部的机制使得它们在训练和eval模式下运行的不一样。现在模型的mode已经是eval了，我们就可以开始inference了：

```python
# In [12]:
out = My_resnet(batch_t)
out

# Out [12]:
tensor([[ -3.4803, -1.6618, -2.4515, -3.2662, -3.2466, -1.3611,
          -2.0465, -2.5112, -1.3043, -2.8900, -1.6862, -1.3055,
...
          2.8674, -3.7442, 1.5085, -3.2500, -2.4894, -0.3354,
          0.1286, -1.1355, 3.3969, 4.4584]])
```

我们现在需要找到上面输出的1000维的向量里哪一个有着最高的score。这将会告诉我们这个模型在这张输入图片中看到了什么。

为了获得这个网络到底给这张图片标了什么标签，我们需要下载一个text file，这个text file列出了所有的可能的类别，而且这些类别的排列顺序和网络训练时输出的排列顺序是对应的。几乎所有的image recognition任务都有着和这个问题相似的输出结构。

我们下载一下含有1000个标签的ImageNet数据集的text file：

```python
# In [13]:
with open(../data/p1ch2/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]
```

现在我们需要找到out这个tensor里最大score所对应的index是什么。我们可以用PyTorch里的max function来做，这个function会输出一个tensor的最大值以及这个最大值的位置：

```python
# In [14]:
_, index = torch.max(out, 1)
```

我们现在可以用这个index来确定标签了。但是上述得到的index是一个PyTorch的tensor，一个one-element，one dimensional的tensor，而并不是一个Python的number（实际上应该是tensor($$\left[207\right])$$）。所以我们需要得到这个index的实际值，可以通过index$$\left[0\right]$$来得到。我们再通过torch.nn.functional.softmax来normalize输出到$$\left[0,1\right]$$的范围。softmax的结果有点像模型对于每个类别给出一种置信度。在我们的例子里，模型有96%的自信输入的图片是一只金毛：

```python
# In [15]:
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()

# Out [15]:
('golden retriever', 96.29334259033203)
```

既然模型输出了长度为1000的向量来对每个类别都给了分数，我们还可以找到分数排名第二高，第三高等等所代表的标签。我们可以用sort function来对原out进行排序，这个function会输出排序后的序列，并且会输出排序后序列里每个element在原序列里的位置：

```python
# In [16]:
_, indices = torch.sort(out, descending=True)
[(labels[idx], percentage[idx].item() for idx in indices[0][:5]]

# Out [16]:
[('golden retriever', 96.29334259033203),
('Labrador retriever', 2.80812406539917),
('cocker spaniel, English cocker spaniel, cocker', 0.28267428278923035),
('redbone', 0.2086310237646103),
('tennis ball', 0.11621569097042084)]
```

我们可以看到前四名都是狗，之后就变得奇怪了。tennis ball是第五个结果。这是一个显示neural networks和人类在vision领域的判断区别有多大的例子。


### 2.2 A pretrained model that fakes it until it makes it

假设现在我们是想要模仿知名画家的画的职业罪犯。我们不是画家所以并不会画画。如果我们只是自己在不断地练习，很难达到模仿画家绘画的水平。但是如果我们请一个对所要模仿的画家很了解的专家来评判我们画的画，对于不像的部分进行改进，那我们的绘画水平就会朝着该画家的绘画风格而进步。

而GAN所要做的事情，和上述故事有着很大的相似之处。我们可以制作假的数据，而其风格和我们所看到的数据很类似，不管是image还是video都可以。

#### 2.2.1 The GAN game
在deep learning圈子里，我们刚讨论的故事就是GAN game。这个game有两个networks，一个作为绘画者，而另一个则是评判专家，他们两相互比赛来超过对方，绘画者要画出让专家尽可能无法判断真伪的画，而专家要尽可能地判断正确画的真伪。GAN是generative adversarial network的简称，其中generative表示某些东西被创造了出来（在GAN里就是假画），adversarial表示两个networks在互相竞争。GAN是近些年deep learning领域最重要的原创性工作之一。

记住我们最终的目标是生成尽可能像真data的假example。当和真data混合在一起的时候，有经验的专家也无法分辨他们的真伪。

generator网络扮演绘画者的角色，任务是创造尽可能以假乱真的照片，输入是一个任意值。discriminator网络扮演评判专家的角色，需要判断给的画是出自于绘画者还是来自真的画的集合。这样的两个网络设计是非典型的，但是在GAN这个game里，它们将会给出很惊人的结果。

figure5粗略的展示了GAN的结构。generator的最终目标就是迷惑discriminator使得它无法分辨真实和假的图画。discriminator的最终目标就是能判断画的真假。在最开始，generator画的画很容易被判断出来是假的。随着训练过程的进行，从discriminator返回的信息被generator利用来精进绘画技巧。在训练结束后，generator可以画出以假乱真的画，这时discriminator已经无法分辨画的真假了。

![GAN]({{ '/assets/images/DLP-2-5.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 5. Concept of a GAN game.*

值得注意的是，discriminator和generator并不是真的在比赛。这两个网络分别基于对方的输出而进行训练，对方网络的输出驱动了它们各自网络参数的优化。

GAN game可以使得generators仅仅从噪声和conditional signal（比如某种attribute，对于脸来说，young，female，glass等）或者从另一个image来生成足以以假乱真的图像。

#### 2.2.2 CycleGAN

GAN的一个有趣的变种是CycleGAN。CycleGAN可以将一个domain内的图片变成另一个domain内的图片，而不需要我们在训练集中提供这两个domain图像的pairs。

在figure6里，我们展示了将马变成斑马，以及将斑马变成马的网络结构。这个网络里有两个分开的generators，并且都有着各自的discriminators.

![CYCLEGAN]({{ '/assets/images/DLP-2-6.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 6. A CycleGAN trained to the point that it can fool both discriminator networks.*

如figure6所示，第一个generator要学习如何生成满足一个target distribution（在这个例子里是斑马）的图片，而其出发点则是满足属于另一个distribution（在这个例子里是马）的图片，从而它对应的discriminator无法分辨这个图片是从一张马的图片生成的还是本身就是一个斑马的图片。同时，这也是名字里Cycle的由来，这个生成的假的斑马图片作为另一个generator的输入，这个generator要学会如何从输入的斑马图片生成马的图片，并且也有一个相应的discriminator来分辨真假。构建这样一个循环，能够使得GAN的训练更加稳定，这也是GAN在实际应用中很重要也很常见的一个问题（难以训练）。

有意思的是，我们并不需要输入马/斑马一对照片作为输入。从一系列没有联系的马和斑马照片，就足以让generators学会各自的生成，这个已经超过了单纯的supervised learning的范围。这个模型所蕴含的道理甚至还有更深：generator学习如何选择性的改变物体的appearance而并不需要告诉他物体是什么。并没有任何信息告诉generator鬃毛和腿该如何对应，以及它们分别是这个图里的哪部分，但generator仍然可以对应的将其转换。

#### 2.2.3 A network that turns horses into zebras

CycleGAN被一系列没有关联的从ImageNet里提取出来的马和斑马的图片数据集所训练。这个网络学会以一张或者多张马的图片作为输入并将它们全部转换为斑马的图片，并保证图片中除了马或斑马的部分保持不变（环境不变）。运行已经预训练好的CycleGAN可以让我们更进一步了解一个network——在这个网络里，一个generator——是如何被implemented的。

我们定义一个ResNetGenerator class。实例化这个class，使用default parameters就可以：

```python
# In [2]:
netG = ResNetGenerator()
```

netG model被创建了，但是它的参数仍然是随机的。我们可以加载已经预训练好的这个model的参数。model的参数被存在一个.pth的文件里，这个文件是一个包含着模型参数的pickle file。我们可以用model的load_state_dict method将它们加载到ResNetGenerator里：

```python
# In [3]:
model_path = '../data/p1ch2/horse2zebra_0.4.0.pth'
model_data = torch.load(model_path)
netG.load_state_dict(model_data)
```

从而现在netG已经获得了预训练所得的所有的知识。注意到，此处加载预训练好的模型参数和2.1.3里给resnet101加载预训练参数是一样的，只不过torchvision.resnet101 function隐藏了加载参数的过程。

如同resnet101例子一样，我们将模型mode改成eval：

```python
# In [4]:
netG.eval()

# Out [4]:
ResNetGenerator(
    (model): Sequential(
...
    )
)
```

这个网络以一张图片为输入，识别图片里有几只马，之后分别将这些马所对应位置的像素进行更改，从而将原图片里的马变成斑马输出。

我们现在可以随意输入一张关于马的图片，来看看我们的generator会输出什么。首先我们需要PIL和torchvision模块：

```python
# In [5]:
from PIL import Image
from torchvision import transforms
```

之后我们做一些输入的transformations，从而使得输入能够匹配网络：

```python
# In [6]:
preprocess = transforms.Compose([transforms.Resize(256),
                                 transforms.ToTensor()])
```

打开一张马的照片：

```python
# In [7]:
img = Image.open("../data/p1ch2/horse.jpg")
img
```

![horse]({{ '/assets/images/DLP-2-7.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 7. A man riding on a horse.*

之后，我们将照片通过preprocess function得到恰当shape的变量：

```python
# In [8]:
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t)
batch_out = netG(batch_t)

# batch_out是generator的输出，我们可以将其转换回一张图片

out_t = (batch_out.data.squeeze() + 1.0) / 2.0
out_img = transforms.ToPILImage()(out_t)

out_img.save("../data/p1ch2/zebra.jpg")
out_img

# Out [8]:
<PIL.Image.Image image mode=RGB size=316x256 at 0x23B24634F98>
```

上述代码里输出的<PIL.Image.Image image mode=RGB size=316x256 at 0x23B24634F98>实际上是下图：

![zebra]({{ '/assets/images/DLP-2-8.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 8. A man riding on a zebra.*


### 2.3 A pretrained network that describes scenes

我们已经见过了识别图片的模型，生成新图片的模型，而这个则是融入了自然语言的computer vision的模型。我们将会介绍一个预训练好的image captioning model，NeuralTalk2 model。这个model的输入是一张图片，而输出则是对这张图片所含内容的英文描述，如figure9所示。这个模型被一个很大的数据集训练，这个数据集里的数据是图片和描述图片的文字pair所构成的。

![caption]({{ '/assets/images/DLP-2-9.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 9. Concept of a captioning model.*

这个model由两个互相连接的部分所组成。一个部分负责输出具有代表性的描述性的representations，作为另一部分的输入。另一部分是一个recurrent neural network，可以生成句子。这两个部分在image/captioning pair上共同被训练。

这个模型的第二部分叫recurrent是因为它以一个接一个的方式生成每一个output（individual word），每一次的input都包含了上一次的输出。这样就使得后面的word都和前面的word有关联，而这在生成句子的任务里正是我们所需要的。

#### 2.3.1 NeuralTalk2

[NeuralTalk2](https://github.com/deep-learning-with-pytorch/ImageCaptioning.pytorch)可以以如下的方式来运行，我们要先将一些图片放在data文件夹下，然后运行脚本：

```python
python eval.py --model ./data/FC/fc-model.pth --infos_path ./data/FC/fc-infos.pkl --image_folder ./data
```

试一下2.2里的马的图片，结果是 A person riding a horse on a beach。很准确。


### 2.4 Torch Hub

预训练好的模型在deep learning发展的早期便被放在了网上，但是直到PyTorch 1.0的发行，用户并没有一个统一的界面来获得这些预训练模型。

PyTorch 1.0介绍了PyTorch Hub，这个机制可以让作者以某种PyTorch可以理解的方式，通过Github来将已经预训练好的或者没有预训练的模型发布出来。这使得发布模型变得很方便。torchvision模块包含了一些模型，但只是很少一部分，而PyTorch Hub的发布，使得作者可以随意发布自己的模型供其他人参考使用。

通过Torch Hub机制发布模型，作者仅仅需要在该项目的Github repository的根目录下加一个hubconf.py文件。这个文件有着以下的结构：

```python

# 这个文件有着两个部分

# 第一部分是你的代码所需要依赖的modules
dependencies = ['torch', 'math']

# 第二部分是一个或多个函数，这些是用来给用户了解你的模型和代码用的。这些函数需要有根据输入的参数初始化模型的作用。

def some_entry_fn(*arg, **kwargs):
    model = build_some_model(*args, **kwargs)
    return model
    
def another_entry_fn(*args, **kwargs):
    model = build_another_model(*args, **kwargs)
    return model
```

我们查找感兴趣的预训练模型，我们现在就可以搜索Github repository看看哪个包含hubconf.py文件，只要包含了，我们就可以通过torch.hub模块来调用它。我们来看一个实际中的例子。为了方便起见，还是从torchvision看起，因为它提供了一个如何和Torch Hub互动的清晰的例子。

我们访问torchvision repository[所在的位置](https://github.com/pytorch/vision)发现它含有一个hubconf.py文件。这点check，说明这个项目可以用torch.hub module调用。我们要做的第一件事就是看看hubconf.py文件，找到repository的入口处。在Torchvision这个例子里，hubconf.py文件里有两个函数，resnet18和resnet50。我们已经知道这两个函数的作用：它们分别返回一个18层和一个50层的ResNet model。我们同样看到hubconf.py文件里的入口函数有一个pretrained keyword argument。如果是True，返回的model就是已经在ImageNet上预训练过的。

现在我们知道了repo，知道了入口函数，也知道了所需要了解的入口函数的argument，用torch.hub下载模型所需的全部知识我们就都有了，即使不clone那个repo也可以，PyTorch会帮我们解决：

```python
import torch
from torch import hub

resnet18_model = hub.load('pytorch/vision:master',
                          'resnet18',
                          pretrained=True)

# 'pytorch/vision:master'是Github repo的name和branch，'resnet18'是入口函数，entry-point function，的name，pretrained=True是keyword argument
```

上述代码会下载pytorch/visio repo的master branch的一个镜像，随着模型的参数，一起存到本地（系统默认存在home文件夹的 .torch/hub里），并运行resnet18 entry-point function，从而返回一个实例化的model。取决于具体的运行环境，Python可能会报错module缺失，比如PIL。Torch Hub不会主动下载缺失的module，但是它会给出具体缺失什么module的error。

我们现在就可以用返回的model，给出适当的argument，来运行一个forward pass整个模型了。Torch Hub的好处是所有按照上述机制发布的Github repo都可以被其他人所获取了，而不仅仅是torchvision里一小部分的model。

值得注意的是entry-point function需要返回model；但是严格来说，它们并不是一定要这样。比如在一个hubconf.py文简历，我们可以定义一个entry-point function的作用是将output probability转换为文字描述，而另一个的作用是transform输入。或者可以定义一个entry-point function输出model，另一个entry-point function来对这个model做进一步的处理。这种flexibility为将来PyTorch Hub的发展提供了无限可能。


### 2.5 Summary

* 一个预训练的网络是一个已经在某个数据集上被训练过的model。这样的网络在载入之后可以直接给出有用的结果。
* 通过学习如何使用一个预训练的网络，我们可以将一个预训练好的模型直接整合到项目里，而不需要对它有过多的了解或者对它进行训练。
* AlexNet和ResNet是两个十分著名的computer vision的model。
* GAN有两个部分：generator和discriminator。这两个部分相互促进来使得model生成足以以假乱真的data。
* CycleGAN的结构使得它可以在两个不同类的图片间互相转换。
* NeuralTalk2用了一种混合的模型架构，以图片为输入，生成描述该图片内容的文字。
* Torch Hub是一个从有满足条件的hubconf.py文件的任意Github项目载入model和model的参数的的标准的方法。


## Chapter3 It starts with a tensor

>这章包括的内容
>* 理解PyTorch里最基本的data structure：tensor
>* 在tensor上index和operate
>* tensor和Numpy的multidimensional array相互操作
>* 将tensor上的computation转移到GPU上以加速计算速度

在上一章里，我们看了一些各个领域的deep learning的pretrained model，并加以运行。它们都需要以某种格式的data做输入，比如image或者text，然后输出另外一种格式的data，比如labels，numbers，或者别的images、text。从这个角度来看，deep learning实际上就是将data从某种representation转换成另一种的一个system。这个转换是通过寻找所给的数据example里的共同点来学习到的。比如，这个system可能会注意到狗的一些普遍的特征以及金毛特殊的颜色，将这两个特征结合起来，系统能将输入的金毛图片转换为输出的代表金毛的label。学习后的系统能够广泛的识别相似的输入，并将它们归于同一个label。

上述的过程开始于将输入转换为floating-point numbers。figure1里的第一步，将输入的image pixels转换为numbers，这个内容将会在Chapter4里详细说。但在此之前，在这章里，我们将会介绍在PyTorch里如何用tensor来处理所有的floating-point numbers。


### 3.1 The world as floating-point numbers

既然floating-point numbers是一个network能够理解的处理信息的数据表达方式，我们就需要将各种各样的现实世界里的data encode成networks能够理解的数据格式（floating-point numbers），然后再将输出转换回我们人类能够更好理解的数据格式。

![floating-point number]({{ '/assets/images/DLP-3-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. A deep neural network learns how to transform an input representation to an output representation (Note: The numbers of neurons and outputs are not to scale.).*

一个deep learning network一般学习从一种数据格式到另一种数据格式的transformation，这表明在network中间每层之间的那些中间层数据可以被理解成一系列中间层的数据representations。对于image recognition来说，网络前面层的representations一般是一些edge detection或者某些特定的textures比如fur。而后面更深的层的representations就会表达一些更复杂的结构比如ears，noses等。

总的来说，这样的中间层representations是一系列的floating-point numbers，这些floating-point numbers表示了输入的内容和数据结构的特征，并对network如何描述从输入转换为输出这个过程有十分有用。这样中间层的representation是从输入的众多examples里学习到的。这些floating-point numbers（中间层representations）和对于它们的操作是现代AI领域最重要的内容，在这本书里我们会一直提及。

我们需要知道的是，这些中间层的representations（如figure1里的step2里的内容）是输入和前面层的neurons的weights相结合的产物。每个中间层representation对于每个输入都是特别的。

在学习如何将我们的数据转换为floating-point numbers之前，我们需要首先理解PyTorch如何处理以及存储各个层面的数据，包括输入，中间层的representations，和输出。这章将会详细讲解这方面的内容。

为此，PyTorch引入了一个基础的data structure：tensor。在Chapter2里，当inferece预训练模型的时候，我们已经遇到了tensor了。对于数学里的tensor而言，其和space，transformations等等相关，而数学里的tensor和PyTorch里的tensor并不是一个东西。在deep learning，或者PyTorch的领域，tensor指的是vector，matrix的任意维度的推广，我们可以在figure2里看到。tensor这个概念的另一个名字叫multidimensional array。一个tensor的维度和想要获取这个tensor里某个位置的scalar值所需要的index的长度是相同的（从figure2里也可看出）。

![tensor]({{ '/assets/images/DLP-3-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. Tensors are the building blocks for representing data in PyTorch.*

PyTorch并不是支持multidimensional array唯一的库。Numpy是目前最流行的multidimensional array库，它已经成为了data science领域的通用语言。PyTorch可以和Numpy无缝衔接，互相交互使用。从而PyTorch和众多的Python库比如SciPy，Scikit-learn以及Pandas都有交互。

和Numpy arrays相比，PyTorch tensors有一些”超能力“，比如能在GPU上很快的计算，能部署在多个设备上，以及能够跟踪计算图。这些都是现在deep learning库所需要的特点。

我们将首先介绍PyTorch tensor，介绍tensor的基本内容以便之后使用。第一步是学习如何用PyTorch tensor库来操作tensor。这些操作包括如何将数据存在内存里，如何在任意大的tensor上运行一些特定的operation，如何和Numpy相交互，以及如何加速GPU计算。理解tensor的能力以及它的API是学会PyTorch最重要和基础的内容。在下一章，我们将会把这章学习到的知识应用于实践，学习如何将各种形式的输入数据转化为neural network能处理的tensor。


### 3.2 Tensors: Multidimensional arrays

我们已经知道tensor是PyTorch里最基础的data structure。一个tensor就是一个array：一种存有一系列数据的data structure，这些数据里的每一个都可以用一列index来获取。

#### 3.2.1 From Python lists to PyTorch tensors

我们来看看Python里的list是怎么被index的，从而可以对比tensor的index操作。

```python
# In [1]:
a = [1.0, 2.0, 1.0]

# 我们可以access这个list的第一个element：
a[0]
a[2] = 3.0
a

# Out [1]:
1.0
[1.0, 2.0, 3.0]
```

用Python list来存储更高维度的数据也是很常见的，比如存储一个2D直线上点的坐标。但是用更高效的tensor data structure，很多类型的数据，比如image，time series，甚至是sentences，都能被tensor表示出来。我们将会在这章里介绍一些tensor上的operations用来操作tensors，这些operation可以被很高层的语言（Python）调用。

#### 3.2.2 Constructing our first tensors

让我们来构建第一个PyTorch tensor来看看它到底是什么样的。

```python
# In [2]:
import torch        # 导入torch module
a = torch.ones(3)   # 创建一个一维的tensor，size是3，每个element都是1
a

# Out [2]:
tensor([1., 1., 1.])

# In [3]:
a[1]

# Out [3]:
tensor(1.)

# In [4]:
float(a[1])

# Out [4]:
1.0

# In [5]:
a[2] = 2.0
a

# Out [5]:
tensor([1., 1., 2.])
```

在导入torch module之后，我们可以调用一个function来构建一个一维的tensor，其size是3，所有的值都是1.0。我们可以用index来access它的element或者更改它。虽然表面上看这个PyTorch tensor和Python的list没什么区别，但背后的运行机制完全不同。

#### 3.2.3 The essence（本质，要素） of tensors

Python的list或者tuple也可以包含一系列的numbers，它们是一系列Python objects的合集，每个number都是一个object，都分配给了单独的内存，而这些内存可能完全是分散开的，如figure3左边所示。而PyTorch的tensors或者Numpy的arrays，是一些包含着unboxed C numeric type的连续的内存空间的views，也就是说tensor或者Numpy array内的numbers是用unboxed C numeric type表示的，而不是Python的objects，而且这些数据在内存中的地址是连续的，如figure3右侧所示。假设PyTorch tensor或Numpy array的每个element都是一个32-bit（4 bytes）的float，如果要存一个一维的包含着1 000 000个element的tensor，则需要精确的4 000 000 bytes，以及一些额外的空间用来存储metadata（比如dimensions或者numeric type）。

>用Python list来存放数据，每个element都是一个单独的python object，分配了单独的内存，并有一些其他的信息。而且list本身也有这些element的信息从而才能找到他们。boxed说明每个element都是本身的value外面还包裹了本身的信息或供list查找的信息。而PyTorch tensor或者Numpy array是直接把这些element放在内存的一整块连续的位置上，从而不需要包裹，只需要存下来element的值就行，所以是unboxed的。从而这样节省了空间，也加快了运行速度。

![memory]({{ '/assets/images/DLP-3-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 3. Python object (boxed) numeric values versus tensor (unboxed array) numeric values.*

假设我们有一列坐标数据用来表示一个几何物体，比如说一个三角形的三个顶点的坐标（4，1），（5，3），（2，1）。不用Python的list来表示这个数据，我们可以用一个一维的tensor来存储x坐标在偶数index上，存储y坐标在奇数index上：

```python
# In [6]:
points = torch.zeros(6)  # 利用zeros函数构建一个所需要的size的tensor
points[0] = 4.           # 之后在overwrite我们所需要的值到这些0上。
points[1] = 1.
points[2] = 5.
points[3] = 3.
points[4] = 2.
points[5] = 1.

# 还有一种简便的方法，直接将一个Python list作为torch.tensor的参数：

# In [7]
points = torch.tensor([4., 1., 5., 3., 2., 1.])
points

# Out [7]:
tensor([4., 1., 5., 3., 2., 1.])

# 要获取第一个顶点的坐标：

float(points[0]), float(points[1])   # 输出是(4.0, 1.0)
```

上述方法有效，但是还是复杂了，我们可以使得tensor的每一项都直接就是一个顶点的坐标：

```python
# In [8]:
points = torch.tensor([[4., 1.], [5., 3.], [2., 1.]])
points

# Out [8]:
tensor([[4., 1.],
        [5., 3.],
        [2., 1.]])

# 我们直接把一个Python list传递给了torch.tensor函数。我们可以查看这个tensor的shape：

# In [9]:
points.shape

# Out [9]:
torch.Size([3,2])  # tensor的shape attribute直接返回了这个tensor的形状，但返回值仍然是个torch.Size object。

# 当我们知道了要构造的tensor的shape时，
# 我们也可以直接用torch的zeros或ones method直接实例化一个tensor
# 此时需要给ones或者zeros method提供一个包含shape信息的tuple：

# In [10]:
points = torch.zeros(3, 2)

# points = torch.zeros((3,2))是一个效果，所以说 a = (3,2), points = torch.zeros(a)也是一样的结果。

points

# Out [10]:
tensor([[0., 0.],
        [0., 0.],
        [0., 0.]])

# 我们现在access这个tensor里每个element就需要两个index了：

# In [11]:
points = torch.tensor([[4., 1.], [5., 3.], [2,. 1.]])
points[0, 1]

# Out [11]:
tensor(1.)

# 这个返回的是第1个点的第2个坐标，也就是y坐标。

# In [12]:
points[0]

# Out [12]:
tensor([4., 1.])

# 这个返回的就是第1个点。
```

上面这些index所返回的都仍然是tensor（除非用float函数强行转换掉，如float(points$$\left[0, 1\right])$$），这些tensor是内存里相同位置所存的数据的一个view。上面的points$$\left[0\right]$$是一个新的1D tensor，size是2，参考的是points这个tensor第一行的值。新的这个tensor并不代表新的内存被开辟，然后值被复制后放在新的内存，再将这些值打包返回给新的这个tensor。因为这样做非常的效率低。我们将会在这章的3.7的tensor的view再来详细介绍发生了什么。


### 3.3 Indexing tensors

如果我们想获得一个tensor，只包含之前的tensor的除去第一个point以外的所有的points，可以使用range index notation，这个在Python的list里也经常被使用。

```python
# In [1]:
some_list = list(range(6))
some_list[:]     # 返回这个list所有的值
some_list[1:4]   # 返回这个list从第2个值开始到第5个值结束，包括第2个但不包括第5个
some_list[1:]    # 返回这个list从第2个值开始到最后一个值的所有值，包括第2个
some_list[:4]    # 返回这个list从第1个值开始到第5个值的所有值，不包括第5个
some_list[:-1]   # 返回这个list从第1个值开始到最后一个值，不包括最后一个
some_list[1:4:2] # 返回这个list从第2个值开始到第5个值每隔两个取一个值，不包括第5个

# 上面是Python里的range index的用法，而PyTorch的tensor有着一样的用法。

points = torch.tensor([[4., 1.], [5., 3.], [2., 1.]])
points[1:]       # 返回这个tensor从第2个值到最后一个值，tensor([[5., 3.], [2., 1.]])
points[1:, :]    # 返回这个tensor维度1里第2个值到最后一个，维度2里所有的值，tensor([[5., 3.], [2., 1.]])
# 前一行代码没有提到维度2，所以默认取所有的，这两行代表的是一个意思，结果当然也是一样的。

points[1:, 0]    # 返回这个tensor维度1里第2个值到最后一个，维度2里第一个值，tensor([5., 2.])
points[None]     # 给这个tensor加上一个维度为1的新维度，这个新加的维度是在最外层的，就像unsqueeze function的效果一样，tensor([[[4., 1.], [5., 3.], [2., 1.]]])
```

除了range index以外，PyTorch还给tensor提供了另外的更强力的index方法，叫advanced indexing，将在下一章说。


### 3.4 Named tensors

tensors的各个dimensions（或axes）通常是某些东西的index（索引），比如image的pixel location，或者color channels。这个表明当我们想index一个tensor的时候，我们需要记住每个dimension代表的含义，并有效的按顺序的进行索引。但因为数据在作为tensor输入之后，中间层会将这个tensor经过很多次转换，所以跟踪每个dimension的含义是很容易出错的。

举一个具体的例子来说明上面所说的问题。想象我们有一个3D的tensor，比如2.1.4里的img_t，这是一个RGB的三通道的image（在下面的例子里我们就随机生成一个维度相同的tensor）， 我们想将它转为gray-scale（灰度图）。而生成灰度图所需要的RGB三通道的权重，就选一个很常见的生成亮度值的：

```python
# In [1]:
img_t = torch.rand(3, 5, 5)  # shape [channels, rows, columns]，这就是随机生成的RGB三通道的输入图片
weights = torch.tensor([0.2126, 0.7152, 0.0722])  # 用于RGB三通道的权重
```

我们希望我们的代码能更具有通用性，比如，从表示为标记着height和width的2D tensor的grayscale image变成RGB三通道的3D tensor，或者从一张输入image到一个batch的images。在2.1.4里，我们用torch.unsqueeze function给输入的3D RGB image tensor加了一个维度。类似的，我们也给输入的3D RGB image tensor加上batch这个新的维度，在我们的例子里，batch=2：

```python
# In [2]:
batch_t = torch.rand(2, 3, 5, 5)  # shape [batch, channels, rows, columns]，这就是随机生成的RGB三通道的输入图片，一共有两张
```

有时候我们加了batch这个维度，从而输入的tensor是4维的，那么RGB channels这个维度就在dimension 1，而有时候我们没加batch这个维度，从而输入的tensor是3维的，那么RGB channel这个维度就在dimension 0。但不管怎样，我们从末尾开始数，RGB channel这个维度永远在dimension -3。我们不用weights tensor给RGB三通道加权平均，只是单纯的取个平均数：

```python
# In [3]:
image_gray_naive = img_t.mean(-3)     # 将3D tensor img_t按照RGB channel这个维度取算术平均，得到一个2D tensor灰度图
batch_gray_naive = batch_t.mean(-3)   # 将4D tensor batch_t按照RGB channel这个维度取算术平均，得到一个3D tensor灰度图
img_gray_naive.shape, batch_gray_naive.shape

# Out [3]:
(torch.Size([5, 5]), torch.Size([2, 5, 5]))
```

现在变得更复杂一点，我们现在用weights tensor来给RGB三通道加权平均，而不仅仅像上面的例子那样直接取算术平均数。PyTorch允许我们将有相同shape的两个tensor进行multiply操作，当某个乘数的某一个dimension的size是1时，仍然是可以的。

>这就是为什么在2.1.4里要将img_t加上一个维度变成batch_t，尽管这个维度的size是1，但是ResNet生成的实例model默认输入的tensor的dimension是4，分别是(batch, channels, height, width)。

当PyTorch里做运算的两个operands的维度不一样的时候，PyTorch和Numpy一样提供了处理这种情况的一个简便的机制：broadcasting。意思是，将两个tensor的维度按右对齐对准，只要它们满足在对应的维度上的shape是相同的，或者某一个tensor这个维度上的shape是1，那么即使它们的维度长度不同，短的那个左侧将会被补上1，相当于将tensor的内容复制几份来扩充维度。

在我们这个例子里，我们可以用一个unsqueezed_weights这样一个tensor，它的维度是(3, 1, 1)，用它来乘以batch_t这个维度是(2, 3, 5, 5)的tensor，就可以得到结果为(2, 3, 5, 5)的tensor，其RGB channel的维度被乘以了weights相应的值，并且这个操作在batch的每个image和image的每个pixel上都做了操作：

```python
# In [4]:
unsqueezed_weights = weights.unsqueeze(-1).unsqueeze(-1)
img_weights = img_t * unsqueezed_weights
batch_weights = batch_t * unsqueezed_weights
img_gray_weighted = img_weights.sum(-3)
batch_gray_weighted = batch_weights.sum(-3)
batch_weights.shape, batch_t.shape, unsqueezed_weights.shape, batch_gray_weighted.shape

# Out [4]:
(torch.Size([2, 3, 5, 5]), torch.Size([2, 3, 5, 5]), torch.Size([3, 1, 1]), torch.Size([2, 5, 5]))
```

einsum是一种处理index的function，在PyTorch和Numpy里都有用，此处不再赘述。

通过上面的内容可以看到，这样的代码很容易出错，因为往往人们就忘了每个tensor每个dimension都是代表了什么意思。所以给tensor的dimension加上名字就变得合理了。

PyTorch1.3加了named tensor的特征。Tensor factory functions比如torch.tensor或torch.rand都可以接收一个叫做names的argument，这个argument必须是一个string sequence。

```python
# In [5]:
weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names=['Channels'])
weights_named

# Out [5]:
tensor([0.2126, 0.7152, 0.0722], names=('Channels',))
```

当我们已经有了一个tensor之后，我们还想给它加上names（但是不改变已经存在的那些），我们就可以调用这个tensor的一个method，refine_names。ellipsis省略号（...）可以允许我们跳过任意的dimensions。

```python
# In [6]:
img_named = img_t.refine_names(..., 'channels', 'rows', 'columns')
batch_named = batch_t.refine_names(..., 'channels', 'rows', 'columns')
print("img named:", img_named.shape, img_named.names)
print("batch named:", batch_named.shape, batch_named.names)

# Out [6]:
img named: torch.Size([3, 5, 5]), ('channels', 'rows', 'columns')
batch named: torch.Size([2, 3, 5, 5]), (None, 'channels', 'rows', 'columns')
```

利用rename这个sibling method，我们还可以覆盖或者丢弃原有的names（丢弃的话就传入None）：

```python
# In [7]:
img_new_named = img_named.rename(..., 'height', 'width')
batch_new_named = batch_named.rename(..., None)
image_new_named.names, batch_new_named.names

# Out [7]:
(('channels', 'height', 'width'), (None, 'channels' , 'rows', None))
```

>注意，不管是rename还是refine_names，内部只能最多有一个省略号。

当对两个tensor做operation的时候，PyTorch除了会进行dimension的验证（验证它们对应位置的dimension的shape是否一样或某个为1，如果dimension长度不同是否能够broadcast），它还会对names也进行检查。目前的PyTorch还不会根据names自动对齐dimensions，我们还需要在code里明确指出。align_as method会返回一个tensor，这个tensor会补全所缺少的dimensions而且将已经存在的dimensions按正确的顺序排列，而补上的缺少的dimensions都是shape=1的。

```python
# In [8]:
weights_aligned = weights_named.align_as(img_named)
weights_aligned.shape, weights_aligned.name

# Out [8]:
(torch.Size([3, 1, 1]), ('channels', 'rows', 'columns'))
```

对于那些接收dimension作为argument的function，比如sum，同样可以接受这个dimension的names作为argument：

```python
# In [9]:
gray_named = (img_named * weights_aligned).sum('channels')
batch_gray_named = (batch_named * weights_aligned).sum('channels')
gray_named.shape, gray_named.names, batch_gray_named.shape, batch_gray_named.names

# Out [9]:
(torch.Size([5, 5]), ('rows', 'columns'), torch.Size([2, 5, 5]), (None, 'rows', 'columns'))
```

我们可以注意到，在上面的例子里，weights_aligned的dimension是(3,1,1)，而batch_named的dimension是(2,3,5,5)，但是它们符合broadcasting的规则，且对应dimension的名字是一样的，从而可以运算。我们还可以注意到，结果的tensor是(2,5,5)，'channels'维度被sum函数消去，而第一个维度的名字和shape都继承了batch_named的，因为weight_aligned根本没有这个维度。

如果operation的两个tensor对应位置的names不一样，会直接报错：

```python
# In [10]:
gray_named = (img_named[..., :3] * weight_named).sum('channels')

# Out [10]:
RuntimeError: Error when
 attempting to broadcast dims ['channels', 'rows',
 'columns'] and dims ['channels']: dim 'columns' and dim 'channels'
 are at the same position from the right but do not match.

如果我们希望将已经命名的tensors还原到没有命名的时候，直接使用rename function将它们的名字改为None：

```python
# In [11]:
gray_plain = gray_named.rename(None)
gray_plain.shape, gray_plain.names

# Out [11]:
(torch.Size([5, 5]), (None, None))
```

named tensors仍然处于试用期，要谨慎使用。


### Tensor element types

到目前为止，我们已经讲了tensor是怎么运行的基础知识，但还没有讲什么类型的numeric types我们可以储存在tensor里。正如我们在3.2里所说的，用standard Python numeric types存放在tensor里并不是最优的选择，有以下几个原因：
* Numbers in Python are objects。因为一个floating-point number可能只需要32-bits来存放，Python会将它转换为一个有很多功能的复杂完整的Python object。这样的operation，叫做boxing，在存储小量数据时没有问题，但在数据量十分大的时候效率就变得十分低下了。
* Lists in Python are meant for sequential collections of objects。没有一个高效的算法能计算两个vectors之间的内积，或者计算一个vector内部所有element的和。同样的，Python的lists也没有一个高效的方法来优化如何在内存中存放list，因为Python的list内部存储的是list各个element的位置的指针，而各个element其实也都是Python object，并不一定非要是numbers。而且Python的list是一维的，尽管我们可以创建list of list，但这个效率就更低了。
* The Python interpreter is slow compared to optimized, compiled code。在大量的数据上运行数学计算，利用已经编译好的，底层的C代码要比Python快得多。

因为上述这些原因，data science libraries依赖Numpy或者是PyTorch tensors来提供numerical data structures以及它们上的operations的效率高的low-level的实现，并将它们包装在一个high-level的API里。为了使这个操作能够实现，一个tensor里的objects必须是同一种类型的numbers，而且PyTorch必须明确表示这个number的numeric type。

#### 3.5.1 Specifying the numeric type with dtype

tensor constructors（就是返回一个tensor的functions，比如torch.tensor，torch.ones，torch.zeros等）用dtype argument来指明要构建的tensor里的numbers的numerical data type（简写为dtype，就是从这来的）。dtype argument指明这个tensor内的numbers是intergers还是floating numbers，以及每个number占据多少个bytes，以及每个number是signed还是unsigned。PyTorch的tensor constructor的dtype argument故意设计的和标准的Numpy构建ndarray时的dtype叫同一个名字。

下面是dtype argument所有可能的取值：
* torch.float32 或者 torch.float：32-bit的floating-point
* torch.float64 或者 torch.double：64-bit的双精度的floating-point
* torch.float16 或者 torch.half：16-bit的半精度的floating-point
* torch.int8：有符号的8-bit整数
* torch.uint8：无符号的16-bit整数
* torch.int16 或者 torch.short：有符号的16-bit整数
* torch.int32 或者 torch.int：有符号的32-bit整数
* torch.int64 或者 torch.long：有符号的64-bit整数
* torch.bool：布尔值

默认的tensor的data type是32-bit的floating-point

>注意，floating-point都是带符号的。


#### 3.5.2 A dtype for every occasion

我们在后续的章节可以看到，在neural networks里发生的绝大多数运算用的都是32-bit floating-point的精度。更高的精度，64-bit floating-point不会给精度带来多少提高，但是会大大增加计算量。而更小的精度，16-bit floating-point在现在流行的CPU上不支持，在GPU上支持。用16-bit floating-point来计算可以减小计算量，但它对精度会带来多大的影响也是不可知的。

tensors同样可以用作其它tensors的index，那在这种情况下，PyTorch要求作为index的tensors的dtype是64-bit interger。构建一个tensor，利用integers作为初始值，那么PyTorch将会将dtype默认置为64-int integer，比如torch.tensor($$\left[2, 2\right]$$)。所以说，我们绝大多数时间都是处理dtype为int64或者float32的tensors。

最后，有tensors参与的谓词，比如points > 1.0，会产生bool tensor，来表示tensor的每个element是否满足条件。

以上就是PyTorch里tensor的dtype的一个概括。


#### 3.5.3 Managing a tensor's dtype attribute

为了给一个tensor分配正确的numeric type，我们需要在tensor constructor中给dtype特别赋值。

```python
# In [1]:
double_points = torch.ones(10, 2, dtype=torch.double)
short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)
```

我们直接通过访问一个tensor的dtype attribute来查看它：

```python
# In [2]:
short_points.dtype

# Out [2]:
torch.int16

我们还可以用相应的casting method将一个tensor constructor的输出投射到一个正确的numeric type：

```python
$ In [3]:
double_points = torch.zeros(10, 2).double()
short_points = torch.ones(10, 2).short()

# 或者使用更简单的to method：
double_points = torch.zeros(10, 2).to(torch.double)
short_points = torch.zeros(10, 2).to(dtype=torch.short)
```

在背后的机理里，to method会检查这个转换是否是有必要的（也就是说是不是本来就已经是该data type了），如果有必要再执行。那些dtype-named casting methods比如float是to method的简单表达方式，但是to method还可以有更多的argument，我们在3.9里会进一步讨论。

当运算的两个tensors有着不一样的data type时，较”小“的data type将会自动被转换为较”大“的那种。比如，如果我们希望计算只涉及32-bit的数，那所有的输入的tensor的dtype至多只能是32-bit的。

```python
# In [1]:
points_64 = torch.rand(5, dtype=torch.double)   # torch.rand用来生成0到1之间的随机数，5表示生成的维度是一维，shape=5
points_short = point_64.to(torch.short)         # 转换到torch.int16的情况下，每个值就变成0了
points_64 * points_short                        # integer和floating-point也可以相乘

# Out [1]:
tensor([0., 0., 0., 0., 0.], dtype=torch.float64)
```


### 3.6 The tensor API

现在我们已经知道了PyTorch的tensor是什么，以及它们背后的运行机制是什么样的。现在有必要来了解一下PyTorch提供的tensor operations。

首先，绝大部分一个tensor的operations或者两个tensors之间的operations都在torch module里，而且它们都可以被当作tensor的method来调用。

比如：transpose function可以从torch module中被调用：

```python
# In [1]:
a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)
a.shape, a_t.shape

# Out [1]:
(torch.Size([3, 2]), torch.Size([2, 3]))

# 而transpose function也可以被当成a的一个method被调用：

# In [2]:
a_t = a.transpose(0, 1)
a.shape, a_t.shape

# Out [2]:
(torch.Size([3, 2]), torch.Size([2, 3]))

# 这两种使用方法完全没有区别。
```

所有的tensor operations都可以在这个[online doc](https://pytorch.org/docs)上找到。它组织的很合理，将tensor的operations分为了以下几类：

* Creation ops：用来construct一个tensor的function，比如ones，from_numpy等
* Indexing，slicing，joining，mutating ops：用来改变一个tensor的shape，stride，或者内容的functions，比如transpose等。
* Math ops：通过计算来改变tensor的内容的funtions：
    * Pointwise ops：通过对一个tensor里每个element都作用该function来得到一个新的tensor，比如abs，cos等
    * Reduction ops：通过循环一个tensor里所有的element来得到一个总体的值，比如mean，std，norm等
    * Comparison ops：对tensors做numerical predicates，比如equal，max等
    * Spectral ops：在频域区间内转换和操作的函数，比如stft，hamming_window等
    * Other operations：一些tensor的特殊的function，比如cross等
    * BLAS和LAPACK ops：对scalar，vector-vector，matrix-vector，matrix-matrix的operations适用的BLAS（Basic Linear Algebra Subprogramming） functions。
* Random Sampling：从概率分布里随机取数的functions，比如randn，normal等
* Serialization：加载和存储tensor的functions，比如laod和save
* Parallelism：控制parallel CPU多线程运行的线程数量的functions，比如set_num_threads。


### 3.7 Tensors: Scenic views of storage

现在是时候看看PyTorhc里tensor的实现机理了。tensors里所储存的values被torch.Storage的instance分配给一块连续的内存空间。一个storage就是一个一维的Numerical data的array，也就是一个包含某个给定数据类型（比如float或者int64等）的连续内存空间。一个PyTorch tensor实例是这样一个torch.Storage实例的一个view，并且tensor加上了能够index等的功能。

多个tensors可以index同一个storage，即使形式上它们是不一样的，如figure4所示。实际上，在3.2里，point$$\left[0\right]$$会返回另一个tensor，而这个tensor index的storage和points tensor所指向的storage是同一个（只不过point$$\left[0\right]$$只是指向了一部分数据）。背后的内存实际上只被分配过一次，所以当创建这些新的tensors时，如果还是这个storage的或者其一部分的view，那么就会很快，因为并没有实际上在内存中创建了新的数据，这些都是被tensor.Storage实例来妥善解决的。

![Tensor storage]({{ '/assets/images/DLP-3-4.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 4. torch.Tensors are views of a torch.Storage instance.*

#### 3.7.1 Indexing into storage

我们用之前的2D points例子来看看如何在storage里index。一个给定的tensor的storage可以用.storage性质来获取：

```python
# In [1]:
points = torch.tensor([[4., 1.], [5., 3.], [2., 1.]])
points.storage()

# Out [1]:
4.0
1.0
5.0
3.0
2.0
1.0
[torch.FloatStorage of size 6]
```

尽管tensor本身有三行两列，是个2维的tensor，但是其后背的内存是一个size为6的array。所以说，tensor知道如何将一对index对应到storage里具体的value上。

我们可以手动index torch.Storage里的内容：

```python
# In [2]:
points_storage = points.storage()
points_storage[0]

# Out [2]:
4.0

# In [3]:
points.storage()[1]

# Out [3]:
1.0
```

我们并不能用一对index来index一个2维tensor背后的storage。一个storage的布局永远都是一维的，不管view它的tensor是什么维的。

现在，改变一个tensor背后storage的值就会改变这个tensor的值是显而易见的。

```python
# In [4]:
points_storage = points.storage()
points_storage[0] = 2.0
points

# Out [4]:
tensor([[2., 1.],
        [5. ,3.],
        [2., 1.]])
```

#### 3.7.2 Modifying stored values: In-place operations

之前介绍了很多tensor的operations既可以用torch.tensor的methods调用，也可以用以tensor为输入的function来调用，但有一小部分一小部分这样的operations仅仅可以作为torch.tensor的method被使用。它们通过在名字后面加个下划线来分辨，比如zero_，表明这个method是通过改变输入的in place的作用而不是创建饿了一个新的tensor作为输出。比如，zero_ method会将输入tensor的所有值都变成0。所有的名字后面没有下划线的method，都会保持输入的tensor不变，而输出一个新的tensor：

```python
# In [1]:
a = torch.ones(3, 2)
a.zero_()
a

# Out [1]:
tensor([[0., 0.]
        [0., 0.]
        [0., 0.]])
```


### Tensor metadata: Size, offset, and stride

为了能够index一个tensor的storage，除存储的数据以外tensor需要有一些格外的信息，叫做metadata，明确的定义size，offset和stride。figure5表现了这些metadata之间如何互相作用的。size（在Numpy中叫做shape）是一个tuple，表示这个tensor每个维度有多少个element，也就是每个维度的size。offset是存储这个tensor的storage第一个element所在的index（tensor内的内容作为一个一维array存储在storage里，offset指的是这个array第一个element的index）。stride指的是storage这个一维的array每一个element之间所跳过的element的个数，因为每个tensor的数据类型是知道的，所以根据element的个数也能算出具体要跳过多少个bytes，而stride仍然是个长度和tensor维度相同的tuple，它表示在tensor的每个维度下，保持别的维度值不变，从这个维度的一个element到这个维度的下一个element要跳过的element的个数。

![Tensor storage metadata]({{ '/assets/images/DLP-3-5.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 5. Relationship between a tensor's offset, size, and stride. Here the tensor is a view of a larger storage, like one that might have been allocated when creating a larger tensor.*

#### 3.8.1 Views of another tensor's storage

```python
# In [1]:
points = torch.tensor([[4. ,1.], [5., 3.], [2., 1.]])
second_point = points[1]
second_point.storage_offset()

# Out [1]:
2

# In [2]:
second_point.size()

# Out [2]:
torch.Size([2])
```

所获得的新的tensor，second_point在storage里的offset是2（因为跳过了points tensor里的2个elements），其size用tensor的size() method所获得，返回的是一个torch.Size类的实例，内部参数是长度为1的list，因为second_point是一维的，具体数值表示这个维度的大小。实际上同样的信息也被存储在tensor的shape property里：

```python
# In [3]:
second_point.shape

# Out [3]:
torch.Size([2])
```

stride是一个tuple，维度和tensor的维度一样，tuple内每个值表示这个值所在位置表示的维度的index增加1在storage里所需要跳过的element的个数。比如points的stride是(2,1):

```python
# In [4]:
points.stride()

# Out [4]:
(2, 1)
```

在一个2维的tensor里获取一个element，index是i,j，实际上是在storage里通过storage_offset + stride$$\left[0\right]$$ * i + stride$$\left[1\right]$$ * j位置的element获得。offset一般都是0，但如果这个tensor是一个更大的tensor的storage的一部分的view，那么offset就会是一个正整数。

torch.Tensor和tensor.Storage之间的这种关系让一些operations变得计算更容易，比如transposing一个tensor或者获取一个子tensor，因为这些operations都不必再在memory里重新分配内存了。取而代之的是，这些operations返回的是一个拥有新offset，size和stride的新的tensor，但并没有新的storage。

在上面的例子里我们已经在points这个tensor里获取了子tensor，second_point，发现second_point的offset增加了。我们再来看看second_point的size和stride是怎样的：

```python
# In [5]:
second_point = points[1]
second_point.size()

# Out [5]:
torch.Size([2])

# In [6]:
second_point.storage_offset

# Out [6]:
2

# In [7]:
second_point.stride()

# Out [7]:
(1, )
```

second_point是points tensor的子tensor，相对于points tensor少了一个维度。second_point和points仍然指向的是同一个storage。这意味着改变second_point这个tensor同样会改变points tensor的值：

```python
# In [8]:
second_point[0] = 10.0
points

# Out [8]:
tensor([[4., 1.],
        [10., 3.],
        [2., 1.]])
```

如果我们希望不要改变原tensor的值（实际上所有的view都会改变它们所指的storage的值），我们clone那个子tensor到一个新的tensor：

```python
# In [1]:
points = torch.tensor([[4., 1.], [5., 3.], [2., 1.]])
second_point = points[1].clone()
second_point[0] = 10.0
points

# Out [1]:
tensor([[4., 1.],
        [5., 3.],
        [2., 1.]])
```


#### 3.8.2 Transposing without copying

points tensor是一个2维的tensor，每一个row代表一个point的XY坐标，一共三个点，所以shape是(3, 2)。而我们现在希望将其transpose，从而shape变成(2, 3)，每一列代表一个point的XY坐标。我们趁机介绍一个t function，是transpose的简称，专门用来转秩2维tensor：

```python
# In [1]:
points = torch.tensor([[4., 1.], [5., 3.], [2., 1.]])
points

# Out [1]:
tensor([[4., 1.],
        [5., 3.],
        [2., 1.]])

# In [2]:
points_t = points.t()
points_t

# Out [2]:
tensor([[4., 5., 2.],
        [1., 3., 1.]])

# 我们可以来验证一下这两个tensor是不是指向同一个storage：

# In [3]:
id(points.storage()) == id(points_t.storage())

# Out [3]:
True

# 实际上这两个tensor，points和points_t仅仅在size和stride上有所不同：

# In [4]:
points.stride()
points_t.stride()

# Out [4]:
((2, 1), (1, 2))
```

所以说在points tensor里的第一个index增加1，比如说从points$$\left[0, 0\right]$$到points$$\left[1, 0\right]$$会在storage里跳过两个element，而第二个index增加1，比如说从points$$\left[0, 0\right]$$到points$$\left[0, 1\right]$$会在storage里跳过一个element。对于二维tensor来说，storage按照每行来存储值，而对于多维tensor来说，storage按照括号从内到外存储。

figure6表示了transpose一个2维tensor的过程。tranpose操作的正规定义为：并没有新的storage被创建，transpose是通过建立一个新的Tensor实例，这个实例的stride ordering与原tensor不同，而获得的。


#### 3.8.3 Transposing in higher dimensions

在PyTorch里，tranpose并不仅限于matrices。我们可以沿着任意两个dimension来transpose一个multidimensional array。

```python
# In [1]:
some_t = torch.ones(3, 4, 5)
transpose_t = some_t.transpose(0, 2)
some_t.shape

# Out [1]:
torch.Size([3, 4, 5])

# In [2]:
transpose_t.shape

# Out [2]:
torch.Size([5, 4, 3])

# In [3]:
some_t.stride()

# Out [3]:
(20, 5, 1)

# In [4]:
transpose_t.stride()

# Out [4]:
(1, 5, 20)
```

一个tensor，如果它所指向的storage所存储的数据是按照这个tensor内部括号从内到外存储的，比如之前的points，some_t等，那么称这个tensor是contiguous的，因为获取它的值只要按照括号从内到外的顺序，就不需要在storage里跳过任何element。经过transpose的tensor不是contiguous的。


#### 3.8.4 Contiguous tensors

某些PyTorch里的tensor operations仅仅作用于contiguous tensors，比如view，我们在下个chapter会遇到。tensor的contiguous method可以将一个non-contigous的tensor变成contigous的，而本来就是contiguous的tensor不会受到影响。

points是contiguous的，而points_t不是：

```python
# In [1]:
points.is_contiguous()

# Out [1]:
True

# In [2]:
points_t.is_contiguous()

# Out [2]:
False
```

我们可以通过contigous method来从non-contigous tensor获取一个新的contiguous tensor。新的tensor的内容是一样的，但是stride和storage都会变：

```python
# In [1]:
points = torch.tensor([[4., 1.], [5., 3.], [2., 1.]])
points_t = points.t()
points_t

# Out [2]:
tensor([[4., 5., 2.],
        [1., 3., 1.]])

# In [3]:
points_t.storage()

# Out [3]:
4.0
1.0
5.0
3.0
2.0
1.0
[torch.FloatStorage of size 6]

# In [4]:
points_t.stride()

# Out [4]:
(1, 2)

# In [5]:
points_t_cont = points_t.contigous()
points_t_cont

# In [6]:
tensor([[4., 5., 2.],
        [1., 3., 1.]])

# In [7]:
points_t_cont.stride()

# Out [7]:
(3, 1)

# In [8]:
points_t_cont.storage()

# Out [8]:
4.0
5.0
2.0
1.0
3.0
1.0
[torch.FloatStorage of size 6]
```

我们注意到，在上面的代码里，points_t.contiguous()重新创建了一个新的tensor，points_t_cont，使用contiguous methos使得points_t_cont变成了contiguous的，这个时候给points_t_cont tensor实际上分配了一个新的内存地址，而此时的storage就按照points_t_cont是个contiguous tensor的方式来存储，而且stride也变成了contiguous的。


### 3.9 Moving tensors to the GPU

到目前为止，当我们讨论storage，我们都讨论的是CPU上的memory。PyTorch tensor可以被存储在另一种processor上：graphics processing unit(GPU)。每个PyTorch tensor都可以被转移到某个GPU上来进行大规模快速并行计算。每个tensor的operation都会被PyTorch的GPU routines来执行。

>PyTorch Support for various GPUs
>在2019中旬之前的PyTorch版本只支持有CUDA的GPU。PyTorch也可以在AMD的ROCm上运行，但是需要用户自己编译。对Google的tensor processing units(TPUs)的支持仍在建设中，而别的类型的GPU，比如OpenCL,目前还不支持。

#### 3.9.1 Managing a tensor's device attribute

一个PyTorch的tensor，除了有dtype这样一个argument以外，还有一个device argument，指明这个tensor的数据放在computer的哪个设备上。我们通过指明这个argument来构造一个在GPU上的tensor：

```python
# In [1]:
points_gpu = torch.tensor([[4., 1.], [5., 3.], [2., 1.]], device = 'cuda')

# 我们也可以用to method将一个创建在CPU上的tensor转移到GPU上

# In [2]:
points_gpu = points.to(device = 'cuda')
```

通过上述操作，可以创建一个新的tensor，和原来的tensor有着一样numeracal data，但是存储在GPU的RAM里而不是默认系统的RAM里。现在tensor已经存在GPU里了，我们来看看它到底是如何加速计算的。在绝大多数情况下，CPU和GPU上的tensors使用相同的API，使得代码移植和编写变得简单。

如果设备上有多于一片GPU，我们可以直接指定用哪个GPU（从0开始）：

```python
# In [3]:
points_gpu = points.to(device='cuda:0')
```

现在这个tensor上的operations都会在GPU上进行了：

```python
# In [4]:

# 做一个操作，使得tensor里每个值乘以2

# 这个multiplication在CPU上计算
points = 2 * points

# 这个multiplication在GPU上计算
points_gpu = 2 * points.to(device='cuda')
```

上述计算结果points_gpu并不会被传回到CPU上，上面的代码过程如下：
* points tensor被复制到GPU上
* 一个新的GPU上的tensor被创建出来，用来存储multiplication的结果
* 这个GPU上的tensor的一个handle被返回

所以，如果我们给points_gpu加上一个常数，这个加法仍然在GPU上计算，而且并不会有数据流向CPU（除非我们要print它们或者index access它们）。

```python
# In [5]:
# 在GPU上进行
points_gpu = points_gpu + 4
```

如果我们想要将tensor转移回CPU，我们需要给to method提供一个cpu argument：

```python
# In [6]:
points_cpu = points_gpu.to(device='cpu')
```

我们可以直接使用cpu和cuda methods而不用to method来实现同样的操作：

```python
# In [7]:
points_gpu = points.cuda()   # 默认在GPU 0上
points_gpu = points.cuda(0)
points_cpu = points_gpu.cpu()
```

但是通过to method我们可以同时改变数据的部署设备和数据类型，因为to method可以传入device和dtype两个arguments。


### 3.10 Numpy interoperability

PyTorch tensors和Numpy arrays可以非常高效的互相转换。通过将PyTorch tensor转换为Numpy array，我们可以借用Numpy的广泛的功能。

```python
# In [1]:
points = torch.ones(3, 4)
points_np = points.numpy()
points_np

# Out [1]
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]], dtype=float32)
```

上述代码从points这个tensor里获得了points_np这个Numpy multidimensional array，其拥有着正确的size，shape，以及numerical type。有趣的是，返回的这个Numpy array和tensor storage共享同一个buffer。这表明只要tensor是在CPU RAM里的，这个tensor对应的Numpy array的Numpy method可以很高效的被执行。这同样表明改变Numpy array的值同样会改变原tensor的值。如果这个tensor是GPU上的，那PyTorch会复制一份这个tensor的内容给CPU上的Numpy array。

相反的，我们也可以从一个Numpy multidimensional array来获得一个PyTorch tensor：

```python
# In [2]:
points = torch.from_numpy(points_np)
```

这样生成的PyTorch tensor和原Numpy array关于buffer的关系和上面所述的一样。

>注意到，PyTorch默认的数据类型是32-bit floating-point，而Numpy默认的数据类型是64-bit floating-point。但正如我们之前所说的，我们希望使用32-bit floating-point，所以在转换后，我们要确认dtype=torch.float。


### 3.11 Generalized tensors are tensors, too

### 3.12 Serializing tensors

当数据很重要的时候，我们希望能将tensor存储下来，并且以后加载使用。毕竟我们不想每次都重新训练我们的模型。PyTorch使用pickle来serialize一个tensor，同时伴有精心设计的storage的serialization code。这是如何将points tensor存储在outpoints.t file的方法：

```python
# In [1]:
torch.save(points, '../data/p1ch3/ourpoint.t')

# 我们也可以用一个file descriptor来替代filename：

# In [2]:
with open('../data/p1ch3/outpoints.t', 'wb') as f:
    torch.save(points, f)
```

加载points回来也是很简单的：

```python
# In [3]:
points = torch.load('../data/p1ch3/outpoints.t')

# 或者下面的代码也可以：

# In [4]:
with open('../data/p1ch3/outpoints.t') as f:
    points = torch.load(f)
```

虽然说上述的方法存储和加载tensor都很方便，但是我们只能通过PyTorch来存储和加载它，这个文件其它的软件无法识别，not interoperative。所以更有用的是掌握如何将tensors存储为其他格式文件的方法。


#### 3.12.1 Serializing to HDF5 with h5py

HDF5格式是一种很常见的文件存储格式。HDF5是一种可移动的，广泛支持的用来表示serialized multidimensional arrays的文件格式，其组织成一种nested key-value字典的形式。Python通过h5py库来支持HDF5，这个库接收以及返回Numpy的arrays。

```python
# In [1]
conda install h5py  # 安装h5py库
```

从而我们可以通过先将points tensor转换为Numpy array（如之前所提，很方便），之后再将其用create-dataset function来存储：

```python
# In [1]:
import h5py

f = h5py.File('../data/p1ch3/ourpoints.hdf5', 'w')
dset = f.create_dataset('coords', data=points.numpy())
# coords是文件的key

f.close()
```

HDF5文件的key还可以是nested种类的。HDF5一个有意思的事情是我们可以在打开数据之后，数据仍然还在硬盘里的时候，index我们感兴趣的部分，然后只加载这部分的内容。

```python
# In [2]:
f = h5py.File('../data/p1ch3/outpoints.hdf5', 'r')
dset = f['coords']
last_points = dset[-2:]  # 我们只需要后两个点的内容被加载加来
```

上述代码里，当file打开的时候，数据没有被加载，仍然存在硬盘里。直到最后一行代码被执行之前，数据都还在硬盘里，没有加载到内存里。在最后一行执行之后，h5py获取了dset这个数据在硬盘上最后两个column的内容，并返回了那个区域的数据，以Numpy array的形式。

因为加载的数据是Numpy array的形式，所以我们可以使用torch.from_numpy funcion直接将返回的array转换为PyTorch的tensor。

```python
# In [3]:
last_points = torch.from_numpy(dset[-2:])
f.close()
```

当我们结束加载数据后，我们用f.close()关闭这个文件。关闭这个HDF5文件会使得dset失效，从而访问它会报错。我们之后只需要用last_points这个tensor就可以。


### 3.13 Conclusion

现在我们已经具备了将任意输入转换为floating-point number的先前知识。我们将会介绍tensor的其他知识，比如创建tensor的view，用tensor来index tensor，以及broadcasting，随着内容的深入而逐步介绍。

在chapter4里，我们将会学会如何在PyTorch里表示现实世界里的数据。我们从最简单的tabular data开始，之后逐渐到更复杂的数据。在这个过程中，我们会更深入的了解tensors。


### 3.14 Summary
* neural networks将floating-point representations转换为其它的floating-point representations。起始和结束的representations是人类所能够理解的，但是中间的那些representations比较难以理解。
* 这些floating-point representations存储在tensor里。
* tensor是multidimensional array，它们是PyTorch里最基本的数据结构。
* PyTorch对于tensor creation，manipulation和mathematical operations有着详尽的标准库。
* tensors可以被serialized到硬盘里，并加载回来。
* 所有的PyTorch里的tensor operations可以在CPU和GPU上被操作，而不需要改变代码。
* PyTorch使用带有一个下划线的函数来表示这个函数是作用在tensor上的in-place函数（也就是说它改变原tensor的值，不返回新的tensor），比如Tensor.sqrt_等。

```python
# In [1]:
a = torch.tensor(list(range(9)))
a = a.to(dtype=torch.float)       # 必须将a的数据类型转换，否则cos_函数将无法将原始值替换掉，因为原始的数据类型是int。
a.cos_()                          # 带有一个下划线结尾的函数会直接作用在原tensor上，并替换它的值，in-place操作
a

# Out [1]:
tensor([ 1.0000,  0.5403, -0.4161, -0.9900, -0.6536,  0.2837,  0.9602,  0.7539, -0.1455])
```


## Chapter 4 Real-world data representation using tensors

>本章包括的内容：
>* 将现实生活中的data表示为PyTorch tensor的形式
>* 处理一系列不同的数据类型
>* 从文件里加载数据
>* 将数据转换为tensor
>* 将tensor的形状进行改变从而使他们适用于neural networks的输入

在上一章，我们学习到tensors是PyTorch里数据的基本构建模块。neural networks使用tensor作为输入并输出tensor。实际上，一个neural network里进行的，以及optimization时进行的所有的operations都是tensors之间的operations，以及一个neural network里所有的参数（比如weights，bias）都是tensors。对如何处理tensors之间的operations以及如何高效的index它们有良好的理解是学习使用PyTorch的基础。现在我们已经了解了tensor的基础知识，对于它们掌握的熟练度将会随着项目的使用而变得更好。

现在下面这个问题我们已经可以解决了：我们如何处理现实生活中的数据，比如一段视频，一行文字等，将它们转换为tensor从而能被deep learning models作为输入来使用？这将是这章所要解决的问题。我们将会介绍一系列数据类型，并一一介绍如何将他们转换为tensors。之后我们将会学习如何从硬盘上加载数据，学习硬盘上数据的常见格式，并学习如何将它们转换为neural networks可以使用的tensors。通常情况下，原始数据对于所要解决的问题并非完全格式正确，所以我们还需要使用上一章的知识来对转换后的tensors做一些处理。

这一章里的每一节都会描述一种数据类型，后续节可能会用到先前节介绍过的数据类型。

在本书剩下的部分会大量使用image以及volumetric数据，所以我们会更加详细的介绍这方面的内容。我们同样会介绍tabular data，time series和text。我们将会从image开始介绍。之后我们将会介绍volumetric数据。再之后，我们将会介绍tabular data，time series data和text。

这一章的每一节内容将会在开始训练model时结束介绍，也就是只介绍如何将各种类型的数据转换为neural network models所能使用的tensors。关于如何训练neural networks的内容将会在下一章介绍。


### 4.1 Working with images

convolutional neural networks的提出革命了整个computer vision界，基于image的系统从此有了个强有力的工具。那些需要很复杂的预处理算法的image任务现在可以直接被end-to-end网络替代，无需任何预处理，只需要提供input-ouput对即可。我们需要做的，是从常见的image格式里加载image数据，并将其转换为neural networks能够作为输入的tensors。

一张image就是表示为一个通过矩形grid约束的scalars的集合，而这个grid具有height和width两种属性。每个grid点称为这个image的一个pixel。在每个pixel可以只有一个scalar，那么这个称为grayscale image；也可以有多个scalars，比如多种颜色或者多种深度特征等。

在每个pixel的scalars通常用8-bit整数编码表示。更高精确度的比如12-bit或者16-bit有时也会出现。

#### 4.1.1 Adding color channels

如果每个pixel都包含多个scalars来表示颜色信息，有多种表示方法，而最常见的就是RGB表示法。RGB表示法的image数据，每个pixel有三个scalars，分别表示red，green和blue的颜色强度。单个颜色通道可以被认为和grayscale image类似。Figure1显示了一张RGB照片的样子。

![RGB image]({{ '/assets/images/DLP-4-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 1. A rainbow, broken into red, green, and blue channels.*

上图彩虹的红色部分在red channel里对应位置的强度最大，而blue channel在彩虹图片的蓝色部分对应位置的强度最大。而白色部分，三个通道的强度都大。


#### 4.1.2 Loading an image file

images有各种不同的数据格式，而Python都为它们提供了加载方式。我们先介绍如何用imgeio module来加载JPG格式image。

```python
# In [1]:
import imageio

img_arr = imageio.imread('../data/p1ch4/image-dog/bobby.jpg')
img_arr.shape

# Out [1]:
(720, 1280, 3)
```

>在本章我们使用imageio来加载image，因为它可以用同一个API加载各种格式的image。而TorchVision也是加载image和video的工具。

在上述代码执行完，img_arr是一个NumPy array-like object，并有三个维度：两个空间维度height和widt，以及一个颜色维度RGBchannels。任何能够输出一个NumPy array的库都可以作为PyTorch加载image的方式，因为PyTorch tensor和NumPy array联系紧密。我们需要注意的是加载进来的image维度的分布，PyTorch models一般需要输入image的维度是$$ C \times H \times width$$，channels，height，width。（若大于三维，可能还有batch）


#### 4.1.3 Changing the layout

我们可以用tensor的permute method来改变tensor维度的分布。给一个输入tensor，其维度是$$H \times W \times C$$，我们将其转换为$$C \times H \times W$$：

```python
# In [2]:
img = torch.from_numpy(img_arr)
out = img.permute(2, 0, 1)
```

tensor的permute method并不会在内存里再分配新的数据，out和img共用内存里同一片数据，只是stride和size有所不同。这样可以加快计算速度，但改变out或者img则会改变另一个的值，需要注意。

到目前为止，我们描述了一张image该如何加载。用同样的方法，我们可以加载多张images，并将它们作为一个batch，给neural networks作为输入使用，而这个batch的维度则是$$N \times C \times H \times W$$。

我们可以直接用stack function来将tensors摞起来。而另一种方法是我们预先分配一块地方来存放，之后再将images加载进来：

```python
# In [1]:
batch_size = 3
batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)
```

上述代码表明我们的batch有三张长宽为256，256的RGB images。注意我们将tensor的数据类型设置为unit8，这和常用相机的图片数据类型是一样的，8-bit整数。我们现在可以从一个目录里加载一系列images存入这个tensor里：

```python
# In [2]:
import os
data_dir = '../data/p1ch4/image-cats/'
filenames = [name for name in os.listdir(data_dir) if os.path.splittext(name)[-1] == 'png']
for i, filename in enumerate(filenames):
    img_arr = imageio.imread(os.path.join(data_dir, filename)
    img_t = torch.from_numpy(img_arr)
    img_t = img_t.permute(2, 0, 1)
    img_t = img_t[:3]               # 确保只取image的前三个通道，因为可能还有更多的通道表示其他信息
    batch[i] = img_t                # batch[i]表示取batch的第一个维度
```

#### 4.1.4 Normalizing the data

我们之前提到，neural networks一般接受floating-point tensors作为输入。neural networks在数据数据处于0到1或者-1到1之间时有着最好的训练效果（这是由PyTorch的内部机理导致的）。

所以我们需要将tensor转换到floating-point的格式，并normalize这些数据。将tensor数据转换为floating-point类型是容易的，但normalization有些不同。最简单的，将数据除以256（因为8-bit unsigned integer格式数据最大就是256）：

```python
# In [3]
batch = batch.float()
batch /= 255.0
```

另一种可能性就是计算出数据的mean和standard deviation，再将其转换为mean为0，standard deviation为1的数据，这样的操作是沿着每个channel做的：

```python
# In [4]:
n_channels = batch.shape[1]
for c in range(n_channels):
    mean = torch.mean(batch[:, c])
    std = torch.std(batch[:, c])
    batch[:, c] = (batch[:, c]- mean) / std
```

>我们上述的normalization是针对一个batch来做的。实际上很多任务都是针对整个training set算出mean和std，比如2.1.4里的ImageNet数据集就是这样做的。

我们还可以对输入做其他的操作，比如geometric transformation，包括rotation，scaling，cropping等。这些操作可以使得training set更加丰富，也可以使得training set满足某些条件，比如特定的输入size等。我们将会在chapter12.6里提到相应的内容。


### 4.2 3D images: Volumetric data

我们已经学习了如何加载和表达2D images，比如我们用相机所拍到的照片。在某些情况下，比如某些医疗影像，CT（computed tomography）scans，我们处理的是一系列的有序排列的照片，对于CT来说，每一张相当于人体的一个切片。在CT图片里，每个pixel的density代表着体内不同成分。而每个pixel位置的density是通过某种方式计算出来的。

每张CT照片的每个pixel位置只有一个scalar，也就是只有一个通道，代表着这个点处的强度，类似于grayscale image。这表明，在一般情况下，原始数据格式里，channel的维度就被省略了；所以原始数据的维度是三维的。通过将很多张2D images堆叠起来，我们得到了一个volumetric类型的数据，表达所检测物体的3D信息。和figure1里第三个维度表示的是颜色通道不同的是，figure2里的第三个维度表示的是物理上的空间信息。

![CT]({{ '/assets/images/DLP-4-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 2. Slices of a CT scan, from the top of the head to the jawline.*

储存volumetric data的tensor和储存image的tensor没有本质上的区别，只不过多了一个维度，depth，在channel维度的后面，所以整个tensor就是一个5D的tensor，$$N \times C \times D \times H \times W$$。

#### 4.2.1 Loading a specialized format

我们利用imageio module里的volread function来加载CT数据：

```python
# In [1]:
import imageio

dir_path = "../daat/p1ch4/volumetric-dicom/2-LUNG 3.0   B70f-04083"
vol_arr = imageio.volread(dir_path, 'DICOM')                        # dir_path是存放数据的目录，而'DICOM'是一种数据格式
vol_arr.shape                                                       # volread函数将这个目录下的'DICOM'格式的数据整合在一起，构成一个NumPy 3D array   

# Out [1]:
Reading DICOM (examining files): 1/99 files (1.0%99/99 files (100.0%)
Found 1 correct series.
Reading DICOM (loading data): 31/99 (31.392/99 (92.999/99 (100.0%)
(99, 512, 512)
```

所读取的数据的layout和PyTorch所期待的不太一样（PyTorch期待的是上面所说的5D数据），因为没有channel这个维度。所以我们需要给数据增加一个维度，利用unsqueeze function：

```python
# In [3]:
vol = torch.from_numpy(vol_arr).float()
vol = torch.unsqueeze(vol, 0)

vol.shape

# Out [3]:
torch.Size([1, 99, 512, 512])
```

我们现在就可以将多个上述的数据按照batch维度堆叠在一起，构成一个前面所述的5D数据。


### 4.3 Representing tabular data

我们在机器学习任务里遇到的最简单的数据可能就是tabular data了，存储在一个spreadsheet，CSV file或者一个database里。tabular data的每一行代表了一个sample的数据，而每一列代表每个sample一部分信息的汇总。

我们先假设，每一行数据就是一个sample，行与行之间是没有顺序的，就是独立的数据。

每一列代表着某种特征，可以是numerical number，也可以是string等。因此，tabular data一般都不是同质的（也就是每列数据类型不一定是一样的）。

PyTorch tensors却是同质的，因为tensor内部只含有floating-point numbers。这种将任意输入转换为floating-point number的操作是有意为之，因为只有这样，neural networks才能将这些输入通过各种matrix multiplications和nonlinear functions来输出floating-point numbers。

### 4.3.1 Using a real-world dataset

我们作为深度学习实践者的第一个任务就是将现实生活中异质的data编码为neural networks能够使用的floating-point tensors。我们使用的是Wine Quality dataset。

这个数据集包含了对wine的描述，tabular文件由12列组成，第一行是每列名称的头文件，每列用逗号隔开。前11列是各种化学性质，最后一列是0-10的一个评分。一个有意思的机器学习任务就是基于前11列的数据，给出最后一列总体评分的数据。


#### 4.3.2 Loading a wine data tensor

在我们设计机器学习模型之前，我们需要将数据载入。我们利用Python载入数据，并将数据转换为PyTorch tensor。这个文件是个CSV file，Python提供了很多种方法，比如Python自带的csv module，NumPy，Pandas等。而Pandas是时间和内存效率最高的方式。但本书为了避免介绍更多的内容，就使用NumPy来加载CSV file。

```python
# In [1]:
import csv
import numpy as np

wine_path = "../data/p1ch4/tabular-wine/winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=",", skiprows=1)

wineq_numpy

# Out [1]:
array([[ 7. , 0.27, 0.36, ..., 0.45, 8.8 , 6. ],
       [ 6.3 , 0.3 , 0.34, ..., 0.49, 9.5 , 6. ],
       [ 8.1 , 0.28, 0.4 , ..., 0.44, 10.1 , 6. ],
       ...,
       [ 6.5 , 0.24, 0.19, ..., 0.46, 9.4 , 6. ],
       [ 5.5 , 0.29, 0.3 , ..., 0.38, 12.8 , 7. ],
       [ 6. , 0.21, 0.38, ..., 0.32, 11.8 , 6. ]], dtype=float32)

# 我们来检查一下是不是所有的数据都被加载了：

# In [2]:
col_list = next(csv.reader(open(wine_path), delimiter=","))

wineq_numpy.shape, col_list

# Out [2]:
((4898, 12),
['fixed acidity',
'volatile acidity',
'citric acid',
'residual sugar',
'chlorides',
'free sulfur dioxide',
'total sulfur dioxide',
'density',
'pH',
'sulphates',
'alcohol',
'quality'])

# 我们将上述NumPy array转换为一个PyTorch tensor

# In [3]:
wineq = torch.from_numpy(wineq_numpy)

wineq.shape, wineq.dtype

# Out [3]:

(torch.Size([4898, 12]), torch.float32)
```

#### 4.3 Representing scores

我们可以认为评分是一个连续变量，认为其是一个实数，从而采用regression的方式，或者认为评分是一个标签，是一系列离散的数，从而采用classification的方式。不管哪种方式，我们都要将评分列，也就是最后一列，从原数据中提取出来单独对待，作为ground truth来使用。

```python
# In [4]:
data = wineq[:, :-1]
data, data.shape

# Out [4]:
(tensor([[ 7.00, 0.27, ..., 0.45, 8.80],
         [ 6.30, 0.30, ..., 0.49, 9.50],
         ...,
         [ 5.50, 0.29, ..., 0.38, 12.80],
         [ 6.00, 0.21, ..., 0.32, 11.80]]), torch.Size([4898, 11]))

# In [5]:
target = wineq[:, -1]
target, target.shape

# Out [5]:
(tensor([6., 6., ..., 7., 6.]), torch.Size([4898]))
```

如果你想将target tensor转换为包含label的tensor，有两种选择，取决于我们要如何使用它。第一种是直接将其转换为integer，利用integer来表示label：

```python
# In [6]:
target = wineq[:, -1].long()
target

# Out [6]:
tensor([6, 6, ..., 7, 6])
```

如果target tensor包含string作为label，那么给每种string指定一个integer，和上述结果是一样的。

#### 4.3.4 One-hot encoding

另一种处理方式是构建评分的one-hot编码：将1-10每个得分都编码为一个长度为10的vector，这个vector除了一个位置的值为1，其余都是0。比如得分为1表示为(1,0,0,0,0,0,0,0,0,0)。

这两种方式表达label有一个显著的区别。直接使用wine的score的整数表示，会给这些label加上了order信息，但在这个情况下是可以的，因为这个任务本身就是得出一个评分，label=2和label=1之间的距离要比label=2和label=10之间的距离要小，而且label=1和label=2之间的差距和label=2与label=3之间的差距是一样的。使用one-hot编码的话，每个label之间的距离都是一样的，label之间互相独立。

我们可以用scatter_ method得到标签的one-hot编码，

```python
# In [7]:
target_onehot = torch.zeros(target.shape[0], 10)

target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
```

我们来看scatter_ method做了什么。首先，scatter_使用一个下划线结尾，正如前面所说的，这种method将不会返回一个新的tensor，而只是改变原有的tensor。而scatter_ method的arguments如下：
> 后面两个arguments沿着哪一个维度来操作
> 一个column tensor用来表明scatter的元素的index，这个index的维度要和被scatter的tensor的维度相同
> 一个包含着需要被scatter的元素的tensor或者一个单个的scalar（在这个例子里就是一个单个的scalar，1）


#### 4.3.5 When to categorize

我们已经看到了如何处理continuous和categorical数据的方法。figure3 显示了该如何选择这两种表达方式的算法图：

![chart]({{ '/assets/images/DLP-4-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 3. How to treat columns with continuous, ordinal, and categorical data.*

回到我们之前所得到的data tensor。我们先计算data tensor每一列的mean和std：

```python
# In [8]
data_mean = torch.mean(data, dim=0)
data_mean

# Out [8]:
tensor([6.85e+00, 2.78e-01, 3.34e-01, 6.39e+00, 4.58e-02, 3.53e+01,
        1.38e+02, 9.94e-01, 3.19e+00, 4.90e-01, 1.05e+01])

# In [9]:
data_var = torch.var(data, dim=0)
data_var

# Out [9]:
tensor([7.12e-01, 1.02e-02, 1.46e-02, 2.57e+01, 4.77e-04, 2.89e+02,
        1.81e+03, 8.95e-06, 2.28e-02, 1.30e-02, 1.51e+00])
```

在上面代码里，dim=0表示计算mean和std是沿着dimension 0来做的。现在我们可以normalize data（可以使得训练更好，在5.4.4里详细介绍）：

```python
# In [10]:
data_normalized = (data - data_mean) / torch.sqrt(data_var)
data_normalized

# Out [10]:
tensor([[ 1.72e-01, -8.18e-02, ..., -3.49e-01, -1.39e+00],
        [-6.57e-01, 2.16e-01, ..., 1.35e-03, -8.24e-01],
        ...,
        [-1.61e+00, 1.17e-01, ..., -9.63e-01, 1.86e+00],
        [-1.01e+00, -6.77e-01, ..., -1.49e+00, 1.04e+00]])
```

#### 4.3.6 Finding thresholds

接下来我们来看能够有简单的方法能一眼看出一些规律。首先，我们找到target tensor里哪些行的score小于3：

```python
# In [11]:
bad_indexes = target <=3   # PyTorch也提供了comparison functions，torch.le(target, 3)可以在这里被使用，而直接使用operators显得更加直接
bad_indexes.shape, bad_indexes.dtype, bad_indexes.sum()

# Out [11]:
(torch.Size([4898]), torch.bool, tensor(20))
```

注意到，bad_indexes里只有20个elements是True。PyTorch里有一个高级特征，叫advanced indexing，我们可以用一个data type为torch.bool的tensor来index另一个tensor，此处我们可以用bad_indexes这个tensor来index data这个tensor：

```python
# In [12]:
bad_data = data[bad_indexes]
bad_data.shape

# Out [12]:
torch.Size([20, 11])
```

注意到bad_data只有20行，和bad_indexes tensor里为True的行数相同。bad_data仍然保持了11列。

```python
# In [13]:
bad_data = data[target <=3]
mid_data = data[(target > 3) & (target < 7)]
good_data = data[target >=7]

bad_mean = torch.mean(bad_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)

for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print('{:2} {20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))

# Out [13]:
0 fixed acidity 7.60 6.89 6.73
1 volatile acidity 0.33 0.28 0.27
2 citric acid 0.34 0.34 0.33
3 residual sugar 6.39 6.71 5.26
4 chlorides 0.05 0.05 0.04
5 free sulfur dioxide 53.33 35.42 34.55
6 total sulfur dioxide 170.60 141.83 125.25
7 density 0.99 0.99 0.99
8 pH 3.19 3.18 3.22
9 sulphates 0.47 0.49 0.50
10 alcohol 10.34 10.26 11.42
```

我们可以看到sulfur dioxide比较高的，wine的score就比较低。所以我们可以提前使用这个数值来选取数据：

```python
# In [14]:
total_sulfur_threshold = 141.83
total_sulfur_data = data[:, 6]
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)

predicted_indexes.shape, predicted_indexes.dtype, predicted_indexes.sum()

# Out [14]:
(torch.Size([4898]), torch.bool, tensor(2727))

# In [15]:
actual_indexes = target > 5
actual_indexes.shape, actual_indexes.dtype, actual_indexes.sum()

# Out [15]:
(torch.Size([4898]), torch.bool, tensor(3258))

# In [16]:
n_matches = torch.sum(actual_indexes & predicted_indexes).item()
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indexes).item()

n_matches, n_matches / n_predicted, n_matches / n_actual

# Out [16]:
(2018, 0.74000733406674, 0.6193984039287906)
```


### 4.4 Working with time series

在之前的内容里，我们了解了如何处理表格数据，这些表格数据里的每一行都是相互独立的，每行的顺序是没有影响的。或者这样说，并没有决定行排列顺序的列信息出现。

我们介绍一个新的数据集：Washington, D.C., bike-sharing system，存的是租借自行车所需要的费用，而影响的因素有weather，temperature等，而且每日的信息是不同的。我们将每日的表格数据汇总，就可以将2D的表格数据变成3D的带有时间的表格数据，如figure4所示。

![3Dtable]({{ '/assets/images/DLP-4-4.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 4. Transforming a 1D, multichannel dataset into a 2D, multichannel dataset by separating the date and hour of each sample into separate axes.*

#### 4.4.1 Adding a time dimension

```python
# In [1]:
bikes_numpy = np.loadtxt("../data/p1ch4/bike-sharing-dataset/hour-fixed.csv",
                         dtype=np.float32,
                         dlimiter=",",
                         skiprow=1,
                         converters={1: lambda x: float(x[8:10])})
## 上面converters的作用是将第1列里的日期string转换为代表每个月几号的numbers

bikes = torch.from_numpy(bikes_numpy)
bikes

# Out [1]:
tensor([[1.0000e+00, 1.0000e+00, ..., 1.3000e+01, 1.6000e+01],
        [2.0000e+00, 1.0000e+00, ..., 3.2000e+01, 4.0000e+01],
        ...,
        [1.7378e+04, 3.1000e+01, ..., 4.8000e+01, 6.1000e+01],
        [1.7379e+04, 3.1000e+01, ..., 3.7000e+01, 4.9000e+01]])
```

对于每个小时，这个数据集报告以下的变量：
* index of record：从小到大的整数
* day of month：日期
* season：1是spring，2是summer，3是fall，4是winter
* year：0是2011，1是2012
* month：1到12的数字
* hour：0到23的数字
* holiday status：是否是holiday
* day of week：是否是weekday
* working day status：是否是workingday
* weather situation：1是chear，2是mist，3是light rain/snow，4是heavy rain/snow
* temperature：温度
* perceived temperature：体感温度
* humidity：湿度
* wind speed：风速
* number of causual users：临时用户数量
* number of registered users：注册用户数量
* count of rental bikes：自行车总数量

上面这个数据集，行与行之间有时间序列关系。我们当然可以忽略这种时间关系，把它们当成之前的那种表格数据来操作，只通过当前的数据来预测租车要花的钱，而不管之前和之后的数据。但是每行数据之间的时间序列关系，给我们提供了预测数据的新的信息。



#### 4.4.2 Shaping the data by time period

我们将Washington D.C. bike-sharing system dataset数据重新组合，原数据是2D的，每行就是每天24个小时的数据按顺序排列，之后再按天排列，每列就是各个指标的值。而我们现在将其变为3D的，第1维度表示日期，第2维度表示24小时，第3维度表示各个指标的值。

```python
# In [2]:
bikes.shape, bikes.stride()

# Out [2]:

(torch.Size([17520, 17]), (17, 1))

# In [3]:
daily_bikes = bikes.view(-1, 24, bikes.shape(1))
daili_bikes.shape, daily_bikes.stride()

# Out [3]:
(torch.Size([730, 24, 17]), (408, 17, 1))
```

我们上面用到了view function，它的输出daily_bikes和bikes公用一个storage，但读取数据的方式不同，从而维度，stride都不同。正如之前所学的，通过调用view method，会返回一个新的tensor，其维度和stride都变了，但原tensor对应的storage没有改变。这样的操作十分省时省空间。view method需要我们给返回的tensor提供size，而上面的-1是一个placeholder，表明这个维度的值是通过原始tensor的维度和所提供的新tensor的其他维度的值算出来的，最多只有一个-1。

之前的章节还提到storage是连续的，线性分布的。所以bikes tensor的storage将每行连续的存储在内存里。这点也可以从bikes.stride()看出来。

注意到我们现在有了$$N \times\ L \times C$$的tensor，daily_bikes，而作为neural networks的输入，它要求我们的输入形式为$$N \times C \times L$$，所以我们还得做一下转秩：

```python
# In [4]:
daily_bikes = daily_bikes.transpose(1, 2)
daily_bikes.shape, daily_bikes.stride()

# Out [4]:
(torch.Size([730, 17, 24], (408, 1, 17))
```

#### 4.4.3 Ready for training

channel里的weather situation指标是ordinal的，我们可以将其当作categorical的或者continuous的，而当成categorical的则需要将其变为one-hot编码。

```python
# In [5]:
first_day = bikes[:24].long()
weather_onehot = torch.zeros(first_day.shape[0], 4)
first_day[:, 9]

# Out [5]:
tensor([1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2])

# In [6]:
weather_onehot.scatter_(dim=1, index=first_day[:, 9].unsqueeze(1).long()-1, value=1.0)
## index-1是因为weather_situation是从1到4，而python index是从0开始

# Out [6]:
tensor([[1., 0., 0., 0.],
        [1., 0., 0., 0.],
        ...,
        [0., 1., 0., 0.],
        [0., 1., 0., 0.]])

## 用cat function来将bikes tensor与weather_onehot tensor沿着dimension=1，也就是列，拼接起来
# In [7]:
torch.cat((bikes[:24], weather_onehot), 1)[:1]

## 这个显示的是拼接后的数据的第一行
# Out [7]:
tensor([[ 1.0000, 1.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000,
          6.0000, 0.0000, 1.0000, 0.2400, 0.2879, 0.8100, 0.0000,
          3.0000, 13.0000, 16.0000, 1.0000, 0.0000, 0.0000, 0.0000]])
```

cat function要求提供沿着哪个维度来拼接tensors，并且要求这个维度以外的两个tensors的其它维度相同。

我们将已经处理好的daily_bikes tensor里的weather_situation指标也按上述的方式进行更改，需要注意daily_bikes的维度已经变成了$$N \times C \times L$$。

```python
# In [8]:
daily_weather_onehot = torch.zeros(daily_bikes.shape[0], 4, daily_bikes.shape[2])
daily_weather_onehot.shape

# Out [8]:
torch.Size([730, 4, 24])

# In [9]:
daily_weather_onehot.scatter_(dim=1, index=daily_bikes[:,9,:].long().unsqueeze(1)-1, value=1.0)

# In [10]:
daily_bikes = torch.cat((daily_bikes, daily_weather_onehot), dim=1)
```

而如果我们直接将weather_situation这个指标当成Continuous的：

```python
# In [11]:
daily_bikes [:, 9, :] = (daily_bikes[:, 9, :] - 1.0) / 3.0
```

上述即可将这个指标的值转移到0-1之间的连续值。

我们前面已经提到，作为neural networks的输入，最好值都在0到1之间或者-1到1之间（之后将会说明原因），所以其它的指标仍然需要我们进行操作。

```python
# In [12]:
## 这种方法将其转换到0到1之间
temp = daily_bikes[:, 10, :]
temp_min = torch.min(temp)
temp_max = torch.max(temp)
daily_bikes[:, 10, :] = (daily_bikes[:, 10, :] - temp_min) / (temp_max - temp_min))

# In [13]:
## 这种方法将其转换到-1到1之间
temp = daily_bikes[:, 10, :]
daily_bikes[:, 10, :] = (daily[:, 10, :] - torch.mean(temp)) / torch.std(temp)
```


### 4.5 Representing text

deep learning也给natural language processing带来了很大的影响，其中最著名的模型就是recurrent neural networks，其将新输入和旧输出的组合作为输入，输出新的输出，以此循环。RNN在text categorization，text generation，automated translation等领域都获得了很大的成功。最近一种叫transformers的新模型的出现，使得利用旧输出的方法变得更加灵活。传统的NLP方法都需要很多精巧设计的stage，在每个stage通过某些rules来筛选信息。而现在的deep learning可以做到end-to-end，输入直接就是大段的文字，而这些rules都是从数据中直接得出的。

我们这一节的目的是将text转换为一个neural network能够处理的东西：tensor。如果我们将文本转换为了tensor，就可以直接使用PyTorch来做NLP的任务。

#### 4.5.1 Converting text to numbers

直观来看，networks处理text有两种方式：一个一个字符处理，或者一个一个单词来处理。而对于这两种方式，我们将text信息转换为tensor的方式都是一样的，而这个方式就是one-hot encoding。

我们先来看character-level的例子。首先，我们得到了一个text：

```python
# In [1]:
with open('../data/p1ch4/jane-austen/1342-0.txt', encoding='utf8') as f:
    text = f.read()
```

#### 4.5.2 One-hot encoding characters

在开始前我们再来简单介绍一下encoding。每个character都通过一个code来表示。最简单的就是ASCII，a被编码为1100001或者数字97，b被编码为11000010或数字98。ASCII encoding使用8位编码方式。

我们将要对character进行one-hot encode。每个character都被编码为长度为这个set里所有不同character数量的vector，而且这个vector只有一个位置是1，代表了这个character在这个set里所在的位置。

```python
# In [2]:
lines = text.split('\n')
line = lines[200]
line

# Out [2]:
'“Impossible, Mr. Bennet, impossible, when I am not acquainted with him'

# In [3]:
letter_t = torch.zeros(len(len), 128)
letter_t.shape

# Out [3]:
torch.Size([70, 128])

# In [4]:
for i, letter in enumerate(line.lower().strip()):
    letter_index = ord(letter) if ord(letter) < 128 else 0
    letter_t[i][letter_index] = 1
    
## Python的ord function是chr function或者unichr function的alias function
## 文本是ASCII编码的就是chr，是Unicode编码的就是unichr，而其返回ASCII码或者
## 返回Unicode码。ord的输入为长度为1的字符串。

```

#### 4.5.3 One-hot encoding whole words

character-level的encoding很简单，其结果可以作为neural network的输入。而word-level的encoding可以用相似的方法完成，但我们需要建立一个字典。因为这样的字典可能会很长，所以对于one-hot encoding，对每个word会生成很长的tensor，并不方便使用。下一节将会介绍embedding方法，使得这样的encode变得可以操作。而本节则只关注one-hot encoding是什么样的。

```python
# In [5]:
def clean_word(input_str):
    punctuation = '.,;:"!?_-'
    word_list = input_str.lower().replace('\n', ' ').split()
    word_list = [word.strip(punctuation) for word in word_list]
    return word_list

words_in_line = clean_word(line)
line, words_in_line

# Out [5]:
('“Impossible, Mr. Bennet, impossible, when I am not acquainted with him',
 ['impossible',
 'mr',
 'bennet',
 'impossible',
 'when',
 'i',
 'am',
 'not',
 'acquainted',
 'with',
 'him'])

# In [6]:
word_list = sorted(set(clean_word(text)))
word2index_dict = {word: i for i, word in enumerate(word_list)}

len(word2index_dict), word2index_dict['impossible']

# Out [6]:
(7261, 3394)

# In [7]:
word_t = torch.zeros((len(words_in_line), len(word2index_dict))
for i, word in enumerate(words_in_line):
    word_index = word2index_dict[word]
    word_t[i][word_index] = 1
    print('{:2} {:4} {}'.format{i, word_index, word})

print(word_t.shape)

# Out [8]:
0 3394 impossible
1 4305 mr
2 813 bennet
3 3394 impossible
4 7078 when
5 3315 i
6 415 am
7 4436 not
8 239 acquainted
9 7148 with
10 3215 him
torch.Size([11, 7261]
```

character-level和word-level的encoding选择是一个trade-off的过程。在很多语言里，characters的数量比words的数量要少得多，所以character的字典要比word的字典要小很多，而且word有时候还会遇到字典里没有这个word的情况。但另一方面，word比character代表了很多有意义的内容，所以用word来表示文本本身就有更多的信息量。

![3ways]({{ '/assets/images/DLP-4-5.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 5. Three ways to encode a word.*


#### 4.5.4 Text embeddings

one-hot encoding是一种简单有效的方法，但是当字典变得很大时，就变得不再实用了。

一种有效的替代方式就是使用floating numbers来对每个word进行编码，也就是将每个word映射到一个固定长度的值为floating-point numbers的vector上，这个过程就叫做embedding。

原则上，我们可以对字典里的每个word随机生成一个固定长度N的vector。这样是可以操作的，而且我们每个word的encoding就变成了长度为N的vector。但是这样的embedding忽略了words之间的关系。如果model利用了这样的embedding会很难挖掘句子或者文本里蕴含的信息。一个理想的embedding是使得相似词义的word能够拥有相似的embeddings。

如果我们打算手动定义一个满足上述条件的embedding，我们可以通过选择将基本的名词和形容词先映射到轴上。我们可以生成一个2D的空间，x轴对应名词，比如fruit(0.0-0.33)，flower(0.33-0.66)，dog(0.66-1.0)，形容词对应Y轴，比如red(0.0-0.2)，orange(0.2-0.4)，yellow(0.4-0.6)，white(0.6-0.8)，brown(0.8-1.0)。我们之后就可以将每个word按照它所属的名词和形容词对应到具体的位置上。

比如apple可以被对应到fruit和red所决定的空间内。类似的tangerine, lemon, lychee, kiwi等都可以。之后对于flower，rose, poppy, daffodil, lily等也都可以对应到具体的空间里。dogs也是同样用这种方式来对应。从而我们得到了一个figure6所示的embedding。但是手动做embedding对于大的文本是不现实的。

![embeddings]({{ '/assets/images/DLP-4-6.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 6. Our mannual word embeddings.*


上述这种方式可以被自动进行，而区别在于每个轴对应的意义不是具体的意义，而且embedding的vector的长度远大于上例里的2，但原理仍然是意义相近的word的embedding位于相近的位置。embedding有意思的一个地方是，不仅意义相似的word会有相近的embedding，它们还具有连续的空间性质。比如apple-red-sweet+yellow+sour所得到的embeddding会和lemon的embedding距离很近。比较著名的embedding学习的网络有BERT，GPT-3等。


#### 4.5.5 Text embedding as a blueprint

embeddings是将大规模文本内的word转换为可以操作的floating-number point vector的有力方法。但本书并不会再提到NLP或者text的内容。介绍这方面的知识是因为它为我们处理categorical类型的数据提供了新的思路。embeddings在one-hot encode变得不可操作的时候就显得很有作用。

在非文本的问题里，我们通常没有直接构造上述embedding的能力（因为对于自动方法没有上下文供我们使用，对于手动方法我们无法对它们的性质加以分类），我们通常都是将embedding初始化为随机值，然后将确定它们的值作为我们学习任务的一部分。这个方法很常见也很广泛，尤其是对于categorical数据的表示学习。

embeddings也为co-occurence问题提供了思路，比如推荐系统，我们希望通过已有的用户信息来判断它的喜好。processing text同时也是处理time-series类型的数据，所以其方法同样也适用于其它类型的time-series数据的处理。


### 4.6 Conclusion

我们本章介绍了很多种数据类型，学习如何加载它们并将它们转换为neural networks能够利用的tensors。现在我们已经熟悉了tensors以及了解了如何将现实中的数据转换为tensors，我们下一章就可以介绍neural networks的learning机制了。


### 4.7 Summary
* neural networks所能处理的数据都是表示为multidimensional numerical tensors的形式，一般为32-floating point numbers
* 一般情况下，PyTorch希望tensors具有特定的维度，以及每个维度有着特定的意义，比如CNN和RNN的输入是具有不同意义的。reshape data可以用PyTorch的API轻易实现
* 多亏了PyTorch和Python以及其生态圈十分方便的相互联系，加载不同类型的数据并将它们转换为PyTorch的tensors变得比较容易
* images一般都有多个channel，最常见的RGB image有三个channel，red, green和blue
* 一般图像的每个pixel的numerical data是8-bit或者12，16-bits的，这样的数据用32-floating point numbers来表示不会丢失精度
* volumetric data和2D image差不多，只不过除了channel，width，height以外还有个depth维度
* 将spreadsheet数据转换为tensors十分直接。注意categorical和cardinal类型的数据需要进行额外的操作，比如one-hot encoding
* text或者categorical的数据可以转换为one-hot encoding。而embedding是更高效和具有更多信息量的手段




## Chapter 5 The mechanics of learning

>本章包括的内容：
>* 理解algorithms如何从data中学习
>* 重新定义learning为使用differentiation和gradient descent的parameter estimation
>* 从头到尾介绍一个简单的learning algorithm
>* PyTorch如何通过autograd实现learning

随着machine learning过去十年的快速发展，可以从经验中学习的machine变成了技术类文章甚至广播新闻的主要对象。而一个machine到底是如何学习的？这个学习的过程的机理是什么样的？或者说，学习过程的算法是什么？从一个观察者的角度，一个learning algorithm的输入是带有所想要得到的输出的数据（supervised learning）。一旦这个machine经过了正确的learning，那么它就可以对于和训练时的数据**足够近似**的新数据给出正确的输出。对于deep learning，输入的数据和输出甚至可以很不相同，比如说输入是一张图片，而输出是描述这张图片内容的text。


### 5.1 A timeless lesson in modeling

建立能够描述input/output关系的模型起码可以追溯到几个世纪以前。Kepler，一个德国数学天文学家，在十七世纪前期得出了行星运动三大定律，他是基于所观测到的众多行星的运动规律吧数据所总结出来的。当时并没有Newton定律，Kepler只是基于数据，建立了一个能够吻合数据的最简单的模型。figure1总结了这个过程。

![Kepler]({{ '/assets/images/DLP-5-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 1. Johannes Kepler considers multiple candidate models that might fit the data at hand, settling on an ellipse.*

Kepler第一定律说，任何行星的轨道都是一个以太阳位于一个焦点的椭圆。他并不知道为什么轨道是一个椭圆，但是他可以基于大量的数据总结出轨道的形状以及几何性质。Kepler第二定律说，行星与太阳的连线在相同时间段内扫过相同的面积。

所以，让我们总结一下Kepler六年内的成果：
* 从观察中获得了大量行星运动相关的数据
* 尝试通过画图来寻找其中的规律
* 选择了能够符合数据的最简单的模型来描述行星运动规律（椭圆）
* 将数据分开，从而他可以通过一部分数据得出猜想，而利用没有使用过的数据来验证他的猜想的正确性
* 从一个不确定几何参数的椭圆开始，逐渐改变椭圆的大小形状直到符合数据
* 在独立的数据上验证所得到的椭圆轨道模型
* 再次回看改进模型

Kepler的工作完美解释了数据科学的内容。科学基本都是用这七个步骤来进行的，之后，我们再花时间来解释模型背后的原理（除非你是理论物理学家，那情况就有所不同）。

上面的过程就是我们如何从data中learning的过程。实际上，fit data和设计一个algorithm从data中learning是没有区别的。这个过程都是有一个有一些未知参数的function，而这些参数都是从data中估计得来，而这个function就是model。

需要注意，learning from data假设潜在需要学习的function并不是为了解决某个特定的问题而内部有复杂的规则设计，而是具有拟合能力的一簇functions，正如Kepler通过行星数据来总结出开普勒定律，而并非Newton使用万有引力推导出行星运动模型。

这一章就是关于如何自动的进行一般情况下的function-fitting。毕竟，这就是我们用deep learning所要做的事，deep neural networks就是我们的具有拟合能力的一簇functions，PyTorch使得这个拟合过程尽可能的简单透明。

为了更够更深刻的理解learning的过程，我们用一个比deep neural networks更简单的模型来介绍，并在chapter6里正式介绍deep neural networks的learning过程。


### 5.2 Learning is just parameter estimation

在这一节里，我们将会学习如何获取data，选择model，并估计model的parameters，从而我们可以对于新的data有良好的预测结果。

![mental]({{ '/assets/images/DLP-5-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 2. Our mental model of the learning process.*

figure2展示了high-level的描述learning过程的一个overview。给你input data和相应的output（ground truth），以及model里参数的初始值，input data将喂给model（forward pass），然后对于output和ground truth之间差别的一个衡量（error）被计算出来。为了能够优化model的参数，error关于model内的参数的gradient被利用导数的链式法则计算出来（backward pass）。之后model内的参数将会沿着使得error减小的方向被改进。上述这个过程将会持续性的重复，直到在验证集上达到满意的指标。

我们现在考虑一个问题，有noisy的dataset，构建一个model，并在其上建立一个learning algorithm。在一开始，我们将会手动设计每个环节，而本章结束，我们将会学会如何用PyTorch自动进行这个过程。


#### 5.2.1 A hot problem

假设我们有一个不知道刻度标准的温度计，我们希望这个温度计能根据环境温度给出相应的数值。现在我们来设计这个问题以及它的解答。

#### 5.2.2 Gathering some data

```python
# In [1]
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)
```

t_c是摄氏度的温度，而t_u是我们未知标准的温度计给出的数值。这两个数据里都有一定程度的noisy。


#### 5.2.3 Visualizing the data

我们将上述数据放在坐标轴上，横轴是温度计数值，纵轴是已知摄氏温度。从figure2里我们可以看到，它们之间近似是一种线性关系。

![linear]({{ '/assets/images/DLP-5-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 3. Our unknown data just might follow a linear model.*

#### 5.2.4 Choosing a linear model as a first try

既然我们没有更多的先验知识，那我们就为这个数据做最简单的假设，即它们可能是线性关系：t_c = w * t_u + b。这是不是一个有效的假设呢？只能说是有可能。我们需要利用新的数据来验证所学习到的模型是否足够正确。参数w和b是weight和bias的简称。

现在我们需要基于数据来估计w和b的值，也就是我们这个模型的参数。我们需要找到这样的w和b，使得输入的t_u经过模型所得到的值和对应的t_c很接近。我们将用PyTorch来实现这个过程。而且实际上，PyTorch来训练neural network也只是比这个模型参数多一些，数据多一些，但本质是一样的。

为了能够学习参数，我们还需要定义output和ground truth之间误差的衡量，称为loss function。loss function在output和ground truth差异大的时候值比较大，差异小的时候值比较小。所以我们的算法就在于找到一个使得loss function达到最小值的w和b。


### 5.3 Less loss is what we want

一个loss functino（或cost function）是一个输出值被learning process用来最小化的function。loss的计算就是观察对于一些training samples的输入所对应的ground truth和模型所得到的output之间的差别（并非所有的training samples因为计算量会很大）。在我们的例子里，loss就是通过模型对于输入t_u所计算的t_p和真实的对应的t_c之间的差异计算来的。

最简单的几个loss function可以是|t_p - t_c|以及(t_p - t_c)^2。原则上，loss function的设计加入了先验的信息，因为loss function可以强调我们会着重考虑哪种loss信息。这两个loss function都是在其相同的时候取得最小值，在差异大的时候单调增加，所以是满足要求的。这两个loss function还是convex的，相对于w和b这两个参数也是convex的。loss function是convex的时候的问题都有比较好的解，因为local minimum就是global minimum，而且可以通过高效算法计算得到结果。但是我们这里不用这些方法，因为这些方法对于neural networks是行不通的。

从figure4我们可以看到，square of differences会比absolute difference表现得更好，因为在t_p = t_c时，square of difference在这点的导数为0，而absolute difference在这点的导数却没有定义。在我们最关心的点却没有导数定义，这是很致命的。而且square of difference这个loss所训练得到的数据，往往不会有大的偏差，但每个都有些小的偏差，而absolute difference这个loss训练所得的数据，会在某几个值上有很大的偏差，而在其他地方更加精确。

![loss function]({{ '/assets/images/DLP-5-4.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 4. Absolute difference versus difference squared.*


#### 5.3.1 From problem back to PyTorch

我们现在已经有了model，也有了loss function和数据，是时候利用PyTorch来进行训练了。

```python
# In [2]
def model(t_u, w, b):
    return w * t_u + b
```

在这个模型里，w和b将是PyTorch scalars（也就是zero-dimensional tensors），而w和t_u之间的乘法以及和b之间的加法将会由broadcasting来完成。

```python
# In [3]:
def loss_fn(t_c, t_p):
    squared_diffs = (t_p - t_c)^2
    return squared_diffs.mean()      # 这个是mean squared loss

# In [4]:
w = torch.ones(())
b = torch.zeros(())

t_p = model(t_u, w, b)
t_p

# Out [4]:
tensor([35.7000, 55.9000, 58.2000, 81.9000, 56.3000, 48.9000, 33.9000,
        21.8000, 48.4000, 60.4000, 68.4000])

# In [5]:
loss = loss_fn(t_p. t_c)
loss

# Out [5]:
tensor(1763.8846)
```

我们已经计算了给模型输入得到的输出，并计算了loss的值。现在我们要解决的是，下一步如何通过最小化loss来得到所要的参数的值呢？


### 5.4 Down along the gradient

我们将会利用gradient descent algorithm来优化loss function得到所想要的参数的值。在这一个section里，我们将会介绍gradient descent algorithm是怎么运作的。

假设我们有个思想实验，如figure5所示。我们可以调控w和b的值，并且可以看到loss随着这两个参数改变而改变的值。我们逐渐的调整w或者b的值，会发现loss一开始下降的快，后来逐渐变慢，直到在一个局部最小值的地方停下来，我们再朝着一个方向调整w或b的话，loss就会再次变大。我们发现，当loss改变较慢的时候，我们也需要调整w和b慢一点，避免错过了最小值点。最终，我们会得到所想要的最小值的位置。

#### 5.4.1 Decreasing loss

gradient descent正如我们上面所描述的那个思想实验一样。重点在于计算每个参数随着loss改变而变化的速率，并将每个参数沿着减小loss的方向而改变。正如上面实验所示，我们通过给w一个很小的改变量，来看loss改变了多少：

```python
# In [6]:
delta = 0.1

loss_rate_of_change_w = (loss_fn(model(t_u, w + delta, b), t_c) - loss_fn(model(t_u, w - delta, b), t_c)) / (2.0 * delta)
```

上述function计算了改变一个单位的w会造成loss改变多少，如果是正的，就要减小w的值，如果是负的，就增加w的值。但是增加或者减小多少呢？增加或者减小的量与loss_rate_of_change_w成比例是比较合理的，特别是当模型有多个参数的时候，每个参数要改变的量会不一样。一直以很小的改变量来修改参数的值也是一种方法，但很慢。我们一般都会给loss_rate_of_change乘一个比较小的值，叫做learning rate。

```python
# In [7]:
learning _rate = 1e-2

w = w - learning_rate * loss_rate_of_change_w

loss_rate_of_change_b = (loss_fn(model(t_u, w, b + delta), t_c) - loss_fn(model(t_u, w, b - delta), t_c)) / (2.0 * delta)

b = b - learning_rate * loss_rate_of_change_b
```

上述过程即是更新参数所经历的过程。当我们重复上述过程很多次，并且使用足够小的learning_rate，我们就能使得参数位于使得loss达到最小值的点（在这个例子里是最小值，一般情况下是极小值）


#### 5.4.2 Getting analytical

计算通过重复性的计算model和loss来得到rate_of_change去更新参数的值，在参数数量很多，模型很大的情况下是不适用的。而且我们也不知道我们需要探索的参数空间应该有多大。我们上面取了delta=0.1，但是这个值很依赖loss function以及w和b的值。如果在某个w和b点，loss function变得很快，那么这个delta就不适用了。

但如果我们使用十分小的delta呢？这实际上也就是我们计算loss function在每个参数点处的导数值。在具有多个参数的模型里，我们计算loss function对于每个参数的导数，并把他们放在一个vector里，叫做gradient。

![gradient]({{ '/assets/images/DLP-5-5.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 5. Differences in the estimated directions for descent when evaluating them at discrete locations versus analytically.*

**COMPUTING THE DERIVATIVES**
为了计算loss对于一个参数的derivative，我们可以计算利用链式法则：d loss_fn / d w = d loss_fn / d t_p * d t_p / d w，从而

```python
# In [8]:
# 这个函数用来计算loss对于t_p的导数，t_p.size(0)是因为是mean squared loss

def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)
    return dsq_diffs
```

**APPLYING THE DERIVATIVES TO THE MODEL**

```python
# In [9]:
# 计算derivatives

def dmodel_dw(t_u, w, b):
    return t_u

def dmodel_db(t_u, w, b):
    return 1.0
```

**DEFINING THE GRADIENT FUNCTION**
```python
# In [10]:
def grad_fn(t_u, t_p, t_c, w, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])
```

figure6展示了上述过程的数学原理。我们最后仍然用sum function来对于每个参数获得对于所有输入的数据一个单一的gradient值。

![math]({{ '/assets/images/DLP-5-6.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 6. The derivative of the loss function with respect to the weights.*


#### 5.4.3 Iterating to fit the model

我们现在已经有了优化参数的所有组件。从参数的一个起始值开始，我们可以循环有限次来更新参数的值，或者一直循环更新直到满足某个条件为止。

**THE TRAINING LOOP**

我们将通过利用所有的training samples来更新参数值的一个training loop称为一个epoch。一个完整的training loop如下所示：

```python
# In [11]:

def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        w, b = params
        t_p = model(t_u, w, b)               # forward
        loss = loss(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, b)  # backward
        
        params = params - learning_rate * grad
        
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
        
    return params
```

接下来，我们运行一下上述代码：

```python
# In [12]:
training_loop(n_epochs=100, learning_rate=1e-2, params=torch.tensor([1.0, 0.0]), t_u=t_u, t_c=t_c)

# Out [12]:
Epoch 1, Loss 1763.884644
    Params: tensor([-44.1730, -0.8260])
    Grad: tensor([4517.2969, 82.6000])
Epoch 2, Loss 5802485.500000
    Params: tensor([2568.4014, 45.1637])
    Grad: tensor([-261257.4219, -4598.9712])
Epoch 3, Loss 19408035840.000000
    Params: tensor([-148527.7344, -2616.3933])
    Grad: tensor([15109614.0000, 266155.7188])
...
Epoch 10, Loss 90901154706620645225508955521810432.000000
    Params: tensor([3.2144e+17, 5.6621e+15])
    Grad: tensor([-3.2700e+19, -5.7600e+17])
Epoch 11, Loss inf
    Params: tensor([-1.8590e+19, -3.2746e+17])
    Grad: tensor([1.8912e+21, 3.3313e+19])
    
tensor([-1.8590e+19, -3.2746e+17])
```

**OVERTRAINING**

上述代码中，我们的training process直接爆炸了，loss变成了inf。这是因为每次更新的值太大了，导致参数发生了振荡，这个optimization的过程是unstable的：它diverge了而不是收敛到minimum。figure7展示了这个现象。

![diverge]({{ '/assets/images/DLP-5-7.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 7. Top: Diverging optimization on a convex function (parabola-like) due to large steps. Bottom: Converging optimization with small steps.*

我们将learning_rate调小，来看看training process会怎样：

```python
# In [13]:
training_loop(n_epochs=100, learning_rate=1e-4, params=torch.tensor([1.0, 0.0]), t_u=t_u, t_c=t_c)

# Out [13]:
Epoch 1, Loss 1763.884644
    Params: tensor([ 0.5483, -0.0083])
    Grad: tensor([4517.2969, 82.6000])
Epoch 2, Loss 323.090546
    Params: tensor([ 0.3623, -0.0118])
    Grad: tensor([1859.5493, 35.7843])
Epoch 3, Loss 78.929634
    Params: tensor([ 0.2858, -0.0135])
    Grad: tensor([765.4667, 16.5122])
...
Epoch 10, Loss 29.105242
    Params: tensor([ 0.2324, -0.0166])
    Grad: tensor([1.4803, 3.0544])
Epoch 11, Loss 29.104168
    Params: tensor([ 0.2323, -0.0169])
    Grad: tensor([0.5781, 3.0384])
...
Epoch 99, Loss 29.023582
    Params: tensor([ 0.2327, -0.0435])
    Grad: tensor([-0.0533, 3.0226])
Epoch 100, Loss 29.022669
    Params: tensor([ 0.2327, -0.0438])
    Grad: tensor([-0.0532, 3.0226])
    
tensor([ 0.2327, -0.0438])
```

现在optimization的过程稳定了下来，但是因为learning_rate太小，所以可能还没有完成优化，训练过程就停止了。我们可以使得learning_rate和grad成比例，我们在section5.5.2里会看到这样的作法。

上述优化过程还有个问题需要解决，gradient自己。我们来看看epoch1里的grad。


#### 5.4.4 Normalizing inputs

我们可以看到，在第一个epoch里，w的gradient要比b的gradient大50倍。这说明w和b位于不同尺度的参数空间内。如果是这样的话，那对于某个参数适用的learning_rate对于另一个参数就不适用了。所以如果我们不加以更改，甚至无法找到合适的learning_rate。我们可以给每个参数一个learning_rate，但如果参数很多，这种做法是不切实际的。

有一个更为简单的解决问题的办法，改变inputs，从而使得gradients的差别变得不大。在我们的例子里，我们可以这样做：

```python
# In [14]:
t_un = 0.1 * t_u
```

我们现在再来运行一下training process：

```python
# In [15]:
training_loop(n_epochs = 100, learning_rate = 1e-2, params = torch.tensor([1.0, 0.0]), t_u = t_un, t_c = t_c)

# Out [15]:
Epoch 1, Loss 80.364342
    Params: tensor([1.7761, 0.1064])
    Grad: tensor([-77.6140, -10.6400])
Epoch 2, Loss 37.574917
    Params: tensor([2.0848, 0.1303])
    Grad: tensor([-30.8623, -2.3864])
Epoch 3, Loss 30.871077
    Params: tensor([2.2094, 0.1217])
    Grad: tensor([-12.4631, 0.8587])
...
Epoch 10, Loss 29.030487
    Params: tensor([ 2.3232, -0.0710])
    Grad: tensor([-0.5355, 2.9295])
Epoch 11, Loss 28.941875
    Params: tensor([ 2.3284, -0.1003])
    Grad: tensor([-0.5240, 2.9264])
...
Epoch 99, Loss 22.214186
    Params: tensor([ 2.7508, -2.4910])
    Grad: tensor([-0.4453, 2.5208])
Epoch 100, Loss 22.148710
    Params: tensor([ 2.7553, -2.5162])
    Grad: tensor([-0.4446, 2.5165])
    
tensor([ 2.7553, -2.5162])
```

上述训练过程的learning_rate设置成了1e-2，而现在也不会震荡了。我们来看gradients，w的和b的位于差不多的量级上，所以同一个learning_rate就可以。如果我们对输入做normalization可能会效果更好，但在这个例子里，乘以0.1就足够使用了。

我们接下来运行足够多的次数，来使得gradient足够小，从而达到loss的最小值：

```python
# In [15]:
training_loop(n_epochs = 5000, learning_rate = 1e-2, params = torch.tensor([1.0, 0.0]), t_u = t_un, t_c = t_c, print_params=False)

# Out [15]:
Epoch 1, Loss 80.364342
Epoch 2, Loss 37.574917
Epoch 3, Loss 30.871077
...
Epoch 10, Loss 29.030487
Epoch 11, Loss 28.941875
...
Epoch 99, Loss 22.214186
Epoch 100, Loss 22.148710
...
Epoch 4000, Loss 2.927680
Epoch 5000, Loss 2.927648

tensor([ 5.3671, -17.3012])
```

我们看到，在训练的后期，loss改变的已经很小了，但仍然不是0，这可能是因为训练的次数还不够多，或者模型本身就不能很好的拟合数据。但我们的参数其实已经很接近正确值了，说明我们的训练过程起了作用。

#### 5.4.5 Visualizing (again)

```python
# In [16]:
%matplotlib inline
from matplotlib import pyplot as plt

t_p = model(t_un, *params)

fig = plt.figure(dpi=600)
plt.xlabel("Temperature (Fahrenheit)")
plt.ylabel("Temperature (Celsius)")
plt.plot(t_u.numpy(), t_p.detech().numpy())
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
```

我们这里用了Python的一个技巧：argument unpacking。\*params 表示以单独argument的形式来给出params里的元素。在Python里，这个通常用在list或者tuples里，但PyTorch借用了这个用法，我们可以用这个来argument unpack PyTorch的tensors，tensors会沿着第一个dimension裂开。所以上述代码里的model(t_un, \*params)实际上就是model(t_un, params\[0\], params\[1\])。

上述代码会画出figure8。我们的linear model对于数据能够很好的拟合。

![plot]({{ '/assets/images/DLP-5-8.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 8. The plot of our linear-fit model (solid line) versus our input data (circles).*


### 5.5 PyTorch's autograd: Backpropagating all things

在前一节里，我们计算了复合函数，也就是model和loss，对于参数w和b的derivatives，因为模型简单且所有的复合函数的组成函数都可以被计算，我们可以计算出每个derivative的公式。

即使我们使用的是具有上百万个参数的很复杂的model，只要我们的model是可以被微分的，计算loss相对于参数的gradient就相当于将loss相对于每个参数的derivatives的公式写出来，并计算一次它们的值。然而，计算一个很复杂的模型的loss相对于每个参数的derivative的公式是很复杂的，很难计算的。

#### 5.5.1 Computing the gradient automatically

正是因为上面提到的计算很复杂，PyTorch在这个时候就有了作用，它有一个组成部分为autograd，用来解决这个困难。Chapter3展示了什么是tensor并且我们可以在tensors上调用哪些function。我们在那一章里留了一个很有意思的内容没有说，那就是，PyTorch的tensors可以记住它们是从哪里来的，也就是说它们记住了产生它的parent tensors和operations，而且它们可以轻易的用链式法则给出它们对于inputs的derivatives是多少。这个PyTorch的tensors的性质表明我们不需要手动来计算这个model的各个derivatives，给一个forward的expression，不管有多复杂，不管内部嵌套了多少计算，PyTorch都会自动给出这个expression对于这个expression里各个参数的derivatives的值。

**APPLYING AUTOGRAD**

我们现在回到之前温度计的例子，利用autograd来重写这个code

```python
# In [1]:
def model(t_u, w, b):
    return w * t_u + b

# In [2]:
def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()

# In [3]:
params = torch.tensor([1.0, 1.0], requires_grad = True)
```

**USING THE GRAD ATTRIBUTE**

注意到上面代码里，paras里有个参数requires_grad被设置为True，这就告诉了PyTorch去跟踪所有建立在这个tensor上的operations所形成的树。换句话说，任何将params作为某个先前tensor的tensor都能获得如何从params计算得到该tensor的信息。如果这些中间计算的functions都是可微的（绝大多数PyTorch operations都是），那么该tensor对于params的derivative就会计算出来，占据params的grad attribute。

通常来说，所有的PyTorch tensors都有叫grad的attribute。通常情况下，这个attribute是None。

```python
# In [4]:
params.grad is None

# Out [4]:
True
```

而我们如何使得这个params的grad attribute被占据呢？我们首先需要从一个requires_grad被设置为True的tensor开始，然后计算model，从而计算依赖于该tensor的后续tensor，在这个例子里，输入tensor是params，后续tensor是loss，然后我们要在loss tensor上调用其的backward method：

```python
# In [5]:
loss = loss_fn(model(t_u, *params), t_c)
loss.backward()

params.grad

# Out [5]:
tensor([4517.2969, 82,6000])
```

从而，params的grad attribute就包含了loss对于params每个值的derivative的值。

当我们计算loss，并且参数w和b的requires_grad设置为True时，除了计算具体的值，PyTorch还生成了一个autograd的graph，operations（黑色圆圈）作为nodes，如figure9上面的图所示。当我们调用loss.forward()时，PyTorch将上述graph反过来来计算gradient，如figure9下面的图所示。

![autograd]({{ '/assets/images/DLP-5-9.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 9. The forward graph and backward graph of the model as computed with autograd.*

**ACCUMULATING GRAD FUNCTIONS**

我们可以有任意数量的tensors，其requires_grad被设置为True，而且我们也可以构造任意复杂的复合函数。PyTorch会计算loss对于每个requires_grad被设置为True且与loss有计算关系的tensor，将对应的derivative的值，累积到该tensor的grad attribute里。

特别注意，我们这里用的是累积这个词，而不是存储，也就是说，每次调用backward() method都会使得新的derivates加在grad attribute上。这一点很多人容易犯错。如果要利用每次loss对于该tensor的gradients来更新值的话，那么在每次计算之后，要使得该tensor的grad attribute清零，再进行下一次的backward()。

所以说，在我们的代码里，在每个循环里还需要加入zero the gradient的操作。我们可以利用in place的zero_ method来轻易实现：

```python
# In [6]:
if params.grad is not None:
    params.grad.zero_()
   
# In [7]:
def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:
            params.grad.zero_()                            # 这一步骤在loss.backward()之前就行
        
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        loss.backward()
        
        with torch.no_grad():                              # 这一步似乎显得多余，但之后的章节可以看到是有作用的
            params -= learning_rate * params.grad
            
        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
        
    return params
```

注意到上述计算params的代码并不是非常的直接，有两点值得注意的地方。首先，我们将更新params的代码封闭在一个Python with的上下文环境里，使用torch.no_grad()作为内容。这表明，在这个with环境里，Python的autograd机制不会起作用（实际上PyTorch会跟踪它们，将该操作当成in place的操作）：也就是说它不会在forward graph里添加operation对应的edge。实际上，当我们执行这一块的代码时，PyTorch所记录的forward graph关于这一块的代码会在backward()时直接被吸收，只留下一个params node。其次，我们使用in place的方式更新params。这表明我们只有一个params，而我们在执行的过程中更新了它的值。当使用autograd时，我们实际上要避免in place的更新，因为PyTorch的autograd engine可能需要更新前的该tensor的值来进行计算。而在这里，我们的操作并不需要涉及autograd，所以说这样的in place更新是可以的，还可以避免引入新的变量。在5.5.2里，当我们在optimizer里注册parameters的时候还会遇到这个问题。

我们现在来看上述代码如何运作：

```python
# In [8]:
training_loop(
      n_epochs = 5000,
      learning_rate = 1e-2,
      params = torch.tensor([1.0, 1.0], requires_grad = True,     # requires_grad = True一定不能忘记
      t_u = t_un,
      t_c = t_c)

# Out [8]:
Epoch 500, Loss 7.860116
Epoch 1000, Loss 3.828538
Epoch 1500, Loss 3.092191
Epoch 2000, Loss 2.957697
Epoch 2500, Loss 2.933134
Epoch 3000, Loss 2.928648
Epoch 3500, Loss 2.927830
Epoch 4000, Loss 2.927679
Epoch 4500, Loss 2.927652
Epoch 5000, Loss 2.927647

tensor([5.3671, -17.3012], requires_grad=True)
```

#### 5.5.2 Optimizers a la carte

在上面的代码里，我们使用了普通的gradient descent来优化，对于我们这种简单的例子是足够的。而实际上还有一些可供选择的optimization方法，它们对于更复杂的情况可能更加有效。

我们将会在后续章节里介绍更复杂的optimization方法，而现在可以介绍一下PyTorch中为我们已经封装好的这些optimization algorithms。torch module有一个optim submodule，在其中我们能找到实现了各种optimization algorithm的classes。

```python
# In [1]:
import torch.optim as optim

dir(optim)

# Out [1]:
['ASGD',
'Adadelta',
'Adagrad',
'Adam',
'Adamax',
'LBFGS',
'Optimizer',
'RMSprop',
'Rprop',
'SGD',
'SparseAdam',
...
]
```

上述每一个optimizer constructor都接受一个parameters的list（parameters也就是PyTorch的tensors，而且它们的requires_grad都是True）作为输入。所有通过optimizer的parameters都在optimizer object内部被保留了，所以optimizer可以更新他们的值，并且可以访问它们的grad attribute，正如下图figure10所示：

![optimizers]({{ '/assets/images/DLP-5-10.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 10. (A) Conceptual representation of how an optimizer holds a reference to parameters. (B) After a loss is computed from inputs, (C) a call to .backward leads to .grad being populated on parameters. (D) At that point, the optimizer can access .grad and compute the parameter updates.*

每个optimizer都有两个methods：zero_grad和step。zero_grad会在构建的时候，将所有传给optimizer的parameters的grad attribute都置零。而step会根据这个optimizer的algorithm来更新这些parameters的值。


**USING A GRADIENT DESCENT OPTIMIZER**

我们来构建一个params，并初始化一个gradient descent optimizer：

```python
# In [1]:
params = torch.tensor([1.0, 1.0], requires_grad = True)
learning_rate = 1e-5
optimizer = optim.SGD([params], lr=learning_rate)
```

这里SGD是stochastic gradient descent的简称。实际上，这个optimizer也是个vanilla gradient descent算法（只要momentum设置为0.0）。stochastic源于每一次的gradient都是通过平均一个所有input的随机子集来计算得来的，这个随机子集称为minibatch。而optimizer并不知道这个gradient是用所有input算的，还是只用了随即子集来算的，但更新方式是一样的，所以还是vanilla gradient descent算法。

```python
# In [2]:
t_p = model(t_u, *params)
loss = loss_fn(t_p, t_c)
loss.backward()

optimizer.step()

params

# Out [2]:
tensor([9.5483e-01, -8.2600e-04], requires_grad = True)
```

注意到，params的值通过调用optimizer的step method就被更新了，而并不需要自己来进行什么操作。

而现在我们已经可以将上述代码加入我们的training loop里了么？还不行，我们还没有zero_grad每次更新的值。下述代码可以直接被加入training loop里：

```python
# In [1]:
params = torch.tensor([1.0, 1.0], requires_grad = True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr = learning_rate)

t_p = model(t_un, *params)
loss = loss_fn(t_p, t_c)

optimizer.zero_grad()
loss.backward()
optimizer.step()

params

# Out [1]:
tensor([1.7761, 0.1064], requires_grad = True)
```

我们可以看到，optim module帮我们省了更新模型里每个参数的操作，我们只需要将它们整合为一个list提供给optimizer，就可以自动进行更新了，只不过这个list可能会十分的长。

从而我们的training loop即为：

```python
# In [1]:
def traning_loop(n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
        
    return params

# In [2]:
params = torch.tensor([1.0, 1.0], requires_grad = True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)    # 就是这里

training_loop(
    n_epochs = 5000,
    optimizer = optimizer,
    params = params,                                 # 这里training_loop函数里params参数赋值的params，要和上面optimizer里的params一直，否则optimizer不知道该更新什么
    t_u = t_un,
    t_c = t_c)

# Out [2]:
Epoch 500, Loss 7.860118
Epoch 1000, Loss 3.828538
Epoch 1500, Loss 3.092191
Epoch 2000, Loss 2.957697
Epoch 2500, Loss 2.933134
Epoch 3000, Loss 2.928648
Epoch 3500, Loss 2.927830
Epoch 4000, Loss 2.927680
Epoch 4500, Loss 2.927651
Epoch 5000, Loss 2.927648

tensor([5.3671, -17.3012], requires_grad=True)
```

**TESTING OTHER OPTIMIZER**

如果我们想用其它的optimizer，只需要实例化另一个optimizer就可以了。以Adam来举例，这个optimizer对于parameters的scaling更不敏感，即使我们用非normalized的数据，并且learning_rate设置为1e-1，它仍然可以正确学习：

```python
# In [3]:
params = torch.tensor([1.0, 1.0], requires_grad = True)
learning_rate = 1e-1
optimizer = optim.Adam([params], lr=learning_rate)

training_loop(
    n_epochs = 2000,
    optimizer = optimizer,
    params = params,
    t_u = t_u
    t_c = t_c)

# Out [3]:
Epoch 500, Loss 7.612903
Epoch 1000, Loss 3.086700
Epoch 1500, Loss 2.928578
Epoch 2000, Loss 2.927646

tensor([0.5367, -17.3021], requires_grad=True)
```

optimizer并不是我们的training loop里唯一可以替换的东西。让我们把注意力集中到模型上。我们可以将上述的model function替换成neural network。虽然我们知道这个温度计问题的背后的model就是线性的，但这并不影响我们使用neural network来拟合。我们将在chapter6里说。

我们已经接触了很多核心的概念，使我们能够训练一些很复杂的deep learning models，而且理解它们背后的原理：反向传播来估计gradients，autograd，以及利用optimizers来更新模型的参数。剩下的内容便只有补漏补缺。

接下来我们来看如何划分我们的数据，这个可以很好的锻炼我们控制autograd的能力。

#### 5.5.3 Training, validation and overfitting

在Kepler的工作里，他将一部分的数据留在一边不适用，而使用它们来验证自己猜想的模型的正确性。这个事情很重要，特别是当我们的模型能够拟合任意函数的时候，比如neural networks。一个高度可适应化的模型会用他十分多的parameters来尽可能的拟合数据点，使得loss尽可能地低，但这并不保证这个学习后的模型对于未见过的数据还表现良好。产生这种现象的原因是，毕竟我们要求optimizer来使得loss达到最小，而且optimizer就是通过可见到的数据来计算的。而对于训练数据loss很小，对于独立的验证数据loss很大的情况，叫做overfitting。

我们用来阻止overfitting的第一个操作是认识到overfitting可能会出现这样的事实。我们需要在数据集中留下一部分不使用，作为validation set，只使用剩下的数据来拟合我们的模型，training set，如figure11所示。从而我们就可以用validation set来验证我们所学习到的模型的好坏。

![overfitting]({{ '/assets/images/DLP-5-11.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 11. Conceptual representation of a data-producing process and the collection and use of training data and independent validation data.*

**EVALUATING THE TRAINING LOSS**

training loss可以告诉你模型对于训练数据拟合的好坏，以及模型能否具有拟合数据的能力。一个deep neural network理论上可以拟合任意的函数，只要neurons，也就是参数，足够的多。参数越少，我们的neural network所能拟合的函数就越简单。所以说，规则1：如果training loss不再下降，那可能是对于这个数据，这个模型过于简单了。另一种可能性是数据本身就不具有能够解释输出的信息，比如在温度计的例子里，输入是压力，而输出是摄氏度，它们之间没有任何关系，也就无法学习到相对应的关系。


**GENERALIZING TO THE VALIDATION SET**

所以说validation set是什么？如果validation set的loss并不随着training loss的降低而降低，那说明我们的模型在增进自己拟合训练数据的能力，而不是在增加自己generalizing到这个训练数据集合背后的数据分布里的数据的能力。一旦我们在新的，未见过的数据上测试训练好的模型，loss就会变得很大。所以说，规则2：如果training loss和validation loss分道扬镳了，那说明我们overfitting了。

让我们更加深入的研究一下这个现象，使用之前的温度计的例子。我们可以用一些更复杂的模型来拟合，比如分段多项式，或者一个很大的neural network。这些模型可以学习到一个能够完美拟合所有训练数据的的模型，如figure12下面的图所示，因为这样的话，training loss会变成0，也就是最小。但这样的模型对于新的数据，效果会很不好。

![over]({{ '/assets/images/DLP-5-12.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 12. Rather extreme example of overfitting.*

所以说，怎么解决这个问题呢？这是个好问题。首先，我们要保证我们有足够多的数据。如果我们使用很低的频率从一个正弦函数上采样，那么能恢复原采样函数的几率是很小的。

假设我们已经有了足够多的数据，我们需要保证我们能够拟合training data的模型在未见到的数据上仍然能够尽可能地保持稳定的表现。有多种方式来实现这个目标。一种方法是给loss function加上penalization terms，从而使得模型更容易变得光滑，且模型本身改变的更慢一点。另一种方法是对于input数据加上噪声，并使得模型也去拟合加了噪声的数据。还有很多种不同的方法。但我们需要注意的一点是，尽量使用简单的模型。简单的模型可能不如复杂的模型拟合训练数据拟合的好，但简单的模型在数据之间表现得更加稳定。

所以我们得到了一个trade-off。一方面，我们需要具有足够学习能力的模型来拟合数据集，另一方面我们也需要这个模型不要overfitting。因此，为了选择具有合适参数的neural networks，这个过程分两步，逐步增加模型的size直到它能拟合好训练数据；再逐渐减小模型的size直到它停止overfitting。

我们会在chapter12里更加详细的解释。现在，我们来看对于我们的例子，我们如何将数据分为training set和validation set。我们先将t_u打乱，但保持t_u和t_c的对应关系不变，再将打乱的set分为两个部分：

```python
# In [1]:
n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

train_indices, val_indices

# Out [1]:
(tensor([9, 6, 5, 8, 4, 7, 0, 1, 3]), tensor([ 2, 10]))

# In [2]:
train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]

val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]

train_t_u = train_t_u * 0.1
val_t_u = val_t_u * 0.1
```

我们的training loop其实并没有过多的更改，我们再每个epoch验证一下我们的validation set的loss，来看看是否overfitting了。

```python
# In [3]:
def training_loop(n_epochs, optimizer, params, train_t_u, train_t_c, val_t_u, val_t_c):
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)
        
        val_t_p = model(val_t_u, *params)
        val_loss = loss_fn(val_t_p, val_t_c)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        if epoch <=3 or epoch % 500 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss.item(): .4f},"
                  f"Validation loss {val_loss.item(): .4f}")
    
    return params

# In [4]:
params = torch.tensor([1.0, 1.0], requires_grad = True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)

training_loop(
    n_epochs = 3000,
    optimizer = optimizer,
    params = params,
    train_t_u = train_t_un,
    train_t_c = train_t_c,
    val_t_u = val_t_un,
    val_t_c = val_t_c)

# Out [5]:
Epoch 1, Training loss 66.5811, Validation loss 142.3890
Epoch 2, Training loss 38.8626, Validation loss 64.0434
Epoch 3, Training loss 33.3475, Validation loss 39.4590
Epoch 500, Training loss 7.1454, Validation loss 9.1252
Epoch 1000, Training loss 3.5940, Validation loss 5.3110
Epoch 1500, Training loss 3.0942, Validation loss 4.1611
Epoch 2000, Training loss 3.0238, Validation loss 3.7693
Epoch 2500, Training loss 3.0139, Validation loss 3.6279
Epoch 3000, Training loss 3.0125, Validation loss 3.5756

tensor([5.1964, -16.7512], requires_grad=True)
```

在我们上述的例子里，validation set有点偏小了，所以validation loss只是针对两个点进行衡量。我们注意到，validation loss一直都比training loss要大。我们的模型本身就会在training set上表现得更好，因为它的参数就是从training set中学来的，但我们的主要目的是使得training loss和validation loss同步下降。最理想的状态是validation loss和training loss曲线挨得很近，这说明我们的模型generalization的能力很好。在figure13里，C是理想状态，D是能接受的状态，A是模型压根就没有学习，B是overfitting。我们将会在chapter12里学到更多的overfitting的例子。

![learning]({{ '/assets/images/DLP-5-13.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 13. Overfitting scenarios when looking at the training (solid line) and validatoin (dotted line) losses. (A) Training and validation losses do not decrease; the model is not learning due to no information in the data or insufficient capacity of the model. (B) Training loss decreases while validation loss increases: overfitting. (C) Training and validation losses decrease exactly in tandem. Performance may be improved further as the model is not at the limit of overfitting. (D) Training and validation losses have different absolute values but similar trends: overfitting is under control.*


#### 5.5.4 Autograd nits and switching it off

在之前的training loop代码里，我们仅仅对train_loss调用了backward方法。因此，errors仅仅会根据training set来反向传播，validation set是用来衡量模型在未训练数据上的表现的。

那么问题来了，模型在每次epoch里被evaluate了两次，一次在train_t_u上，一次在val_t_u上，然后再backward。那么为什么autograd不会出错呢？backward的值难道不会因为validation set所计算的loss而受到影响么？

在我们这个例子里，不会出现这个问题。model在train_t_u上evaluate得到train_t_p，然后train_loss是通过train_t_p计算来的，从而这个过程形成了一个computation graph，将train_t_u连到train_t_p，再连到train_loss。而model再在val_t_u上evaluate得到val_t_p，然后val_loss是通过val_t_p计算来的，这个过程同样形成了一个computation graph，将val_t_u连到val_t_p，再连到val_loss。虽然它们都使用了同一个model，也就是使用了同样的parameters，但是是两个独立的computation graph，如figure14所示。所以，之后调用train_loss.backward()，其会将train_loss对于params的derivatives放在params的grad attribute里，而这个过程，和val_loss没有一点关系。

![separate]({{ '/assets/images/DLP-5-14.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 14. Diagram showing how gradients propagate through a graph with two losses when .backward is called on one of them.*


这两个computation graph唯一共同的tensors就是parameters。当我们调用train_loss的backward method时，我们是在第一个computation graph上调用的。也就是说，我们基于train_t_u计算得来的train_loss对于parameters的derivatives被加到了parameters的grad attribute上。

如果我们也在val_loss上调用了backward() method，我们将会将通过val_loss计算的params的derivatives也加到它的grad attribute上。从而实际上，我们每个epoch就是使用了整个数据集来训练的，因为gradient实际上用了training set和validation set里所有的数据来计算。

还有另一个值得注意的地方。既然我们对于val_loss从未调用backward method，PyTorch为什么还要构建一个computation graph呢？我们可以只是将model和loss_fn作为普通的function来使用，而不建立computation graph。这样可以省下大量的算力。

为了解决这个问题，PyTorch允许我们在不需要它的时候关掉autograd，使用torch.no_grad上下文管理系统。这个在问题和模型很庞大的时候，可以节省很多空间和算力。

```python
# In [1]:
def training_loop(n_epochs, optimizer, params, train_t_u, val_t_u, train_t_c, val_t_c):
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)
        
        with torch.no_grad():
            val_t_p = model(val_t_u, *params)
            val_loss = loss_fn(val_t_p, val_t_c)
            assert val_loss.requires_grad == False
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

```

使用相关的set_grad_enabled上下文管理，我们也可以使得代码运行在autograd启动或者不启动的状态下，其取决于Boolean expression的值。

```python
# In [1]:
def calc_forward(t_u, t_c, is_train):
    with torch.set_grad_enabled(is_train):      # 下面的loss和t_p的requires_grad是否是True，取决于is_train是True还是False
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
    return loss
```


### 5.6 Conclusion

本章的开始，我们提出了一个大问题：机器是如何从例子中学习的？我们用了本章剩下的部分解释了它的机理，并用例子来详细解释。

到现在，我们已经具备了所有的知识，从而Chapter6，我们将会学习如何使用neural network来拟合数据。我们将会同样解决温度计的问题，但是使用torch.nn提供的更有力的工具。我们同样也是使用这样一个简单的例子来解释PyTorch的工作原理。从而能让我们明白我们需要什么来训练一个神经网络。


### 5.7 Summary

* 线性模型是拟合数据的最简单的有效模型
* 凸优化可以被用于线性模型，但不能被推广到neural networks上，所以我们使用stochastic gradient descent来优化neural networks
* deep learning可以被用于很广泛的应用上，其并不是为了某个特殊的任务而设计，而是一种很普遍的模型
* learning algorithm就是利用数据来优化模型的参数。一个loss function就是用来衡量错误指标的。learning algorithm的目标就是使得loss function尽可能的小
* loss function对于parameter的gradient可以用于更新该parameter
* PyTorch里的optim module提供了已经包装好的optimizers，可以用来最小化loss function以及更新parameters
* optimizers使用PyTorch的autograd特性来计算parameters的gradient，其取决于parameter如何贡献给那个做了.backward()的tensor。这个特性允许用户在复杂的forward时可以依赖dynamic computation graph
* 上下文管理器with torch.no_grad():可以用来控制autograd的开关
* 数据通常都被分为training set和validation set。这个使得我们可以在未被使用过的数据上测试训练好的模型的性能
* 当模型在training set上的效果越来越好而在validation set上的效果越来越差的时候，overfitting就发生了。这是因为模型没有generalize，而只是记住了training set里的信息。



## Chapter6 Using a neural network to fit the model

>本章内容包括
>* nonlinear activation functions是和linear models最关键的不同点
>* 使用PyTorch的nn module
>* 利用neural network来解决一个linear-fit问题


到现在为止，我们已经学习了一个linear model该如何学习，以及如何利用PyTorch来实现这个模型。我们研究了一个十分简单的regression问题，它们使用仅有一个input和一个output的linear model。研究如此简单的问题使得我们可以将模型如何学习的机制进行分解，而不需要过分注意模型内部的细节。正如我们在chapter5里的figure2里所见到的（本章figure1重复了这个图），一个模型内部的细节对于理解学习的过程是不重要的。将errors反向传播到parameters以及利用loss对于parameters的gradient来更新这些parameters，对于模型本身是什么样的并不重要。

![diagram]({{ '/assets/images/DLP-6-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 1. Our mental model of the learning process, as implemented in chapter5.*

在本章里，我们对模型结构进行一些改变：我们将会用一个neural network来解决之前的温度计问题。我们还会使用上一章里的training loop，并且数据集也还是分为training set和validation set。

这一章是我们开始介绍PyTorch里那些你做你自己的project的时候所需要经常使用的内容。你会对PyTorch API下面到底发生了什么有跟多的了解，而不仅仅只是调用它们。在我们设计新模型之前，先介绍一下什么是artificial neural network。

### 6.1 Artificial neurons

deep learning的核心是neural networks：通过一系列简单函数的复合能够表示复杂函数关系的数学实体。neural networks这个术语是由脑神经科学所引来的。事实上，尽管早期的模型是由neuroscience所启发的，但现在的artificial neural networks仅仅和脑中神经元的工作方式有一丝的相似。artificial和physiological neural networks都用了相似的方式来近似很复杂的函数关系，因为这种方式十分的高效。

neural networks的基本构成模块是neuron，如figure2所示。它其实就是对input做一个线性变换（乘上weight，再加上bias），之后再通过一个nonlinear function的计算（这个function叫做activation function）。

![neuron]({{ '/assets/images/DLP-6-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 2. An artificial neuron: a linear transformation enclosed in a nonlinear function.*

数学上来说，我们可以将一个neuron的输出写为$$o = f(w * x + b)$$，其中$$x$$是输入，$$w$$是weight，$$b$$是bias，$$f$$是activation function，可以是hyperbolic tangent，tanh或者ReLU，等。一般情况下，x以及o，可以是一个scalar，也可以是一个vector，而相对应的w可以是scalar或者matrix，b可以scalar或者vector。对于x是vector，w是matrix，b是vector的情况，这代表了一个layer的neurons。


#### 6.1.1 Composing a multilayer network

一个multilayer neural network，如figure3所示，就是由我们上面所描述的那些neuron function所组成的：

$$x_1 = f(w_0 * x + b_0)$$
$$x_2 = f(w_1 * x_1 + b_1)$$
$$...$$
$$y = f(w_n * x_n + b_n)$$

每一层neuron的输出，都作为下一层neuron的输入。$$w_0, w_1, ..., w_n$$是matrices，而$$x$$是vector。

![nn]({{ '/assets/images/DLP-6-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 3. A neural network with three layers.*


#### 6.1.2 Understanding the error function

我们现在所用的neural network和之前所用的linear model一个很重要的区别在于error function的形状。linear model以及error-squared loss function有唯一一个确定的最小值点。使用合适的optimization algorithm，我们就能使得模型学习到唯一的最优值点。

而neural networks并没有error function是convex的这一个性质，即使使用相同的error-squared loss function。对于我们所要估计的模型的parameter，它们并不会每次都收敛到唯一的点。我们尝试使得所有的parameters一致作用得到有用的output。因为这个有用的output只是在近似事实，所以永远会存在一些不完美的地方。这些不完美的地方会怎么体现在哪里体现都是不知道的，从而控制出现这些不完美的parameters在某种程度上也是不知道的。

neural network有着non-convex的error surface很大程度上因为activation function。全体neurons能够拟合很复杂函数的能力取决于每个neuron所体现出的linear和nonlinear的性质。


#### 6.1.3 All we need is activation

正如我们所看到的，neural networks里最基本的组成部分就是一个linear operation后面跟着一个activation function。在chapter5里我们已经研究了linear operation。而activation function主要起到两个作用：
* 在neural network的中间层里，activation function使得output function对于不同的value有不同的slope，而这点仅仅有linear function是做不到的。将这些不同sloped的部分按某些方式组合起来，neural networks就可以拟合任意函数，我们在6.1.6里将会详述。
* 在neural network的最后一层，activation function还起到了将先前的linear function计算的结果集中到某一个区间内的作用。

让我们来说说上面第二点是什么意思。假设我们要给图片按照图片是否像狗进行打分，那么金毛或者拉布拉多的照片就应该打高分，而飞机或者垃圾箱的照片就该打低分。熊的照片也该打低分，但要比飞机和垃圾箱打的分数要高。

问题是，我们需要定义一个高分：我们的分数区间是无穷的，有整个float32的区间来使用，这意味着我们可以打十分高的分数。如果我们没有activation function，我们只有linear function的话，那输出就是$$w * x + b$$，其是没有上下限的。

**CAPING THE OUTPUT RANGE**

我们想要完全将我们的linear function的输出限定在一个范围内。一个可能的方法是直接将小于0的置为0，大于10的置为10.这个简单的操作也有对应的activation function：torch.nn.Hardtanh，这个函数的default range是-1到1。

**COMPRESSING THE OUTPUT RANGE**

另一个种类的效果不错的一系列functions就是torch.nn.Sigmoid，包括$$1/(1 + e ** (-x)$$, torch.tanh，等。这些函数有一个渐进的曲线，值域在0到1之间，并在无穷大逼近1，在负无穷大逼近0，并且在0附近有着近似常数的slope。这个function应该会工作的很不错，因为其中间一部分对于linear function的输出很敏感，而两侧饱和部分数据都被挤在了一起。如figure4所示，垃圾箱图片的分数为-0.97，熊，狐狸，狼等在-0.3到0.3的区间内。

![sigmoid]({{ '/assets/images/DLP-6-4.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 4. Dogs, bears, and garbage trucks being mapped to how dog-like they are via the activation function.*

```python
# In [1]:
>>>import math
>>>math.tanh(-2.2)  # garbage truck
-0.9757431300314515
>>>math.tanh(0.1)   # bear
0.09966799462495582
>>>math.tanh(2.5)   # good dog
0.98661429815143032
```

bear的照片在敏感区域内，所以它的linear function一点的改变，就会使得它的最后的得分改变很大。比如，将bear换成polar bear，因为其更具有dog的特征，它的分数要高很多，而换成koala bear，则因为更不像dog了而分数变低。

#### 6.1.4 More activation functions

有很多种activation functions，其中一部分展现在figure5里。

![activation]({{ '/assets/images/DLP-6-5.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 5. A collection of common and not-so-common activation functions.*

在第一列里，我们有光滑的activation functions，tanh和softplus，而第二列里的两个activation functions在某些点是不光滑的：hardtanh和ReLU。ReLU（recfied linear unit）是现在最火的activation function，表现良好。第三列里是sigmoid和leakyReLU。sigmoid又称为logistic function，在早期的模型里很常用，现在仅仅在需要输出区间在0到1的时候用，比如输出的是概率。LeakyReLU在左半区间有着很小的slope，而不是和ReLU一样直接为0。

#### 6.1.5 Choosing the best activation function

我们讨论一下activation functions一些共有的性质：

* 它们都是nonlinear的。重复的堆叠linear function最后得到的还是一个linear function。nonlinearity使得模型能够拟合复杂的functions。
* 它们是differetiable的。只有这样，gradient才能够传播回去。可数个非连续点是可以的，比如hardReLU。

没有上述两个特性，networks要么就没法训练，要么就变成了linear models。

下面的几个特性也是activation functions拥有的：

* 它们有至少一个sensitive range，输入的nontrivial变化会导致输出的nontrivial变化。这对于training来说很重要。
* 它们当中的很多还有insensitive range（saturated range），也就是说输入的变化基本不会导致输出有什么变化。

比如说，hardReLU可以轻易的通过组合sensitive range来拟合任意的函数。

而下面的性质，有很多activation functions有（但远远达不到所有的activation functions都有）：

* 在输入接近负无穷大的时候，输出接近或到达最下界
* 在输入接近正无穷大的时候，输出接近或到达最上界

回想一下反向传播是如何工作的，我们就可以知道，在输入位于activation function的sensitive range的时候，error对于parameters的更新是很有效的，而输入位于saturated range时，更新不太高效，因为这个时候，activation的gradient接近于0了。

将上面的所有特性放在一起，我们发现它们组成了一个很有力的机制：在一个由linear和activation units构成的network里，当不同的输入给到这个network，(a) 对于相同的输入，不同的neuron units会因为linear function的参数不同计算出不同的值，从而在activation function里作用在不同的range上；(b) 在训练过程中，这些输入所对应的errors会影响那些activation range在sensitive range的neurons，而那些activation range在saturated range的neurons受到的影响比较小（也就是不怎么更新这些neurons里的参数）。而且，大部分的activation function的sensitive range的derivative都接近于1，所以估计activation function之前的linear function的参数，和chapter5里估计linear model的参数的过程是差不多的。

我们现在来更加深入的理解，为什么将neuron摞起来做成很多层的neural network就有能够拟合任意函数的能力。neuron units不同的组合会使得最后得到的function有着不同的range（sensitive和saturated range），而这些units的parameters可以通过gradient descent比较容易的学习，因为learning在output还没有saturate之前表现得和linear function是很相似的。


#### 6.1.6 What learning means for a neural network

通过堆叠linear function + activation function，也就是neuron units来构造的模型，能够拟合十分复杂的函数，而且它的parameters可以通过gradient descent来有效地学习。这个过程对于具有上百万个parameters的模型仍然是有效的。使得使用neural networks十分有吸引力的一个点在于，我们不需要担心到底要设计什么样的模型，什么样的函数来解决这个问题。只要有neural network，我们就有了一个普适的estimator，并且它的parameters都可以用gradient descent来学习。neural networks的强大的模型拟合能力和学习能力，可以适应于任意的问题。figure6介绍了几个例子。

![neurons]({{ '/assets/images/DLP-6-6.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 6. Composing multiple linear units and tanh activation functions to produce nonlinear outputs.*

figure6里，四个左上侧的图，A，B，C，D是四个neuron，它们的前面的linear function的参数已经被随机设定好了。每个neuron都是用的tanh activation function，区间在-1到1之间。不同的linear function的参数使得这四个neuron具有不同的中心，以及不同的slope，但是它们都有着相似的变化趋势。最右侧第一排第二排显示了A+B和C+D的样子。这里我们已经开始看到一些有意思的结果，它们有些类似于一层neurons的样子。A+B是一个S型曲线，有着一个正峰和负峰。而C+D仅有一个正峰，而且最大值超过了1。

在第三行，我们将neurons组合起来，构成了两层的network。C(A+B)和D(A+B)都有着A+B所显示出的一个正峰一个负峰，而此时正峰却显得更加平缓。C(A+B)+D(A+B)的组合则显出了一个新的性质，两个明显的负峰，一个明显的正峰，以及一个十分平缓的正峰。

利用neural network成功学习到了任务，我们意思是对于生成training数据的数据生成器所生成的新的数据，我们仍然有正确的结果。一个成功训练的neural network，能够学习到数据内部的知识，从而对于未看到的数据仍有正确的判断结果。

让我们对于learning机制的理解更进一步：deep neural networks给我们提供了不需要明确设计模型的结构就可以拟合高非线性函数的能力。取而代之的是，从一个普适的，没有被训练的neural network，我们通过给它提供输入，输出以及loss function来让它通过误差反传学习到适用于我们任务的模型。将一个一般的的模型通过数据来specialize到适用于这个任务的模型这一过程，就是learning，因为这个模型一开始并不是为了这个任务而特别设计的，并没有描述该任务的规则被考虑进模型的设计中。

对于我们的温度计例子，我们假设两个温度计测量温度都是线性的。这个假设就为我们设计模型增加了一个隐形的规则：我们认为模型是线性的。然而随着数据的增多，我们很难再从数据的可视化里看出来规则。数学家和物理学家经常需要假设模型来解释数据，而neural networks不需要假设任何模型，它只需要利用数据便可以从一个十分一般的模型最后学习到适应于该任务的模型。


### 6.2 The PyTorch nn module

我们现在来看如何用PyTorch实现neural networks，第一步，先将之前的linear model换成neural network。虽然对于温度计这个任务，linear model已经足够了，但用neural network来实现这个简单的任务同样也可以清晰的介绍PyTorch实现neural network的整个过程。

PyTorch有一整个submodule都是为了neural network而设计的，叫做torch.nn。这个module含有所有的构建neural networks的组件，这些组件在PyTorch里称为modules（注意并不是PyThon里的module），而在其他深度学习框架里可能叫做layers。一个PyTorch module就是一个继承了nn.Module class的subclass。一个PyTorch module可以有parameters作为它的attributes，这些parameters就是那些在training process中被优化的那些tensors（比如linear model里的w和b）。一个PyTorch module还可以有一个或者多个submodules（也是nn.Module class的subclass）作为attributes，而且这个PyTorch module仍然可以追踪这些submodule里的parameters。

>如果一个PyTorch module有submodules的话，那这些submodules也得是top-level attributes，而不能隐藏在list，dict等里面。因为如果这样的话，optimizer就没法定位这些submodules，从而没法定位它们的parameters。当你的模型需要一个list或者一个dict的submodules时，PyTorch提供了nn.ModuleList和nn.ModuleDict这两个class以供使用。

并不奇怪的是，我们可以发现nn.Module有个subclass叫nn.Linear，是用来对输入做affine transformation的，其有两个parameter attributes，weight和bias。nn.Linear可以实现chapter5里我们的linear model。接下来，我们先将chapter5里的linear model用nn module来实现。


#### 6.2.1 Using __call__ rather than forward

所有的PyTorch提供的nn.Module的subclass都有它们自己的__call__ method。这允许我们在实例化一个nn.Linear之后，用类似于调用function的方式直接调用这个实例，比如：

```python
# In [1]:
import torch.nn as nn

linear_model = nn.Linear(1, 1)
linear_model(t_un_val)

# Out [1]:
tensor([[0.6018], [0.2877]], grad_fn=<AddmmBackward>)
```

调用一个nn.Module的实例（可能还会有一些arguments），基本等于调用这个实例的forward method（使用同样的arguments）。forward method是执行模型forward计算的部分，而__call__还会解决调用forward method之前以及之后一些杂活。所以说，技术上直接调用forward method也是可以的，他会产生和调用__call__ method一样的结果，但实际上我们最好还是不要用forward method：y = model(x)是正确的，但y = model.forward(x)可能会有潜在的错误。

下面是nn.Module的__call__ method的实现：

```python
>>> def __call__(self, *input, **kwargs):
        for hook in self.__forward_pre_hooks.values():
            hook(self, input)
        
        result = self.forward(*input, **kwargs)
        
        for hook in self.__forward_hooks.values():
            hook_result = hook(self, input, result)
            # ...
        
        for hook in self.__backward_hooks.values():
            # ...
        
        return result
```

从上面的代码可以看到，如果直接调用forward method，就会使得很多hooks不能被正确调用。


#### 6.2.2 Returning to the linear model

回到我们的linear model。nn.Linear实例化时接受三个arguments：the number of input features，the number of output features和需不需要bias（default的是True）：

```python
# In [1]:
import torch.nn as nn

linear_model = nn.Linear(1, 1)     # 输入输出的feature数量都是1，bias default为True
linear_model(t_un_val)

# Out [1]:
tensor([[0.6018], [0.2877]], grad_fn=<AddmmBackward>)
```

我们已经实例化了一个input和output feature都是1的nn.Linear，这个实例只有一个weight和一个bias：

```python
# In [2]:
linear_model.weight

# Out [2]:
Parameter containing:
tensor([[-0.0674]], requires_grad=True)

# In [3]:
linear_mode.bias

# Out [3]:
Parameter containing:
tensor([[0.7488]], requires_grad=True)
```

我们可以输入一些数来调用这个PyTorch module：

```python
# In [4]:
x = torch.ones(1)
linear_model(x)

# Out [4]:
tensor([0.6814], grad_fn=<AddBackward0>)
```

尽管上述代码是可以执行的，但实际上我们并没有提供具有正确dimension的输入。我们设计的linear model是接受一个input，输出一个output，而PyTorch的nn.Module以及它的subclass设计的是一次处理一批次的数据。为了能一次处理一批数据，这些PyTorch的modules的输入的第0维实际上应该是这一批次数据的数量，也就是batch size。我们在chapter4里已经遇到了这个，并且也知道如何将现实生活里的数据整合成批次的样子。

**BATCHING INPUTS**
任何nn里的PyTorch module都是设计成输入是一个batch的数据的。因此，假设我们需要用nn.Linear运行10个samples，我们可以构造一个input tensor，其size是$$B \times N_{in}$$，其中$$B$$是batch的size而$$N_{in}$$是input feature的number，我们将model运行一次：

```python
# In [5]:
x = torch.ones(10, 1)
linear_model(x)

# Out [5]:
tensor([[0.6814],
        [0.6814],
        [0.6814],
        [0.6814],
        [0.6814],
        [0.6814],
        [0.6814],
        [0.6814],
        [0.6814],
        [0.6814]], grad_fn=<AddmmBackward>)
```

让我们深挖看看到底发生了什么，figure7显示了有一个batch的数据的类似的情况。我们的input是$$B \times C \times H \times W$$，batch size设置为3，也就是$$B$$是3，狗，鸟，车的照片各一张；三个通道，也就是$$C$$是3，分别是RGB，以及图片的长宽。我们可以看到输出的tensor维度是$$B \times N_{out}$$，在这个图里，$$N_{out}$$是4。

![batch]({{ '/assets/images/DLP-6-7.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 7. Three RGB images batched together and fed into a neural network. The output is a batch of three vectors of size 4.*

**OPTIMIZING BATCHES**

我们需要做batch的原因是多方面的。一个很大的原因是为了占满计算资源。GPU的计算是高并行的，所以一个单独的输入会使得大部分的GPU处于空闲状态。而使用一个batch的input，可以将GPU所有部分都调动，其计算时间不会比计算单个输入的结果要慢多少。还有一个原因是有一些模型需要利用整个batch里的一些statistical information，而大的batch会使得这个information更加准确。

回到我们的温度计数据，t_u和t_c是两个size为B的1维数据。由于broadcasting，我们可以直接将模型写为$$w * x + b$$，而其中$$w$$和$$b$$都是scalars。在nn.Linear将input输入的时候，如果检测出并没有batch的信息，那它就会在前面加上一个维度，将所有的input变成单个sample，比如输入是一个size为10的1维tensor，作为nn.Linear的输入时，其就被转换为$$1 \times 10$$的2维tensor，也就是说其输入的feature number是10。之前的单个输入是可以的，是因为输入一个scalar，feature number按上述方式就变成了1，满足我们实例化的linear model。

在我们这个温度计的例子里，我们需要将输入的1维向量，转换为2维的，每行代表一个sample，每列代表的是features。即，将我们的1维size为B的tensor转换为2维size为$$B \times N_{in}$$的tensor，其中$$N_{in}$$是1。这可以用unsqueeze method很轻易地实现：

```python
# In [6]:
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c).unsqueeze(-1)
t_u = torch.tensor(t_u).unsqueeze(-1)

t_u.shape

# Out [6]:
torch.Size([11, 1])
```

处理好数据后，我们先用nn.Linear来代替手写的linear model，之后再将model的parameters传到optimizer里：

```python
# In [7]:
linear_model = nn.Linear(1, 1)
optimizer = optim.SGD(linear_model.parameters(), lr=1e-2)   # linear_model的parameters method，替代了之前手写模型里的[params]
```

先前，我们需要自己构造parameters，并将它们作为optim.SGD的第一个参数传入。而现在，我们可以直接使用parameters method来获取任何nn.Module object的所拥有的parameters以及它的submodules所拥有的parameters的一个list：

```python
# In [8]:
linear_model.parameters()

# Out [8]:
<generator object Module.parameters at 0x7f94b4a8a750>

# In [9]:
list(linear_model.parameters())

# Out [9]:
[Parameter containing:
tensor([[0.7398]], requires_grad=True), Parameter containing:
tensor([0.7974], requires_grad=True)]
```

nn.Module的parameters() method会递归到定义在这个PyTorch module的init constructor里的submodules，并且返回一个含有所有的parameters的一个简单的list（无嵌套），所以我们可以将parameters()直接传给optimizer，很方便。

我们已经知道在training loop里会发生什么了。optimizer被提供了一个list的tensors，它们都有着requires_grad=True，所以它们需要被gradient descent来优化。当training_loss.backward()被调用时，grad，也就是gradients的值就会被放到computation graph的节点上，这些节点也正好就是我们传给optimizer的这些tensors（即parameters）。从而，现在SGD optimizer已经有了所有所需要的东西。当optimizer.step()被执行时，每个传入optimizer的tensor都被更新了，而更新的值的大小与每个tensor的grad attribute里存的值成比例（乘上了learning rate）。

```python
# In [10]:
def training_loop(n_epochs, optimizer, model, loss_fn, t_u_train, t_u_val, t_c_train, t_c_val):
    for epoch in range(1, n_epochs+1):
        t_p_train = model(t_u_train)
        loss_train = loss_fn(t_p_train, t_c_train)
        
        t_p_val = model(t_u_val)
        loss_val = loss_fn(t_p_val, t_c_val)
        
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        if epoch == 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item(): 4f},"
                  f"Validation loss {loss_val.item(): 4f}")
```

这和chapter5里的training loop几乎差不多，除了我们不需要手动定义parameters以及手动将parameters传入optimizer。

还有一点我们可以从torch.nn里汲取：loss。nn带来了很多常见的loss functions，比如nn.MSELoss，mean squared error，和我们手写的error是一样的。nn里的loss functions仍然是nn.Module的subclass，所以我们将会直接实例化，并像调用function一样调用它。

```python
# In [11]:
linear_model = nn.Linear(1, 1)
optimizer = optim.SGD(linear_model.parameters(), lr=1e-2)

training_loop(
    n_epochs = 3000,
    optimizer = optimizer,
    model = linear_model,
    loss_fn = nn.MSELoss(),
    t_u_train = t_un_train,
    t_u_val = t_un_val,
    t_c_train = t_c_train,
    t_c_val = t_c_val)

print()
print(linear_model.weight)
print(linear_model.bias)

# Out [11]:
Epoch 1, Training loss 134.9599, Validation loss 183.1707
Epoch 1000, Training loss 4.8053, Validation loss 4.7307
Epoch 2000, Training loss 3.0285, Validation loss 3.0889
Epoch 3000, Training loss 2.8569, Validation loss 3.9105

Parameter containing:
tensor([[5.4319]], requires_grad=True)
Parameter containing:
tensor([-17.9693], requires_grad=True)
```


### 6.3 Finally a neural network

现在还有最后一步，将linear model换成neural network作为拟合函数。我们之前提到了，将拟合函数（也就是模型）换成neural network并不会有很高的效果，因为这个模型本身就是线性的。但我们还是用neural network来试试看。

#### 6.3.1 Replacing the linear model

我们将6.2里的所有部分都保持不变，除了model进行重新定义。我们来构建一个最简单的neural network：一个linear PyTorch module，跟着一个activation function，之后再将结果输入另一个linear PyTorch module。前面那层linear + activation一般被称为hidden layer，因为它的输出并不可见，而是作为输入给到了下一层。input和output的feature number都是1，但hidden layer的feature number一般要更宽一点。正如我们之前关于activation function所讨论的内容里说的，更多的中间层设计可以让不同的neuron对不同range的input产生回应（不同的neuron的sensitive range不一样），从而更多的neuron的组合可以增加模型的capacity。最后一层linear层会将前面的activation function的输出线性整合成output。

并没有一个如何画neural network的标准方法。figure8显示了两种常见的画法：左侧的一般用来解释neural networks的原理，而右侧的在论文中用的更多，用来更清晰的展示结构和hyperparameters。

![diagram]({{ '/assets/images/DLP-6-8.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 8. Our simplest neural network in two views. Left: begineer's version. Right: higher-level version.*

nn module提供了一个简单的方法来将PyTorch modules连起来：nn.Sequential container：

```python
# In [1]:
seq_model = nn.Sequential(
            nn.Linear(1, 13),    # 这里的13是随意选择的
            nn.Tanh(),
            nn.Linear(13, 1))    # 这里的13必须和前一个相吻合

seg_model

# Out [1]:
Sequential(
  (0): Linear(in_features=1, out_features=13, bias=True)
  (1): Tanh()
  (2): Linear(in_features=13, out_features=1, bias=True)
)
```

上述的seq_model是一个模型，它的输入是nn.Sequantial第一个参数（一个PyTorch module，也是一个模型）的输入，传给了nn.Sequential的第二个参数，以此类推直到最后一个，输出则是nn.Sequantial最后一个参数的输出。这个模型将输入feature number为1的input映射到中间层feature number为13的空间，再经过activation function，最后将这大小为13的feature线性组合输出为大小为1的output。

#### 6.3.2 Inspecting （审查） the parameters

调用model.parameters()会将第一层和第二层linear layer里的weight和bias都收集起来。我们将这些parameters的形状输出来看看：

```python
# In [2]:
[param.shape for param in seq_model.parameters()]

# Out [2]:
[torch.Size([13, 1]), torch.Size([13]), torch.Size([1, 13]), torch.Size([1])]
```

上述就是optimizer能获得的参数。一样的，在我们调用model.backward()之后，每个参数的grad attribute都会被加上gradient的值，然后在optimizer.step()之后，这些grad attribute会乘以learning_rate加到每个对应的参数之上，完成参数的更新。

我们再来讨论一下nn.Modules的一些参数。当检查由几个submodules组成的model的时候，直接通过name来辨别它们是比较高效的。实际上是有个method可以调用的：

```python
# In [3]:
for name, param in seq_model.named_parameters():
    print(name, param)

# Out [3]:
0.weight torch.Size([13, 1])
0.bias torch.Size([13])
2.weight torch.Size([1, 13])
2.bias torch.Size([1])
```

nn.Sequential里每个PyTorch module的name就是通过它在Sequential里的位置来命名的。有意思的是，nn.Sequential还可以接受OrderedDict作为输入，我们可以为每个PyTorch module自己命名：

```python
# In [4]:
from collections import OrderedDict

seq_model = nn.Sequential(OrderedDict([
    ('hidden_linear', nn.Linear(1, 8),
    ('hidden_activation', nn.Tanh(),
    ('output_linear', nn.Linear(8, 1))
]))

seq_model

# Out [4]:
Sequential(
    (hidden_linear): Linear(in_features=1, out_features=8, bias=True)
    (hidden_activation): Tanh()
    (output_linear): Linear(in_features=8, out_features=1, bias=True)
)
```

上述让我们对于submodules有了解释性更好的名字：

```python
# In [5]:
for name, param in seq_model.named_parameters():
    print(name, param)

# Out [5]:
hidden_linear.weight torch.Size([8, 1])
hidden_linear.bias torch.Size([8])
output_linear.weight torch.Size([1, 8])
output_linear.bias torch.Size([1])
```

这样的命名方式更具有描述性，但是并没有对于数据如何通过网络提供更多的灵活性。我们会在chapter8里看到如何通过subclassing nn.Module的方式来自定义model从而全面的控制数据的流动。

我们同时还可以通过使用submodules作为attributes来获取特定的参数：

```python
# In [6]:
seq_model.output_linear.bias

# Out [6]:
Parameter containing:
tensor([-0.0173], requires_grad=True)
```

这对于检查参数或者它们的gradients是有好处的：比如我们希望能在training的过程中监视gradients。假设我们希望能在trainingj结束后，将hidden layer的weight参数的gradient输出：

```python
# In [7]:
optimizer = optim.SGD(seq_model.parameters(), lr=1e-3)

training_loop(
    n_epochs = 5000,
    optimizer = optimizer,
    model = seq_model,
    loss_fn = nn.MSELoss(),
    t_u_train = t_un_train,
    t_u_val = t_un_val,
    t_c_train = t_c_train,
    t_c_val = t_c_val)

print('output', seq_model(t_un_val))
print('answer', t_c_val)
print('hidden', seq_model.hidden_layer.weight.grad)


# Out [7]:
Epoch 1, Training loss 182.9724, Validation loss 231.8708
Epoch 1000, Training loss 6.6642, Validation loss 3.7330
Epoch 2000, Training loss 5.1502, Validation loss 0.1406
Epoch 3000, Training loss 2.9653, Validation loss 1.0005
Epoch 4000, Training loss 2.2839, Validation loss 1.6580
Epoch 5000, Training loss 2.1141, Validation loss 2.0215

output tensor([[-1.9930],[20.8729]], grad_fn=<AddmmBackward>)
answer tensor([[-4.], [21.]])
hidden tensor([[ 0.0272],
               [ 0.0139],
               [ 0.1692],
               [ 0.1735],
               [-0.1697],
               [ 0.1455],
               [-0.0136],
               [-0.0554]])
```


#### 6.3.3 Comparing to the linear model

我们可以将学习之后的模型对所有数据都进行计算，来看看和一条直线之间的差距有多大：

```python
# In [8]:
from matplotlib import pyplot as plt

t_range = torch.arange(20., 90.).unsqueeze(1)
fig = plt.figure(dpi=600)
plt.xlabel("Fahrenheit)
plt.ylabel("Celsuis")
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.plot(t_range.numpy(), seq_model(0.1 * t_range).detach().numpy(), 'c-')
plt.plot(t_u.numpy(), seq_model(0.1 * t_u).detach().numpy(), 'kx')
```

结果显示在下图figure9里。我们可以看到neural network有一定程度的overfitting，正如我们在chapter5里所说的，它尝试完美拟合training数据。但尽管我们的neural network对于这个简单的问题参数过于多了，但学习的效果不是很差，还不错。

![result]({{ '/assets/images/DLP-6-9.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 9 The plot of our neural network model, with input data (circles) and model output (Xs). The continuous line shows behaviour between samples*



### 6.4 Conclusion

尽管用了很简单的例子，我们在chapter5和chapter6里讲了很多的内容。我们讲述了如何设计differentiable model并且如何利用gradient descent训练它，一开始用原始的autograd后来用了nn module。现在你应该已经知道PyTorch背后的运行机制是什么样的了。

### 6.5 Summary

* neural networks可以自动被适应到具体问题上
* neural network使得获取loss关于model里任意一个parameter的derivative都变得很容易，从而使得更新参数变得容易。多亏了PyTorch的automated differentiation engine，其才可以很轻松的提供那些derivatives
* linear function后的activation function使得neural networks有能够拟合任意复杂函数的能力，同时neural networks本身也不难优化
* torch.nn module和tensor标准库提供了构建neural networks的所有部件
* 为了识别出overfitting，将training set和validation set分开很重要。没有一个通用的方法来避免overfitting，但是有更多的数据，或者有更多样化的数据，或者用更简单的模型都是避免overfitting的好办法
* 任何做data science的人都需要经常plotting data来看看直观规律




## Chapter7 Telling birds from airplanes: Learning from images

>本章包括的内容：
>* 构建一个feed-forward的neural network
>* 使用Dataset和DataLoader来加载数据
>* 理解classification loss

上一章我们了解了通过gradient descent进行learning的内部机理，以及PyTorch提供的构造和优化模型的工具。我们用了一个简单的input feature number和output feature number都是1，而且潜在是线性模型的例子来做了示范，虽然内容都覆盖了，但确实例子简单了点。

在本章里，我们继续探索构建neural networks基础的内容。这次，我们将目光转向images。image recognition毫无疑问是人们了解到deep learning实力的重要问题。

我们将会逐步解决一个image recognition的问题，从一个很简单的neural network开始。这次，不用很简单的很小的数据集了，我们将用一个image dataset。


### 7.1 A dataset of tiny images

image recognition最基础的一个数据集是handwritten digit-recognition dataset，叫做MNIST。这里我们用一个同样简单的数据集，但是更加有趣。它叫做CIFAR-10，和它的兄弟CIFAR-100一样，它们被用作最基础的cv数据集几十年了。

CIFAR-10有着超过60000张大小为$$32 \times 32$$的彩色（RGB）图片，标有1-10的标签，代表10类：airplane（0），automobile（1），bird（2），cat（3），deer（4），dog（5），frog（6），horse（7），ship（8），truck（9）。现如今，CIFAR-10对于现在的任务过于简单了，但作为我们的实例是很合适的。我们将会用torchvision module来自动下载这个数据集并将它们加载为PyTorch tensors。figure1展示了CIFAR-10里的几张照片。

![CIFAR10]({{ '/assets/images/DLP-7-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 1 Image samples from all CIFAR-10 classes.*

#### 7.1.1 Downloading CIFAR-10

我们import torchvision module来使用其包含的datasets module来下载CIFAR-10数据集：

```python
# In [1]:
from torchvision import datasets
data_path = '../data-unversioned/p1ch7/'
cifar10 = datasets.CIFAR10(data_path, train=True, download=True)      # 为training数据实例化一个dataset；如果第一个argument没有下载到数据集，Torchvision会自己来下载数据集
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True) # train=False，会使得我们得到一个validation set，同样的，如果第一册argument所表明的地址没有该数据集，Torchvision则会自己下载
```

上述CIFAR10 function的第一个argument是我们要下载的数据集的地址；第二个argument指明了我们需要training set还是validation set；

除了CIFAR10，datasets submodule还为我们提供了绝大多数流行的CV数据集的预先处理好的获取途径，比如MNIST，Fashion-MNIST，CIFAR-100，SVHN，Coco和Omniglot。这些情况下，返回的数据集都是一个torch.utils.data.Dataset的subclass。我们可以看到，我们上面创建的cifar10数据集的method-resolution order里将torch.utils.data.Dataset作为了其的一个base class：

```python
# In [2]:
type(cifar10).__mro__

# Out [2]:
(torchvision.datasets.cifar.CIFAR10,
 torchvision.datasets.vision.VisionDataset,
 torch.utils.data.dataset.Dataset,
 object)
```


#### 7.1.2 The Dataset class

现在是解释一下torch.utils.data.Dataset的subclass是什么的好机会。如figure2所示，我们可以看到PyTorch Dataset是什么。它是一个object，必须要实现两个methods，也就是__len__和__getitem__。第一个是__len__，其需要返回这个dataset里item的数量；而后一个是__getitem__，其需要返回一个sample（在我们这个例子里还会返回对应的label）。

![datasets]({{ '/assets/images/DLP-7-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 2 Concepts of a PyTorch Dataset object: it doesn't necessarily hold the data, but it provides uniform access to it through __len__ and __getitem__.*

在Python里，如果一个Python object具有__len__ method，那么我们就可以用Python的内建函数len() function，其argument就是这个Python object本身：

```python
# In [3]:
len(cifar10)

# Out [3]:
50000
```

相似的，一个Python object具有__getitem__ method，我们可以用Python里index tuple或list的方法来获取这个object里的每个item。在我们的例子里，通过index，我们可以得到一个PIL（Python Imaging Library）image以及一个以integer表明的label：

```python
# In [4]:
img, label = cifar10[99]
img, label, class_names[label]

# Out [4]:
(<PIL Image.Image image mode=RGB size=$$32 \times 32$$ at 0x7FB383657390>,
 1,
 'automobile')
```

所以，cifar10数据集里的一个item是一个PIL image，我们将它显示看看：

```python
# In [5]:
plt.imshow(img)
plt.show()
```

结果如figure3所示：

![car]({{ '/assets/images/DLP-7-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 3 The 99th image from the CIFAR-10 dataset: an automobile.*


#### 7.1.3 Dataset transforms

我们现在还需要将PIL image转换为PyTorch的tensor，因为我们所操作的都是基于tensor的。这个时候torchvision.transforms module就有用了。这个module提供了一系列可组合的，像function的objects，它们可以作为argument被传给一个torchvision的dataset，比如torchvision.datasets.CIFAR10(...)，它们会在数据被加载进来但在__getitem__返回数据之前的时候对其进行transformation操作。torchvision.transforms module内包括的这样的objects有：

```python
# In [1]:
from torchvision import transforms
dir(transforms)

# Out [1]:
['CenterCrop',
 'ColorJitter',
 ...
 'Normalize',
 'Pad',
 'RandomAffine',
 ...
 'RandomResizedCrop',
 'RandomRotation',
 'RandomSizeCrop',
 ...
 'TenCrop',
 'ToPILImage',
 'ToTensor',
 ...
]
```

在上述的这些transforms里，我们看到了ToTensor这个特殊的transform，它会将Numpy arrays和PIL Image转换为PyTorch tensors。它还会将tensor按照$$C \times H \times W$$的方式输出。

让我们来试试ToTensor transform。torchvision.transforms.ToTensor实际上是一个class，一旦它被实例化了，就可以直接将一个PIL Image作为其的argument，输出就是一个PyTorch的tensor：

```python
# In [1]:
from torchvision import transforms

to_tensor = transforms.ToTensor()
img_t = to_tensor(img)
img_t.shape

# Out [1]:
torch.Size([3, 32, 32])
```

我们可以看到，torchvision.transforms.ToTensor()实例化后的to_tensor将输入的PIL Image转换为了一个$$3 \times 32 \times 32$$的tensor。注意到，对于这个image的label来说没有发生变化，其仍然是一个integer。

正如我们之前所说的，我们可以直接将一个torchvision.transform里的一个transform作为argument传给torchvision.datasets.CIFAR10：

```python
# In [2]:
tensor_cifar10 = torchvision.datasets.CIFAR10(data_path, train=True, download=True, transform=torchvision.transforms.ToTensor())
```

现在，获取该dataset tensor_cifar10里的一个值，将会返回一个tensor，而不是一个PIL Image:

```python
# In [3]:
img_t, _ = tensor_cifar10[99]
type(img_t)

# Out [3]:
torch.Tensor
```

正如我们所预料的，现在所获取的这个tensor第一个dimension将会是channel：

```python
# In [4]:
img_t.shape, img_t.dtype

# Out [4]:
(torch.Size([3, 32, 32]), torch.float32)
```

原PIL Image里的像素值是从0到255的，而且都是8-bits的整数，但ToTensor transform将数据转换为了32-bits的floating-point，而且值的范围变成了从0.0到1.0：

```python
# In [5]:
img_t.min(), img_t.max()

# Out [5]:
(tensor(0.), tensor(1.))
```

我们可以来显示一下：

```python
# In [6]:
plt.imshow(img_t.permute(1, 2, 0))
plt.show()

# Out [6]:
<Figure size 432 $$\times$$ 288 with 1 Axes>
```
![car_again]({{ '/assets/images/DLP-7-4.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 4 We've seen this one already.*

因为Matplotlib和PyTorch对图片的维度处理不一样，所以我们想要用plt来显示图片，需要先用PyTorch tensor的permute method将维度重新排列，才能使用plt进行显示，从figure4可以看到，其和figure3，也就是PIL Image直接显示的结果是一样的。


#### 7.1.4 Normalizing data

torchvision.transforms十分的有用，因为我们可以用torchvision.transforms.Compose将它们连起来。比如，我们可以轻易地将数据集转换为，沿着每个channel的mean是0，而standard variation是1。我们在Chapter4里就提到了这样的数据transform方法，但没有说原因。现在我们从chapter5和chapter6里可以知道，很多activation function的sensitive range都是在0附近的一小片区域，所以如果原始数据也位于这个区间内，经过linear function，activation function之后，在反向传播的时候，各个parameter的导数不至于接近0，从而使得learning更加高效。而且将每个channel内的数据都normalization之后，可以保证我们能使用同样的learning_rate来更新各个channel对应的parameter的值。这和我们在chapter5.4.4里的温度计例子里，将某些数据提升一个数据集从而使得weight和bias位于差不多的数量级的操作是类似的。

为了使得每个channel都有mean为0而standard variation为1，我们可以计算每个channel的整个数据集的mean和standard variation，然后$$v_n\[c\] = (v\[c\] - mean\[c\]) / stdev\[c\]$$。而这也正是torchvision.transforms.Normalize所做的。但mean和stdev的值需要预先计算好。我们先来计算一下CIFAR-10这个数据集每个channel的mean和stdev。

因为CIFAR-10这个数据集比较小，我们可以直接来计算它。我们先将所有的返回的数据tensor沿着一个新增的dimension堆起来：

```python
# In [7]:
imgs = torch.stack([img_t for img_t, _ in tensor_cifar10], dim=3)
imgs.shape

# Out [7]:
torch.Size([3, 32, 32, 50000])

# 现在我们来计算每个channel的mean和stdev：

# In [8]:
imgs.view(3, -1).mean(dim=1)

# Out [8]:
tensor([0.4915, 0.4823, 0.4468])

# In [9]:
# imgs.view(3, -1).std(dim=1)

# Out [9]:
tensor([0.2470, 0.2435, 0.2616])

# 我们现在有了这些数值，就可以来做torchvision.transforms.Normalize了：

# In [10]:
torchvision.transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))

# Out [10]:
Normalize(mean=(0.4915, 0.4823, 0.4468), std=(0.2470, 0.2435, 0.2616))

# 我们再将这个transform连在ToTensor这个transform的后面：

# In [11]:
transformed_cifar10 = datasets.CIFAR10(data_path, train=True, download=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))]))
```

注意到，现在我们从transformed_cifar10里获取一张图片来显示，已经不是真实的原始图片的样子了：

```python
# In [12]:
img_t, _ = transformed_cifar10[99]

plt.imshow(img_t.permute(1, 2, 0))
plt.show()
```
figure5展示了normalized的red car图片是什么样的，我们上面的操作已经将这个图片的值的范围变成了-1.到1.，而matplotlib要显示的图片的值的范围为0.到1.，小于0.的都直接是黑色，所以图片中有很多黑色的部分。但是数据仍然是在的，信息也是在的，只是matplotlib的显示并不能显示出来他们这些信息罢了。

![car_transformed]({{ '/assets/images/DLP-7-5.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 5 Our random CIFAR-10 image after normalization.*


### 7.2 Distinguishing birds from airplanes

假设我们需要一个model，能将bird和airplane的图片分开，如figure6所示的一个系统。我们从CIFAR-10里将bird和airplane的图片挑出来，并训练一个neural network来分辨他们。

![system]({{ '/assets/images/DLP-7-6.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 6 The problem at hand: we're going to help our friend tell birds from airplanes for her blog, by training a neural network to do the job.*


#### 7.2.1 Building the dataset

第一步就是先来处理数据。我们可以构建一个torch.utils.data.dataset.Dataset的subclass来表示仅有bird和airplane的数据集。然而这个数据集很小，而且我们仅仅需要其有index和len这两个作用。这个数据集实际上并不需要通过subclassing torch.utils.data.dataset.Dataset来实现。我们直接取之前生成的cifar10的一部分组成数据集：

```python
# In [1]:
label_map = {0:0, 2:1}
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label] for img, label in cifar10 if label in [0, 2]]
cifar2_val = [(img, label_map[label]) for img, label in cifar10_val if label in [0, 2]]
```

cifar2 object满足一个torch.utils.data.dataset.Dataset的基本条件，即有__len__和__getitem__这两个method（这是因为其是个list object，这两个method是list object自带的）。


#### 7.2.2 A fully connected model

我们在chapter5里学习了如何构建一个neural network。对于image这样的3维输入，我们可以直接将其展平为一个1维的tensor，从而这个维度就是输入的feature number，figure7展示了这样的过程：

![image]({{ '/assets/images/DLP-7-7.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 7 Treating our image as a 1D vector of values and training a fully connected classifier on it.*

我们输入的图片的tensor维度本来是$$3 \times 32 \times 32$$，展平之后就变成了一个3072大小的1维tensor。正如我们在chapter5里所构建的模型那样，那我们的nn.Linear的输入feature number就是3072，hidden layer的宽度设定为某个值，之后再通过一个activation function，之后再用另一个nn.Linear将其线性整合为一个feature number为2的输出：

```python
# In [1]:
import torch.nn as nn

n_out = 2

model = nn.Sequential(nn.Linear(3072, 512), nn.Tanh(), nn.Linear(512, n_out))  # 512是hidden layer的宽度
```

我们只是随意选择了一个hidden layer features的值，512。一个neural network需要至少有一个hidden layer（且有activations），从而才能在输入和输出之间引入nonlinearity，这样才能够有能力学习到任意函数（正如chapter6.3里所说的），要不然这就只是一个linear model。hidden features代表的是通过所学习到的weight matrix来将输入编码为某种representation。

现在我们已经有了一个模型，现在我们需要讨论一下输出应该是什么样的。


#### 7.2.3 Output of a classifier

在chapter6里，neural network直接输出一个连续的值作为输出。我们也可以做类似的输出：让我们的neural network输出一个单个的scalar，从而n_out=1，将label的值变成floating-point number，0.0就是bird，1.0就是airplane，从而可以使用MSELoss来计算输出和label之间的loss。这样做，我们就将这个问题变成了一个regression问题。但实际上，我们这个问题本质上和regression问题是有区别的。

我们需要意识到，output是categorical的：要不然是bird，要不然就是airplane，并不应该有中间的其他情况。正如我们在chapter4里所学的，当我们需要表示一个categorical variable时，我们需要将其转换为one-hot编码，比如bird的label是01而airplane是10。而对于整个CIFAR-10数据集，one-hot编码对于label来说也是可以的，因为也就是长度为10的vector而已。

在理想情况下，neural network对于bird图片将会输出torch.tensor(\[0.0, 1.0\])，对于airplane图片将会输出torch.tensor(\[1.0, 0.0\])。但我们的neural network并不会学习到完美的结果，所以输出是介于其中间的值。一个关键点是，我们可以将这个输出看作probability：输出的第一个值代表着是airplane的概率，而输出的第二个值代表的是bird的概率。

从上述概率的角度来考虑neural network的输出，会对其产生一些新的约束：
* output的每个值要在0.0到1.0之间
* output的所有值加起来要是1

而有一种十分聪明的办法能够使得上述要求满足，且所添加的约束还是differentiable的：softmax。


#### 7.2.4 Representing the output as probabilities

softmax是一个function，其输入是一个vector，而输出是同样size的vector，而满足我们上面对于probabilistic输出的要求。softmax的计算如figure8所示：

![softmax]({{ '/assets/images/DLP-7-8.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 8 Handwritten softmax.*


softmax就是，对于输入vector的每个值，取它的exponential，再除以这些exponentials的和，用代码来说就是：

```python
# In [1]:
def softmax(x):
    return torch.exp(x) / torch.exp(x).sum()

# In [2]:
x = torch.tensor([1., 2., 3.])
softmax(x)

# Out [2]:
tensor([0.0900, 0.2477, 0.6652])

# In [3]:
softmax(x).sum()

# Out [3]:
tensor(1.)
```

softmax是个单调函数，但并不是scale invariant的，比如上面的例子，输入的第一个和第二个element之间的ratio是0.5，而结果的第一个和第二个element之间的ratio是0.3678。

nn module同样也将softmax作为一个module放在里面了。因为input tensors经常会有第0个维度来表示batch，或者还有额外的维度表示别的信息，所以softmax module需要你特别指出是沿哪个维度来算的：

```python
# In [1]:
softmax = nn.Softmax(dim=1)

x = torch.tensor([[1., 2., 3.],
                  [1., 2., 3.]])

softmax(x)

# Out [1]:
tensor([[0.0900, 0.2447, 0.6652],
        [0.0900, 0.2447. 0.6652]])

```

上面的例子中，我们沿着dimension=1来做的softmax，就如同dimension=0代表着batch一样。

从而现在，我们可以在之前的neural network之后加上softmax来输出probability了：

```python
# In [1]:
model = nn.Sequential(nn.Linear(3072, 512), nn.Tanh(), nn.Linear(512, 2), nn.Softmax(dim=1))
```

我们在训练model前，其实也可以对于输入运行一下模型，看看输出到底是什么样的。我们先构造一个只有一张图片的batch，是一张鸟的图片,如figure9所示：

```python
# In [2]:
img, _ = cifar2[0]

plt.imshow(img.permute(1, 2, 0))
plt.show()
```

![bird]({{ '/assets/images/DLP-7-9.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 9 A random brid from the CIFAR-10 dataset (after normalization).*

为了能处理这个image，我们还需要将其转换为适应的维度，注意nn module都是需要让input的第0维是batch的：

```python
# In [3]:
img_batch = img.view(-1).unsqueeze(0)
```

```python
# In [4]:
out = model(img_batch)
out

# Out [4]:
tensor([[0.4784, 0.5216]], grad_fn = <SoftmaxBackward>)
```

我们通过torch.max，在指定维度的情况下，返回最大值以及最大值所在的index：

```python
# In [5]:
_, index = torch.max(out, dim=1)    # dim=0代表batch，dim=1代表沿着输出维度
index

# Out [5]:
tensor([1])
```

现在我们已经使得模型可以输出probability了，下一步，我们就需要定义合适的loss来训练整个网络了。


#### 7.2.5 A loss for classifying

在chapter5和chapter6里，我们用的是MSELoss。我们同样也可以还用MSELoss来让我们的输出靠近\[0.0, 1.0\]和\[1.0, 0.0\]。然而实际上我们并不是真的想让结果输出这样的值，我们实际上希望对于airplane的图片，第一个值要比第二个值大，而对于bird图片反过来。也就是说，我们更加想要对错误的分类做出惩罚，而不是要让输出结果尽可能的逼近0.0或者1.0。

我们需要maximize的是正确的那一个种类所对应的probability，也就是out\[class_index\]，out是softmax的输出，class_index对于airplane是0，对于bird是1。这个值，也就是正确分类的概率，被称为likelihood（给定数据，给定模型参数）。也就是说，我们需要有一个loss function，它在likelihood小的时候大，而在其大的时候小。

有一个loss function是这样的，称为negative log likelihood (NLL)。NLL = -sum(log(out_i\[c_i\]))，这个sum是所有N个samples的，而c_i指的是sample i对应的正确的类别，out_i是sample i的输出。让我们来看figure10，可以看出NLL满足我们的条件。

![NLL]({{ '/assets/images/DLP-7-10.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 10 The NLL loss as a function of the predicted probabilities.*

figure10说明当这个数据模型预测正确的probability很低的时候，NLL变成了无穷大，而NLL在预测正确的probability大于0.5的时候已经足够小了。注意，NLL是将预测正确的概率作为输入的。

总结一下，我们的classification loss可以这样进行计算。对于batch里的每一个sample：

* 运行forward pass，获得output的值
* 计算softmax，得到probabilities
* 将预测正确类别的概率值拿出来（这个值就是parameters的likelihood）。注意到我们是知道正确类别是哪一个的，因为这个是supervised learning，正确类别就是我们的ground truth
* 计算第三步里的log，加上一个负号，将其加到loss里

所以说，我们在PyTorch里该如何实现它呢？PyTorch有一个nn.NLLLoss class。但是这个class的计算是$$-sum(out_i \left[ c_i \right])$$，其中c_i是数据i的标签，而sum针对的是一个batch的所有的数据，我们可以看到求和之前并没有取log，所以这个class所期待的输入并不是probabilities来作为argument，而是log probabilities的tensor作为argument。之后他会计算对于一个batch的数据，这个模型的NLL。nn.NLLLoss这么做的原因是，当概率很小时，取log会使得值变成很大的负数。所以说，我们在使用nn.NLLLoss的时候，它的输入就最好是已经套上log的probabilities了，要不然理论上就不对了。PyTorch提供了一个解决办法，采用nn.LogSoftmax而不是nn.Softmax作为模型的最后一层（LogSoftmax实际就是log(softmax(x)))，之后再输入到nn.NLLLoss里，就在理论上正确了。

>一定要注意，nn.NLLLoss里定义的计算和negative likelihood loss本身理论上的计算公式是不一样的！

我们现在可以将我们的模型的softmax改成logsoftmax：

```python
# In [1]:
model = nn.Sequential(nn.Linear(3072, 512), nn.Tanh(), nn.Linear(512, 2), nn.LogSoftmax(dim=1))

# In [2]:

loss = nn.NLLLoss()
```

这个实例化的loss，将nn.LogSoftmax的输出作为第一个argument，并且将class indices的tensor（也就是ground truth）作为第二个argument。

```python
# In [3]:
img, label = cifar2[0]
out = model(img.view(-1).unsqueeze(0))
loss(out, torch.tensor([label]))

# Out [3]:
tensor(0.6509, grad_fn=<NllLossBackward>)
```

作为结束，我们来看看用cross-entropy loss如何改进MSE loss。figure11是cross entropy loss和MSE loss以predicted scores作为变量的函数图（predicted scores就是模型的output，还没有做softmax的结果），但是这两个loss都是在softmax之后再算的。我们可以看到，在cross entropy这个图里，在图的左侧，也就是预测正确的部分，即使靠近预测正确的点，其仍有slope，也就是说，在已经很正确的情况下，其还会继续改进。而在MSE这个图里，其很早就饱和了，也就是再远还没到达最优点的时候，函数的slope就已经很小了，进行不下去了。其内在的原因是，MSE的slope还是小了，softmax本身会使得错误的预测结果的曲线更加平缓一点，而MSE的slope太小，不足以抵消平缓造成的后果。这就是MSE为什么不适合classification的原因。

![entropy]({{ '/assets/images/DLP-7-11.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 11 The cross entropy(left) and MSE between predicted probabilities and the target probability vector(right) as functions the predicted scores——that is, before the (log-) softmax.*

#### 7.2.6 Training the classifier

现在，我们可以用chapter5里类似的training loop来训练我们的模型了，figure12解释了这个过程：

```python
# In [1]:
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(3072, 512), nn.Tanh(), nn.Linear(512, 2), nn.LogSoftmax(dim=1))

learning_rate = 1e-2

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

loss_fn = nn.NLLLoss()

n_epochs = 100

for epoch in range(n_epochs):
    for img, label in cifar2:
        out = model(img.view(-1).unsqueeze(0))
        loss = loss_fn(out, torch.tensor([label]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))    # 这只是输出了最后一个image的loss，在下一章里我们会改进的
```

![train]({{ '/assets/images/DLP-7-12.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 12 Training loops: (A) averaging updates over the whole dataset; (B) updating the model at each sample; (C) averaging updates over minibatches.*

仔细看的话，我们的training loop和chapter5里的有点不一样。在chapter5里，我们只有一个循环，也就是epoch循环（每个epoch指的是training set里所有的数据遍历一遍）。而在我们上面的代码里，在每个epoch循环里，我们还对所有的sample都进行了循环，也就是说在chapter5的训练里，每次我们都是将整个数据集计算之后的loss进行backward更新参数，而在上面的代码里，我们是对于数据集里的每个sample计算了loss直接更新。

对于chapter5里的training loop，所有的samples对应的loss被积聚在了一起来更新参数，而在上面的代码里，我们只用一个sample计算的loss来更新参数。然而对于某个sample计算出的合适的更新方向，对于其它的sample可能就不合适了。所以，我们在每个epoch里打乱所有samples的顺序，并且每次采用若干个samples计算累积的loss来更新参数，这若干个samples就叫做minibatch，这种操作为gradient descent带来了随机性。Stochastic gradient descent里的stochastic就是因为其作用在打乱了的minibatch上。很多实验证明，通过使用minibatch而不是整个数据集，可以使得optimization的过程收敛的更好，而且还能避开一些local minima，尽管minibatch所计算的gradients对于整个数据集的gradient其实差距很远。正如figure13所示，用minibatch算出来的gradients对于用所有数据计算的gradient的轨迹有随机的偏移，这也是我们为什么要用小一点的learning rate的原因。在每个epoch都打乱顺序随机采样minibatch可以保证在统计学上，minibatch的gradient的平均值会逼近用所有数据算出来的gradient。

![SGD]({{ '/assets/images/DLP-7-13.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 13 Gradient descent averaged over the whole dataset(light path) versus stochastic gradient descent, where the gradient is estimated on randomly picked minibatches.*

minibatch的大小是我们预先要设定好的，如同learning rate一样，这些叫做hyperparameters，叫这个名字是为了和模型里的parameters分开。

在我们上面的代码里，我们将minibatch的大小设置为了1，也就是每次取一个。torch.utils.data module有一个class，用来帮助打乱数据集以及将数据整理为minibatch的形式：DataLoader。一个DataLoader的作用就是从一个数据集里随机选择minibatch，而且给我们提供了不同的随机挑选方式。最常见的方式就是在每个epoch里按平均概率随机挑选。figure14显示了torch.utils.data.DataLoader打乱从torch.utils.data.dataset.Dataset里得到的index：

![DataLoader]({{ '/assets/images/DLP-7-14.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 14 A data loader dispensing minibatches by using a dataset to sample individual data items.*

让我们来看看这个过程是怎么完成的。一个torch.utils.data.DataLoader constructor至少有以下几个argument：一个torch.utils.data.dataset.Dataset作为input，batch_size表示size的大小，shuffle是一个Boolean值表示在每个epoch开始的时候，数据是否被打乱：

```python
# In [1]:
train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)
```

一个Dataloader是iterable的，所以我们可以直接在training loop内使用它：

```python
# In [1]:
import torch
import torch.nn as nn

train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)
model = nn.Sequential(nn.Linear(3072, 512), nn.Tanh(), nn.Linear(512, 2), nn.LogSoftmax(dim=1))
learning_rate = 1e-2
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()
n_epochs = 100

for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))   # 因为有shuffling，这个输出的loss是一个random batch的loss
```

在上述代码里的每个iteration，imgs是一个size为$$64 \times 3 \times 32 \times 32$$的tensor，也就是一个有着64张大小为$$32 \times 32$$的RGB图片的minibatch，而labels是一个size为64的1维tensor，内部含有每张图片的label信息。

上述代码的结果为：

```python
# Out [1]:
Epoch: 0, Loss: 0.523478
Epoch: 1, Loss: 0.391083
Epoch: 2, Loss: 0.407412
Epoch: 3, Loss: 0.364203
...
Epoch: 96, Loss: 0.019537
Epoch: 97, Loss: 0.008973
Epoch: 98, Loss: 0.002607
Epoch: 99, Loss: 0.026200
```

我们可以看到，loss确实在降低，但我们不知道它有没有降得足够低。既然我们的目标是希望这个模型能尽可能的对bird和airplane的图片正确分类，最好的办法就是测试模型在独立的validation set上的分类正确率：

```python
# In [2]:
val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in val_loader():
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        _, predicted = torch.max(outputs, dim=1)
        totoal += labels.shape[0]
        correct += int((predicted == labels).sum())
print("Accuracy: %f", correct / total)

# Out [2]:
Accuracy: 0.794000
```

上述准确率并没有很高，但已经比盲猜要高不少。我们的neural network是一个很shallow的classifier，它能够有效果就已经很不错了。它能有效果的一个原因也是因为我们的数据集和问题足够简单，这两类图片本身就有很明显的差别（比如背景颜色深度等等），所以模型其实可以根据几个pixels就能以很大的正确率来分类了。

我们当然可以让我们的模型更加复杂一点：

```python
# In [1]:
model = nn.Sequential(nn.Linear(3072, 1024), nn.Tanh(), nn.Linear(1024, 512), nn.Tanh(), nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 2), nn.LogSoftmax(dim=1))
```

nn.LogSoftmax和nn.NLLLoss的合并实际上等价于直接使用nn.CrossEntropyLoss。

cross entropy在理论上的定义是，对于同一个随机变量x的两个分布，q(x)和p(x)，p(x)是真实分布，而q(x)是估计的分布，我们需要衡量利用q(x)来采样x会和利用真实的p(x)来采样之间有多大的差距，定义为: $$-E_p \left[log q \right]$$。

而在PyTorch里，nn.CrossEntropyLoss这个术语实际上是PyTorch搞得一个特殊化，因为nn.NLLLoss实际上计算的是$$-sum(out_i \left[ c_i \right])$$，其中$$c_i$$是数据i的正确类别，sum是针对一个batch内所有的数据。如果nn.NLLLoss以log probability predictions作为输入的话，那实际上就是计算了模型预测结果的概率和真实标签概率之间的cross entropy，而nn.CrossEntropyLoss的计算是以scores（也叫logits）直接作为输入的cross entropy。也就是说，nn.CrossEntropyLoss就是把nn.LogSoftmax的计算也集成进来了，所以说nn.LogSoftmax再加上nn.NLLLoss就等同于nn.CrossEntropyLoss。技术上来说，前面有nn.LogSoftmax输出结果给nn.NLLLoss的时候，nn.NLLLoss计算的是一个将所有权重都放在target值上的Dirac distribution和log probability inputs所给出的predicted distribution之间的cross entropy，等同于没有nn.LogSoftmax输出结果，直接将nn.CrossEntropyLoss接到模型输出上的结果。

在信息论里，上述的这种cross entropy实际上也可以被理解成一个在target distribution知道的情况下，predicted distribution的negative log likelihood（可能省略了normalization）。也可以被理解为是在给定数据情况下，模型parameters的negative log likelihood。

所以说，既然nn.CrossEntropyLoss等于nn.LogSoftmax加上nn.NLLLoss，那我们就可以在模型中去掉nn.LogSoftmax这一层，并最后使用nn.CrossEntropyLoss：

```python
# In [1]:
model = nn.Sequential(nn.Linear(3072, 1024), nn.Tanh(), nn.Linear(1024, 512), nn.Tanh(), nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 2))

loss_fn = nn.CrossEntropyLoss()
```

按照上述代码，我们的模型会更加简洁，而且和原来的输出会一模一样，仅有的区别就是我们无法将模型的输出当作是probability了，要想这样的话，我们还需要单独做一次softmax。

我们按照之前的代码进行训练，可以发现，更复杂的网络会带来更好的效果，在validation set上取得了0.802000的accuracy，而此时training set上的accuracy已经达到了0.998100，这说明我们的模型已经overfitting了。我们的fully connected neural network在训练集上表现得很好，测试集上没有那么好。

PyTorch为我们提供了一个很快返回一个模型参数的method，也就是nn.Module有一个.parameters()的method（我们在optimizer里，用这个method将模型的parameters传给optimizer）。要想知道一个tensor里有多少elements，我们可以用它的numel method。将这两个方法结合起来，我们就可以得到这个模型的参数到底有多少个elements。在我们这个例子里，我们还需要注意这些参数是否有requires_grad设置为True。我们希望将模型的trainable参数和模型的总的大小分开考虑：

```python
# In [2]:
numel_list = [p.numel() for p in connected_model.parameters() if p.requires_grad == True]
sum(numel_list), numel_list

# Out [2]:
(3737474, [3145728, 1024, 524288, 512, 65536, 128, 256, 2])
```

我们可以看到，我们的模型有3.7million个parameters（按elements来算的），这对于一个简单的问题来说，模型有点过于复杂了。即使是我们一开始的模型，也有很多参数：

```python
# In [3]:
numel_list = [p.numel() for p in first_model.parameters() if p.requires_grad == True]
sum(numel_list), numel_list

# Out [3]:
(1574402, [1572864, 512, 1024, 2])
```

我们第一个模型的参数大约是第二个模型的一半。我们可以看到，之所以模型的参数这么大，有一个原因是因为我们的输入的feature size是3072，从而第一层输出的feature size如果是1024，也就是第二个模型的话，就已经有3072 $$\times$$ 1024 + 1024 = 3146752个参数了，是个很大的值。这说明，我们的fully connected neural network对于一个输入图像的pixel的number并不能很好的scale，也就是说如果pixel number很大的话，模型的参数会十分的多。比如说，如果我们输入的图片是1024 $$\times$$ 1024大小的，然后我们的第一层的输出的feature number是1024，那么我们第一层的parameters的数量就已经超过了3billion，使用32-floating-point number，这已经需要12GB的RAM来存储了，而我们这才只有一层，而且还没有计算backward和forward，这对于计算资源的需求是十分庞大的。

#### 7.2.7 The limits of going fully connected

让我们来看看将输入的图片展平为1维tensor之后再使用linear model，我们得到了什么，figure15显示了这个过程。

![fc]({{ '/assets/images/DLP-7-15.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 15 Using a fully connected module with an input image: every input pixel is combined with every other to produce each element in the output.*

整个过程就是，对于input image里的每个pixel的值，将其与其它所有其他的pixel的值按照某个权重计算出它们的一个linear combination，这就是output一个位置的值，而不同的位置所对应的权重是不一样的，但都是这样的一个input的linear combination。一方面，我们可以利用linear combination将input里的任意一个pixel和其它可能有关系的pixel结合起来。而另一方面，这样的作法忽略了pixel之间的位置关系，也就是说对于一个pixel来说，和它相邻的pixel与和它相距很远的pixel，对于它来说是一样的，也就是丢失了相对位置信息。

一个airplane的32 $$\times$$ 32的在天空中的图片大致是背景为蓝色的十字形。一个如figure15所示的fully connected neural network可能会学习到一个知识：当pixel(0,1)是暗色的，pixel(1,1)也是暗色的，并一直这样下去，那么这个图片很可能就是一张airplane的图片。这个过程在figure16的上半部分进行了阐述。然而，将airplane的位置在图片中移动，正如figure16下半部分所示，那我们可能就需要重新学习某些新的知识，这个时候，如果图片的pixel(0,2)是暗色的，pixel(1,2)也是暗色的，并一直这样，那么这个图片很可能是个airplane。用更专业的话来说，fully connected neural network不是translation invariant的。也就是说一个训练好的能够识别某个位置的物体的模型，在物体位置发生变化时将不一定能够识别。我们需要利用data augmentation来解决这个问题，也就是在训练的过程中，利用某些随机的translation来改变图片，从而network有机会在训练的过程中就已经见过了物体在各个位置的情况，而且我们需要对于每张训练图片都做类似的translation。

从而，在本章的最后，我们有了一个dataset，一个model，一个training loop，以及我们的model确实学习到了知识。然而，因为我们的问题和我们的模型并不完美匹配，最后模型仍然还是overfitting了，而并没有学习到我们希望模型在数据里能够generalize的内容。

我们使用的fully connected neural network是一个模型，其能够将input里的每个pixel的值与任意位置的其他pixel关联起来，而并没有它们之间相对位置的信息。然而我们有一个合理的假设，就是位置足够近的pixel它们的联系也应该足够紧密。而什么样的模型能够结合位置相对信息，而且能够translation-invariant呢？

解决方法就是将我们的模型改为使用convolutional layers。我们将在下一章讲解这个内容。


### 7.3 Conclusion

在这一章里，我们解决了一个简单的classification问题。我们从dataset开始，到model，到在training loop里优化loss。所有的这些内容都是标准的PyTorch深度学习过程。

我们也发现了我们这个模型一个严重的问题：我们将2维的输入image当作1维的vector来处理了。而且，我们并没有一个很自然的解决不translation invariant的方法。在下一章里，我们将会学习如何开发2维输入的信息。


### 7.4 Summary

* computer visoin是使用deep learning最多的一个领域
* 一些有标注的image dataset是公开的；它们当中的很多可以用torchvision直接获取
* torch.utils.data.dataset.Dataset和torch.utils.data.DataLoader给我们提供了一个简单但有效的加载和采样数据集的方式
* 对于一个classification任务，对于网络的输出使用softmax function可以使得输出能够被解释为probability。一个很有效的loss就是cross entropy loss，而nn.CrossEntropyLoss直接将取softmax和计算loss结合到了一起
* 我们将输入的2维images当成1维vector并没有问题，而使用fully connected neural network来处理这样的1维vector也没有问题。但是这样的话，原2维images里的位置信息将会被忽略掉
* 简单的模型利用nn.Sequential就可以直接构造了。







## Chapter8 Using convolutions to generalize

>本章内容包括
>* 理解convolution
>* 构造一个convolutional neural network
>* 创建用户自定义的nn.Module的subclass
>* 使用nn module和使用PyTorch自带的其它functional API来构建neural networks的区别
>* 设计neural networks的众多选择

在之前的章节里，我们构造了一个简单的neural network，其能够拟合数据（或者overfit），linear layers里众多的可供optimize的parameters为模型提供了很强的拟合能力。然而，这个模型还是有问题，它还是更倾向于记住training set里的信息，而不是真的学习到了可以generalize的分辨bird和airplane的图片的能力。基于这个模型的结构，我们猜到了之所以导致这个结果的原因。因为我们需要模型能够检测可能出现在图片内任意地方的bird或者airplane，所以我们的模型参数会很多（则容易overfitting），而且我们需要模型对物体出现在各个位置的图片都有学习（容易记住training set而不是学习到知识）。正如我们在上一章里所说的，我们可以通过data augmentation来使得我们的模型具有更好的generalization能力，但是参数过多这个问题还是解决不了。

对于上述问题，我们有一个好的解决办法，那就是将neural network里的dense fully connected affine transformation换成另一个linear function：convolution。


### 8.1 The case for convolutions

让我们从最基本的问题开始，convolution是什么以及我们怎么把convolution用在我们的neural network里。

在这个section里，我们将会发现convolution能解决translation invariance的问题，并且convolution利用了相对位置信息。

我们之前说，如同nn.Linear所做的，将输入的2维image看成1维vector，之后将其乘上一个n_output_features $$\times$$ n_input_features的weight matrix（再加上一个bias），表明对于每个channel，我们的output的每个element，都是所有的input pixels的一个线性加权和，而这些权重，就是weight matrix里的值，也就是我们要学习的值。

我们之前还说了，如果我们想要学习objects的特征，比如说蓝天里的airplane，我们很可能需要检查相邻的pixels是怎么样的，而不是检查相距很远的pixels在linear combination的计算里是怎么结合的。毕竟，如果我们想要学习airplane的图像，图像的角落里有一棵树也不应该影响我们的学习。

为了将上面的这些想法落实到具体的数学模型设计上，我们可以计算一个pixel关于它的最近的邻居们的加权和，而不是计算这个pixel和所有的其它的pixel的加权和。这等价于构造weight matices，对于每个output feature和每个output pixel location都构造一个weight matrix。这样的操作之后仍然是一个weighted sum，也就是说还是一个linear operation。

### 8.2 What convolutions do

我们之前就指明了希望我们的新的operation能做到：这些局部的特征能够不管objects位于image的哪个位置都能有同样的作用，也就是说，要有translation invariant的性质。对于我们在chapter7里的linear model里的weight matrix，我们如果想让它有这样的性质，那么我们需要使得weight具有一些多余的限制：对于每个pixel，距离太远的pixel对应的weight matrix位置的值就是0。对于剩下的不为0的weight matrix，我们需要使得每个位置的weight matrix的更新保持同步，而且初始值也是同步的。也就是说，相当于对于input每个位置的pixel都用一个相同的weight matrix来进行计算，这样就可以做到translation invariant。

而其实已经有了一个现成的作用在image上的local，translation invariant的linear operation：convolution。

convolution，或者更加精确的描述，discrete convolution，定义为一个关于2维image的每个neighborhood范围和weight matrix之间的scaler product，这个weight matrix称为kernel。考虑一个大小为$$3 \times 3$$的kernel（在deep learning里，我们最常用的是size比较小的kernel，我们可以在后面看到为什么），将这个kernel看作一个2维的tensor：

```python
weight = torch.tensor([[w00, w01, w02],
                       [w10, w11, w12],
                       [w20, w21, w22]])
```

我们还有一个1channel的$$M \times N$$的image：

```python
image = torch.tensor([[i00, i01, i02, i03, ..., i0N],
                      [i10, i11, i12, i13, ..., i1N],
                      ...
                      [iM0, iM1, iM2, iM3, ..., iMN]])
```

而它们两之间计算的convolution的output里的一个元素就是

$$ o11 = i11 * w00 + i12 * w01 + i13 * w02 + i21 * w10 + i22 * w11 + i23 * w12 + i31 * w20 + i32 * w21 + i33 * w22 $$

figure1显示了上述计算过程

![convolution]({{ '/assets/images/DLP-8-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 1 Convolution: locality and translation invariance.*

convolution计算的过程就是，我们将kernel摆在image的pixel的某个位置上，然后计算kernel和对应的pixel位置的乘积的和，也就是利用kernel里的weights来计算image这个位置的一个加权和，这是output的一个element。之后将kernel移动位置，再利用同样的kernel来计算该位置的加权和，这是output的另一个element，以此类推，直到kernel走遍了image的所有位置。

对于多channel的image，就比如RGB图片，上述的weight matrix，也就是kernel的size将会变为$$ 3 \times 3 \times 3 $$，也就是说kernel的channel要和image的channel一样，而这几个channel都计算加权和，再将它们加起来作为输出。

注意到，正如nn.Linear里的weight一样，kernel里的weights是不知道的，是要通过learning process来学习的，它们可以先被随机赋值，之后再利用backpropagation来更新。同时注意到，每个kernel都要滑过整个image，所以在滑过的时候计算的加权和使用的都是同一个kernel，也当然权重是一样的。对于autograd来说，让loss对kernel的weight求导实际上要包括整张图片的所有信息。

现在可以看到我们之前所说的的观点的正确性：一个convolution等价于有多个linear operation，其中它们的weights只有一部分不是0，它们被预设为一样的值，而且它们在training过程中有着同样的更新值。

总结来说，通过将之前的nn.Linear调整为convolution，我们获得了：
* 对于neighborhood的局部计算
* translation invariance
* 具有少了很多parameters的模型

对于上述第三点，我们注意到，模型的参数不再取决于输入图片的大小，而是在于kernel的大小以及kernel的个数，即kernel size和output channel。


### 8.2 Convolutions in action

我们现在对convolution已经有了基本的了解了，下面我们来看看在PyTorch里将如何实现它们。torch.nn module提供了对于1维，2维，3维输入的input的convolution函数：nn.Conv1d，nn.Conv2d以及nn.Conv3d，分别适合time series，image和videos。

对于我们的CIFAR-10数据，我们将会使用nn.Conv2d function。我们至少要给nn.Conv2d function提供number of input features（或者叫channel），number of output features，以及kernel的size，这三个arguments。比如，我们想要设计一个对于RGB输入的convolution层，假设输出的channel是16。注意，output的channel越多，这个模型的学习能力就越强。我们需要不同的output channels负责检测不同类型的features。而且，因为我们这些kernel的参数都是随机起始设定的，所以即使在training完成后，有很多features其实对于任务来说是没有用的。我们的kernel size设为$$ 3 \times 3$$。

将kernel size每个方向都设置为一样的数是很常见的，所以说PyTorch为此提供了简便的写法，对于2D image来说，kernel_size=3就直接代表kernel是一个$$ 3 \times 3$$的大小；而对于一个3D video，kernel_size=3就代表kernel是一个$$ 3 \times 3 \times 3$$的大小。对于之后再Part2里所见到的CT Scan数据，其是3D的数据，但是不同维度有着不同的resolution，所以说这个时候使用每个方向不一样的kernel size是比较合适的。

```python
# In [1]:
conv = nn.Conv2d(3, 16, kernel_size=3)    # 除了kernel_size=3这样简单的表达，我们也可以用完整的表述，kernel_size=(3, 3)
conv

# Out [1]:
Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1))
```

而在convolution里，我们的weight tensor的shape是什么呢？kernel_size的大小是$$ 3 \times 3$$，input_channel为3，output_channel为16，所以说weight的shape应该是$$ 16 \times 3 \times 3 \times 3$$，而bias的shape为16。bias的作用和在linear model里的一样，都是在计算完成后，加在结果上。所以在convolution里，在每个output channel里都对应了一个bias，所以一共是16个。

```python
# In [2]:
conv.weight,shape, conv.bias.shape

# Out [2]:
(torch.Size([16, 3, 3, 3]), torch.Size([16]))
```

我们可以看到，convolutions对于image来说是多么合适，我们可以用很少的参数，而且还能有translation invariant的性质。

一个2D convolution的forward pass会有2D images作为output，output每个位置的pixel都是input对应位置的neighbohood的加权和。在我们的例子里，kernel的weights和bias都是随机初始化的，所以说output并没有什么意义。在PyTorch里，按照老惯例，我们还需要加上第0维作为batch的维度，因为nn.Conv2d所期待的输入维度为$$ B \times C \times H \times W$$：

```python
# In [3]:
img, _ = cifar2[0]
output = conv(img.unsqueeze(0))
img.unsqueeze(0).shape, output.shape

# Out [3]:
(torch.Size([1, 3, 32, 32]), torch.Size([1, 16, 30, 30]))

# In [4]:
plt.imshow(output[0, 0].detach(), cmap='gray')
plt.show()
```

figure2显示了输出的output的一部分的样子（作为gray scale image来输出的）

![gray]({{ '/assets/images/DLP-8-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 2 Our bird after a random convolution treatement.*

我们发现，我们的output的后两个维度的size要比输入的input的后两个维度的size要小了一些，这是为什么呢？

#### 8.2.1 Padding the boundary

output image的长宽比input image的长宽要小的原因是由于kernel在input image的边缘计算时导致的。在input image的一个$$3 \times 3$$的neighborhood上进行一个将convolution kernel当作weight的convolution计算的一个前提是，这个区域能有$$3 \times 3$$大小的neighborhood，也就是说这个pixel的各个方向都能取到值。而在i00这个位置，它只有下方和右方有值。默认的话，convolution会让kernel每次滑动一个pixel，这样的话，output的size就应该是input的size - kernel沿这个方向的size + 1。

而PyTorch给我们提供了padding，来为boundary周边加上0，从而在convolution计算到边缘的时候，仍然可以有值计算。

```python
# In [4]:
conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
output = conv(img.unsqueeze(0))
img.unsqueeze(0).shape, output.shape

# Out [4]:
(torch.Size([1, 3, 32, 32]), torch.Size([1, 1, 32, 32]))
```

figure3显示了padding的过程。我们可以看到，上述操作使得输出的output的长宽和输入就一样了。注意到，weight和bias的大小并没有改变，padding并不会影响它们。

![padding]({{ '/assets/images/DLP-8-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 3 Zero padding to preserve the image size in the output.*

有两个主要的原因让我们做padding。首先，padding可以使得我们的convolution操作不改变input长宽的大小。其次，当我们有更复杂的convolution操作，比如skip convolution（在8.5.3里说）或者part2里要说的U-Net时，我们希望tensors在经过convolution计算之后和计算之前有着相对应的size，不要被convolution而改变size。


#### 8.2.2 Detecting features with convolutions

我们之前说weight和bias是我们需要利用backpropagation来学习的parameters，正如weight和bias在nn.Linear的情况一样。然而，我们也可以通过手动设置convolution里的weight和bias来看看结果会怎么样：

```python
# In [5]:
with torch.no_grad():
    conv.bias.zero_()

with torch.no_grad():
    conv.weight.fill_(1.0 / 9.0)       # 我们也可以直接用conv.weight.one_()，除了output的值是这个的9倍，其余的都是一样的
```

```python
# In [6]:
output = conv(img.unsqueeze(0))
plt.imshow(output[0, 0].detach(), cmap='gray)
plt.show()
```

figure4显示了上述结果。正如我们所预测的一样，这个kernel的设置，使得output输出了input image模糊化的版本。毕竟，output每个pixel都是input该位置的pixel的neighborhood平均得来的，所以output里的pixel更加corelated，并且过度更加顺滑。

![blur]({{ '/assets/images/DLP-8-4.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 4 Our bird, this time blurred thanks to a constant convolution kernel.*

接下来，让我们再试一些其它的：

```python
# In [7]:
conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)

with torch.no_grad():
    conv.weight[:] = torch.tensor([[-1.0, 0.0, 1.0],
                                   [-1.0, 0.0, 1.0],
                                   [-1.0, 0.0, 1.0]])
    conv.bias.zero_()
```

上述的kernel实际上是一个edge detection kernel：对于vertical的edge，它的对应的值就很大，对于constant的区域，值就是0。严格来说，这应该是是一个vertical edge detection kernel。

将上述kernel运用在我们的image里，我们会得到figure5所示的结果。正如我们所预料的，这个kernel增强了vertical edges。

![edge]({{ '/assets/images/DLP-8-5.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 5 Vertical edges throughout our bird, courtesy of a handcrafted convolution kernel.*

我们还可以设计其他类型的kernels来detect各种features，detect的意思是output会在input有特定的features的位置有很大的值。实际上，在传统的cv任务里，计算机视觉学家就是精心设计各种filters，使得输出在输入的特定的features上能有很大的值。

使用deep learning，我们利用minimizing loss来使得模型自己来学习这些kernelparameters，比如使得模型能够minimizing输出和ground truth之间的cross entropy loss，如7.2.5里所说的那样。从这个角度来说，convolutional neural network的作用就是估计一系列排成layers的kernel的parameters，它们会将一个multichannel的输入的转换为新的multichannel 'image'，下一层再进行这样的操作，而每一层以及每一个channel可能对应着不同的操作，比如average，比如vertical edge detection等等。figure6展示了convolutional neural networks的training是如何学习到kernel的parameters的。

![cnn]({{ '/assets/images/DLP-8-6.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 6 The process of learning with convolutions by estimating the gradient at the kernel weights and updating them individually in order to optimize for the loss.*


#### 8.2.3 Looking further with depth and pooling

我们从fully connected layers改成了现在的convolution，我们获得了获取locality feature的能力，以及我们的模型具有translation invariance的性质了。但是还有一个显而易见的问题：我们使用的是$$ 3 \times 3$$或者$$5 \times 5$$的kernel，从而每次只能检测input image里这么大小的区域的feature，而如果有的feature横跨的区域比这个大呢？我们需要让我们的模型具有更广阔的scope。

一个解决的方法是使用kernel size更大的kernel。但是使用大的kernel又会使得参数增加，而且损失掉translation invariance的性质，对于我们的例子，最大的kernel size是32，那么就回到了fully connected layer模型。另一种解决方法是，将一个convolution的输出进行downsampling之后，作为下一个convolution的输入，也就是将convolution layer堆叠起来。

**FROM LARGE TO SMALL: DOWNSAMPLING**

downsampling原则上可以有很多种方式。将一个image按比例缩小一半等价于对于每四个neighboring pixels，采用某种计算方法只输出一个pixel。而如何计算则是我们可以选择的。有以下几种方法：
* average the four pixels。这种average pooling的方法在早期很流行，但后来逐渐被淡忘了
* take the maximum of the four pixels。这个方法叫max pooling，是目前最流行的方式，但是缺点是其它三个pixel的信息被完全遗弃了
* performan a stride convolution, where only every Nth pixel is calculated。

我们将会着重使用maxpooling，如figure7所示。

![maxpooling]({{ '/assets/images/DLP-8-7.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 7 Max pooling in detail.*

从直觉上来看，convolution layer输出的output image，特别是再经过activation function之后再输出，会对于这个kernel所检测到的feature存在的区域的值很高。通过加上了maxpooling，我们在牺牲掉一些值相对较小的output，使得我们所学习到的特征都是很明显的。

max pooling可以通过nn.MaxPool2d module来提供（如同convolution一样，它也有1D，2D和3D版本）。如果我们希望downsampling输入图片使得只有一半大，那么size选择为2。

```python
# In [1]:
pool = nn.MaxPool2d(2)
output = pool(img.unsqueeze(0))

img.unsqueeze(0).shape, output.shape

# Out [1]:
(torch.Size([1, 3, 32, 32]), torch.Size([1, 3, 16, 16]))
```

**COMBINING CONVOLUTIONS AND DOWNSAMPLING FOR GREAT GOOD**

让我们来看看，将convolution和downsampling结合起来是如何让模型能够看到更大尺度的结构的。如figure8所示，一开始，我们将一个$$ 3 \times 3$$的kernel运用在一个$$8 \times 8$$的image，我们获得了一个同样大小的output image。之后，我们使用max pooling来downsampling这个image，获得了一个大小为$$4 \times 4$$的output image，我们再在这个output image上运用$$3 \times 3$$的kernel，我们可以看到，因为这个$$4 \times 4$$的output image是由$$8 \times 8$$的image downsample得来的，所以我们在这个$$4 \times 4$$的image上做convolution，反传到之前的$$8 \times 8$$的原输入image上，就相当于在$$8 \times 8$$的区域做convolution，从而感受野变大了。而且，第二层的kernels是在第一层输出的结果上再学习features，第一层的结果可能是average，edge等等，所以第二层所学习到的features会更加high level一点。

![twolayers]({{ '/assets/images/DLP-8-8.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 8 More convolutions by hand, showing the effect of stacking convolutions and downsampling: a large cross is highlighted using two small, cross-shaped kernels and max pooling.*

所以说，一方面，第一层的kernels作用在初始的，low-level features的neighborhood上，而第二层的kernels作用在对于原始输入来说更广阔的neighborhood上，在前一层的features的基础上学习更high level的features。这样的作法为我们的convolutional neural network提供了学习很复杂的图片内容的能力，远远比CIFAR-10的图片要复杂。


#### 8.2.4 Putting it together for our network

现在我们手里已经有了足够的组件了，现在我们可以构建能够识别bird和airplane图片的convolutional neural network了。

```python
# In [1]:
model = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1),
                      nn.Tanh(), 
                      nn.MaxPool2d(2), 
                      nn.Conv2d(16, 8, kernel_size=3, padding=1), 
                      nn.Tanh(), 
                      nn. MaxPool(2),
                      # ...
                      )
```

第一层卷积将RGB三通道的输入image转换为16通道的，从而给了模型生成16种low-level的能够辨别bird和airplane图片的特征的机会。第一层输出的16通道$$32 \times 32$$的image在经过tanh函数之后，再利用maxpooling将其降采样到$$16 \times 16$$的大小。之后，第二层convolution又生成了8通道的$$16 \times 16$$的image。如果顺利的话，这8个通道的features将会是之前low-level的features综合之后的学习结果，将会给出更high-level的features。同样的，我们再利用tanh函数和maxpooling来得到8通道$$8 \times 8$$的image。

这样的过程该如何结束呢？在得到了$$8 \times 8 \times 8$$的image之后，我们认为它已经可以被转换为输出probability，之后再使用NLLLoss了。然而我们如何将这个multichannel的2D image转换为一个1D的表示probability的tensor呢？我们可以直接使用fully connected layer!

```python
# In [1]:
model = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1),
                      nn.Tanh(),
                      nn.MaxPool2d(2),
                      nn.Conv2d(16, 8, kernel_size=3, padding=1),
                      nn.Tanh(),
                      nn.MaxPool2d(2),
                      # ... 我们这里省略了很重要的东西
                      nn.Linear(8*8*8, 32)
                      nn.Tanh(),
                      nn.Linear(32, 2)
                      )
```

上述代码可以得到figure9所示的网络：

![net]({{ '/assets/images/DLP-8-9.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 9 Shape of a typical convolutional network, including the one we are building. An image is fed to a series of convolutions and max pooling modules and then straightforward into a 1D vector and fed into fully connected modules.*

先不管我们上面的代码里省略的重要的东西。我们看到，我们后面添加的linear层的大小取决于我们之前convolution以及maxpool层输出的image的大小，8 * 8 * 8 = 512。我们来看看这个模型的参数有多少：

```python
# In [2]:
numel_list = [p.numel() for p in model.parameters()]
sum(numel_list), numel_list

# Out [2]:
(18090, [432, 16, 1152, 8, 16384, 32, 64, 2])
```

这样的参数数量对于我们这样一个很小的问题以及很小的数据集是很合适的。我们可以通过增加我们的convolution层的输出channel来增加模型的学习能力。

我们现在来看我们上面代码里省略掉的很重要的部分。其实就是我们需要将一个3维的tensor转换为一个1维的tensor，以适应于之后的nn.Linear层的输入，但是再nn.Sequential里我们不能用view method来直接将tensor变形，实际上在nn.Sequential里，我们不能以任何显式的形式来看任何一个PyTorch module的输出。


### 8.3 Subclassing nn.Module

在我们学习neural networks的某些时候，我们发现我们需要一些已经构造好的PyTorch module里没有涵盖的计算。在我们上面的情况里，我们就需要reshaping的操作使得convolution的结果能够和linear module连接上；在8.5.3里，我们还会举个例子，那时候我们需要实现residual connections。所以在这一章里，我们将会学习如何设计自己的nn.Module subclass，这样我们就可以像使用已经设计好的那些PyTorch modules如nn.Conv2d那样或者像nn.Sequantial那样使用它们。

当我们想要构造一个更加复杂的network，除了将layers堆叠起来以外，我们还想做一些其他的事情，这个时候nn.Sequential就不行了，我们可以通过subclassing nn.Module来实现。

为了能够subclass nn.Module，最少我们需要定义一个forward function，它会摄取input，并且输出output。这就是我们可以自定义我们所需要的computation的地方。这个forward function实际上是一些之前的东西的遗留产物，我们在5.5.1里所见到的模型，需要forward和backward function来使得模型能够学习。而在PyTorch里，如果我们使用标准的torch operations，autograd会自动帮我们实现backward的操作，所以实际上任何nn.Module都不会定义backward function。

而我们的PyTorch module有时候还需要用到其它的已经定义好的PyTorch modules，为了能够在我们的module里使用这些modules，我们需要在__init__里定义他们，并将它们的值传给self，这样才能在forward里使用它们。而它们在被定义并赋值了之后，将会我们自定义的module失效之前都能保持它们内部的parameters。注意，我们需要在__init__里先调用super().__init__()来继承父类的method，这样我们才可以正确使用。

### 8.3.1 Our network as an nn.Module

我们利用subclass nn.Module来定义我们的network。我们需要在__init__里instantiate所需要的nn.Conv2d，nn.Linear，以及一系列我们在nn.Sequential里用到的PyTorch modules，然后在forward里用它们：

```python
# In [1]:
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(32, 2)
    
    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        out = out.view(-1, 8 * 8 * 8)
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out
```

上面定义的Net class和之前利用submodules作为输入的nn.Sequential定义的模型是等价的。但是通过将forward function直接写出来，我们可以操作每个环节的输出，我们将maxpool2的输出的size变为了$$B \times N$$从而能够适应后面的nn.Linear。注意到我们一直都是保持第0维是batch。

上面我们利用subclassing nn.Module的方式来获取整个模型，如figure10所示。

![model]({{ '/assets/images/DLP-8-10.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 10 Our baseline convolutional network architecture.*

注意到我们classification network的目的就是将信息进行压缩，也就是输入是一个多通道的包含了很多pixel信息的image，而输出就是一个1维的size为2的tensor来表示分类的probability。对于这个目的，这个模型结构有两方面需要注意。

首先，我们的目的是通过我们中间层的大小逐渐变小来实现的：convolution层的channel逐渐变小，pooling使得宽度长度逐渐变小，linear层的output feature number要比input feature number小。这对于一个classification network来说是很典型的也是很普遍的。然而在很多流行的architecture里，比如说Resnet，上面所说的information reduction是通过pooling来使得resolution减小来实现的，而channel数量实际上是随着网络增加而加深的（总体来说信息还是被压缩的）。我们上面代码里的信息的迅速减少，对于我们这个简单的任务，简单的数据集，以及简单的网络来说是可以的，但对于更复杂的情况，我们的信息压缩要减缓很多。

第二点，在我们的模型里，第一层convolution层，我们的输出的信息量实际上比输入要大，而这也是在模型设计里很常见的，第一层或者前几层会使得输出的images比输入的还要大，之后再逐渐压缩。


#### 8.3.2 How PyTorch keeps track of parameters and submodules

有意思的是，将一个nn.Module的instance作为另一个nn.Module的attribute的话，正如我们前面代码里在__init__里做的那样，会自动将前一个nn.Module当作后一个nn.Module的submodule。

> 需要注意的是，submodules只能通过top-level attributes来实现，位于list或者dict instances里面是无法实现的，这种情况下，optimizer将无法定位submodule的位置，从而也无法定位它们的parameters的位置。在你的模型需要一个list或者一个dict的submodules的情况下，PyTorch提供了nn.ModuleList和nn.ModuleDict这两个class。

我们可以给nn.Module subclass自定义任何method，比如说，对于一个training过程和正式使用它的过程完全不一样的model，比如说prediction，我们就可以自定义一个predict method。我们需要注意，直接调用我们自定义的这些method的过程，就仿佛像我们直接调用forward而不是通过把model本身当作一个function来调用这个过程一样（model本身是有__call__ method的，因为其是从nn.Module类继承来的），会忽略掉一些hooks，而且JIT将不会看到模型的结构。

上述性质让我们可以直接获取一个nn.Module内部的submodules的parameters：

```python
# In [2]:
model = Net()

numel_list = [p.numel() for p in model.parameters()]
sum(numel_list), numel_list

# Out [2]:
(18090, [432, 16, 1152, 8, 16384, 32, 64, 2])
```

上述代码中发生的是，parameters() method的调用直接查到了所有作为attributes注册的submodules，然后再对这些submodules递归的使用parameters() method。不管这些submodules内部嵌套了多少层（比如一个nn.Module内有一个nn.Module，而内部的这个nn.Module内还有一个nn.Module），所有的nn.Module可以获取到它内部的所有parameters。通过访问它们的.grad attribute，这个.grad的值是autograd给的，optimizer将会知道如何改变parameters来缩小loss。我们在chapter5里已经知道了这个流程。

我们现在知道了如何实现我们自己的PyTorch modules，我们在part2里将会大量使用这个方法。回看Net class的实现，然后思考我们在__init__里将那些要用到的PyTorch modules作为Net的submodules注册给self的意义，我们会发现，那些没有可学习参数的PyTorch modules，比如nn.Tanh()，nn.MaxPool2d()并不需要在__init__里被赋值给self，那么我们是否有更加简便的使用它们的方法，就如同在forward method里使用view() method一样方便呢？

### 8.3.3 The functional API

对于上面的问题，答案当然是可以的。这就是为什么PyTorch对于每个nn module里定义的PyTorch module都有其functional的counterpart。通过叫它functional，我们表明这个object并没有内部状态，也就是说，它的输出完全由输入来决定，而没有可调整的参数。torch.nn.functional提供了很多functions，它们和nn module里提供的PyTorch modules用起来差不多。但是nn module里的PyTorch module是将输入作为argument，而将parameters存储在modules内作为参数，而nn.functional里的functions是将输入和parameters一起都当作arguments，直接输出一个结果，把整个过程看作一个function call。比如说，nn.Linear的counterpart就是nn.functional.linear，这个function的signature是linear(input, weight, bias=None)，weight和bias和input一样都是这个function的arguments。

回到我们的模型，对于nn.Linear和nn.Conv2d来说，我们需要使用nn module里的PyTorch modules，因为这样可以在training的过程中让我们定义的Net能获得它们的parameters。但我们可以很安全的将nn module里的nn.Tanh()和nn.MaxPool2d换成nn.functionals里的counterpart，因为它们并没有可学习参数：

```python
# In [1]:
import torch.nn.functionals as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8*8*8, 32)
        self.fc2 = nn.Linear(32, 2)
    
    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8*8*8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out
```

>一些general use的functions比如tanh，在nn.functional里有，但是更好的方法是直接用最上层nn来使用它。而更小众的max_pool2d则在nn.functional里

因此，上述的functional形式的模型的实现更加阐述了nn.Module API的内容：一个PyTorch module是一个container，里面装着在training过程中需要被更新的parameters的states，也装着内层的submodules。

至于是选择使用functionals还是使用PyTorch modules，这取决于各人的口味。如果模型过于简单，以至于我们使用nn.Sequential来设计，那我们就是在使用PyTorch modules。当我们自己写forward method的时候，对于那些不需要parameters更新的部分，使用functionals会显得更加简洁明确。

在chapter15里，我们将会简要的讲述quantization。这时并没有state的bits比如说activations瞬间变得有state了，因为我们需要获得它的information来quantization。这表明如果我们想要quantize我们的模型，那我们就需要使用PyTorch module的方式来定义模型。

还有个需要注意的点，如果你要多次使用一些没有state的PyTorch modules，比如nn.HardTanh或者nn.Tanh，我们最好在每个地方都使用其不一样的instance。使用同一个instance可能会在分析模型的时候造成问题。

所以现在我们学会了如何自定义我们自己的nn.Module，而且我们也知道nn.functional也可以使用。


### 8.4 Training our convnet

我们现在可以将所有内容整合到training loop里了。我们已经在chapter5里设计了整个流程，而我们这里将会重新回顾一次，并加上一些细节。在我们运行成功之后，我们还有兴趣来看看如何在GPU上运行从而使得我们的模型更快。但是首先，我们来看看training loop。

回想一下，我们的convnet的核心是两层loops，外层loop是epochs loop，内层loop是DataLoader从Dataset生成batch的loop。在每个内层loop里：
* 计算forward pass，也就是将inputs喂给model来计算结果
* 计算loss（也是forward pass的一部分）
* 将过去的gradient清零
* 调用loss.backward()来计算loss关于所有parameters的gradients
* optimizer.step()来更新参数从而达到更小的loss

```python
# In [1]:
import datetime           # 使用Python自带的datatime module来计算时间

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):           # 从1开始，而不是0
        loss_train = 0.0
        for imgs, labels in train_loader:          # train_loader给我们生成了一个batch一个batch排列的数据，我们loop over这些batch
            outputs = model(imgs)                  # 给模型喂一个batch的数据
            loss = loss_fn(outputs, labels)        # 计算我们需要minimize的loss
            optimizer.zero_grad()                  # 给gradient清零，去除上一轮的值
            loss.backward()                        # 计算loss关于所有我们想要更新的parameters的gradients
            optimizer.step()                       # 更新模型参数
            loss_train += loss.item()              # 计算这个epoch的loss的总和，注意我们需要利用.item() method将loss转换为Python number，这样就没有gradient信息了
        
        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train / len(train_loader)  
            # loss除以了train_loader的长度，因为上面我们将一整个epoch的loss都累积起来了，这样的话我们就可以得到一个batch平均的loss的值
```

我们使用chapter7里的torch.utils.data.datasets.Dataset，将其包装为DataLoader，instantiate我们的model、optimizer以及loss。然后我们就可以运行我们上面的training loop。

我们这里改变最大的就是我们现在的模型是nn.Module的subclass。下面我们来运行看看。

```python
# In [2]:
train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)

model = Net()
optimizer = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

training_loop(
    n_epochs = 100,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader,
    )

# Out [2]:
2020-01-16 23:07:21.889707 Epoch 1, Training loss 0.5634813266954605
2020-01-16 23:07:37.560610 Epoch 10, Training loss 0.3277610331109375
2020-01-16 23:07:54.966180 Epoch 20, Training loss 0.3035225479086493
2020-01-16 23:08:12.361597 Epoch 30, Training loss 0.28249378549824855
2020-01-16 23:08:29.769820 Epoch 40, Training loss 0.2611226033253275
2020-01-16 23:08:47.185401 Epoch 50, Training loss 0.24105800626574048
2020-01-16 23:09:04.644522 Epoch 60, Training loss 0.21997178820477928
2020-01-16 23:09:22.079625 Epoch 70, Training loss 0.20370126601047578
2020-01-16 23:09:39.593780 Epoch 80, Training loss 0.18939699422401987
2020-01-16 23:09:57.111441 Epoch 90, Training loss 0.17283396527266046
2020-01-16 23:10:14.632351 Epoch 100, Training loss 0.1614033816868712
```

#### 8.4.1 Measuring accuracy

为了有一个比loss更加直观的对模型效果的衡量，我们可以看看模型对training set和validation set的accuracy是多少：

```python
# In [3]:
train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=False)
val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False)

def Validate(model, train_loader, val_loader):
    for name, loader in [("train": train_loader), ("val": val_loader)]:
        correct = 0
        total = 0
        
        with torch.no_grad():            # 我们不需要gradients，因为我们不需要更新参数
            for imgs, labels in loader:
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum()) #我们直接使用int将一个integer tensor转变为一个数值，这等价于使用.item() method（在training loop里是这么用的）
        
        print("Accuracy {}: {: .2f}".format(name, correct / total))

Validate(model, train_loader, val_loader)

# Out [3]:
Accuracy train: 0.93
Accuracy val: 0.89
```

我们可以看到，结果比fully connected neural network要好得多，而且我们模型的参数还要少得多。这表明convolutional neural network确实是要比fully connected neural network更适合这个问题。


#### 8.4.2 Saving and loading our model

既然我们现在对模型很满意了，那么我们就要想办法把它存下来。这很容易：

```python
# In [4]:
torch.save(model.state_dict(), data_path + 'birds_vs_ariplanes.pt')
```

现在bird_vs_airplanes.pt这个文件就存有model的所有parameters：也就是两个convolution PyTorch modules和两个linear PyTorch modules的weights和bias。也就是说，我们没有存下结构。只是存了数值。这表明，当我们想要用这个模型的时候，我们还需要另外instantiate一个这个模型，然后将数值加载进去：

```python
# In [5]:
loaded_model = Net()
loaded_model.load_state_dict(torch.load(data_path + 'bird_vs_airplanes.pt'))

# Out [5]:
<All keys matched successfully>
```

#### 8.4.3 Training on the GPU

我们现在定义了一个模型，并且我们已经可以训练它了。但是如果能使得训练更快一点岂不是更好。我们将这些计算转移到GPU上则会更快。使用.to method，我们在chapter3里介绍了这个method，我们就可以将从DataLoader里获取的tensor转移到GPU上，之后我们的计算就会自动在GPU上进行。当然，我们还需要将我们的parameters也转移到GPU上，要不然不同设备上的tensor是不能相互计算的。好在，nn.Module本身就自带.to function，其可以将所有的parameters都转移到GPU上。

而nn.Module.to和nn.Tensor.to有着一些区别：nn.Module.to是in place的，也就是说这个PyTorch module本身就被改变了，其本身就转移到了GPU上。但是nn.Tensor.to是out place的，会返回一个新的放在GPU上的tensor。

一个好的习惯是如果GPU可用的话就将计算转移到GPU上，我们可以将device以torch.cuda.is_available来进行设置：

```python
# In [1]:
device = (torch.device('cuda') if torch.cuda_is_available() else torch.device('cpu'))
print(f"Training on device {device}.")
```

现在我们就可以调整之前的training loop，利用nn.Tensor.to method将从DataLoader里获取的tensor转移到GPU上：

```python
# In [2]:
import datetime

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs+1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item()
        
        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train / len(train_loader)))
```

对于Validation function，我们也可以同样进行更改。我们将model重新instantiate，放到GPU上，然后再重新运行我们的training loop，一定要注意，model必须也放到GPU上，要么报错：

```python
# In [3]:
train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)

model = Net().to(device=device)                   # model必须也在GPU上，目前PyTorch还不支持跨CPU和GPU平台的计算
optimizer = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

training_loop(
    n_epochs = 100,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader
    )

# Out [3]:
2020-01-16 23:10:35.563216 Epoch 1, Training loss 0.5717791349265227
2020-01-16 23:10:39.730262 Epoch 10, Training loss 0.3285350770137872
2020-01-16 23:10:45.906321 Epoch 20, Training loss 0.29493294959994637
2020-01-16 23:10:52.086905 Epoch 30, Training loss 0.26962305994550134
2020-01-16 23:10:56.551582 Epoch 40, Training loss 0.24709946277794564
2020-01-16 23:11:00.991432 Epoch 50, Training loss 0.22623272664892446
2020-01-16 23:11:05.421524 Epoch 60, Training loss 0.20996672821462534
2020-01-16 23:11:09.951312 Epoch 70, Training loss 0.1934866009719053
2020-01-16 23:11:14.499484 Epoch 80, Training loss 0.1799132404908253
2020-01-16 23:11:19.047609 Epoch 90, Training loss 0.16620008706761774
2020-01-16 23:11:23.590435 Epoch 100, Training loss 0.15667157247662544
```

即使是对于我们这么小的模型，我们仍然可以看到显著的速度提升。对于大的模型就更不用说了。

在加载存储的模型的参数的时候有一些需要注意的：PyTorch将会尝试将参数加载到这个参数所存储的设备的模型里，也就是，在GPU上存的参数将会被加载到在GPU上的模型里。而传入一个map_location的keyword argument就可以解决问题：

```python
# In [1]:
loaded_model = Net().to(device=device)
loaded_model.load_state_dict(torch.load(data_path + 'birds_vs_airplanes.pt', map_location = device)

# Out [1]:
<All keys matched successfully>
```


### Model Design

我们通过subclassing nn.Module来构造了一个模型，这个过程是除了那些很简单的模型的基本标准构造方法。之后我们成功的训练了它，并将其迁移到GPU上进行了训练。我们已经学会了如何构造一个feed forward convolutional neural network并且成功的训练它来分类image。现在，一个自然的问题是：如果我们遇到了更复杂的问题该怎么办？我们的CIFAR-10数据集里的image尺寸很小，而且图片中主要的object占据了图片的中央主要区域，这是个比较简单的问题。

如果我们将数据集换为ImageNet，我们会发现更大的，更复杂的images，而对于基于这个数据集的更复杂的问题，可能需要不止一条visual clues来得到结果。比如说，如果想让模型学习一个暗色的砖块形的东西是手机还是遥控器，那么模型可能会想要去学习是否有一个screen等。

而且images并不是我们在真实世界里可以获得的唯一的数据，我们还有tabular data，sequence，text等。具有合适结构和合适loss的neural network才能够对于这些各种各样的问题和各种各样的数据都有足够灵活的学习能力。

PyTorch提供了十分详尽的PyTorch modules，loss functions和其它的东西，来实现各种feed forward neural networks，LSTM，transformers等等网络结构。有些模型可以直接在PyTorch Hub或者torchvision里获取。

我们将会在part2里看到一些更复杂的网络结构，在那里，我们将会从头开始研究如何处理CT Scan数据，但是研究所有的neural networks的结构则超出了这本书的内容。但是我们已经学习了构建neural network的知识，这对于构建各种类型的neural networks都是有帮助的。这一节的目的是让我们学习到一些概念上的知识，从而我们可以在最新的research paper里看懂它们的neural networks结构，并将其用PyTorch实现。而有些作者会直接提供PyTorch版本的代码实现，我们可以直接拿来使用。


#### 8.5.1 Adding memory capacity: Width

对于我们之前已经定义好的feed forward architecture，在往下讨论之前，我们还有一些dimension需要来研究一下。第一个dimension是这个neural network的width：也就是每一层的neurons的个数，或者说每一层channels的个数。我们可以很轻易地在PyTorch里将一个neural network的width拓宽：

```python
# In [1]:
class NetWidth(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)
    
    forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 16*8*8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out
```

我们也可以将width作为__init__的一个参数来控制模型的定义:
```python
# In [1]:
class NetWidth(nn.Module):
    def __init__(self, n_chans1 = 32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_chans1, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)
    
    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 16 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out
```

neural networks每一层的channels和features的大小直接关系到parameters数量的多少，而parameters增多，则会增加模型的capacity。正如我们之前所做的，我们可以看看这个模型有多少个参数：

```python
# In [2]:
model = NetWidth()
numel_list = [p.numel() for p in model.parameters()]
sum(numel_list)

# Out [2]:
38386
```

模型的capacity越强，其学习复杂的数据和复杂任务的能力就越强；但是同时，也更加容易发生overfitting，因为模型可以用更多参数来记住training set，会更加容易。我们已经学习过一些缓解overfitting的方法，比如说增加数据，或者对现有的数据做data augmentation。

然而还有一些并不用改变数据，直接在模型上操作就可以增加模型不overfitting的能力。让我们下面来看看这些常用的方法。


#### 8.5.2 Helping our model to converge and generalize: Regularization

训练一个模型需要两个关键的步骤：1. optimization，我们需要loss在training set上逐步减小；2. generalization，模型不仅要在training set上效果好，也要在validation set上效果好。有一个数学工具能够同时增加这两个步骤的效果：regularization。

**KEEPING THE PARAMETERS IN CHECK: WEIGHT PENALTIES**

稳定generalization的第一个方法就是给loss加上一个regularization term。这个term被设计成模型的parameters（部分或全部）会被限制为一些比较小的值。换句话说，这是对于值很大的parameters的一个惩罚。这种做法会让loss曲线更加平滑，从而倾向于不会对于一些个别的点的拟合。

上述这种regularization term里最常见的就是L2 term，也就是所希望惩罚的那些parameters的平方和，以及L1 term，是所想要惩罚的那些parameters的绝对值的和。它们一般都会被一个比较小的常数乘起来，而这也是我们模型设计的一个hyperparameter。

L2 regularization也被称为weight decay。叫这个名字的原因是，考虑一下SGD和backpropagation，L2 regularization term关于w的负梯度，就是$$- 2 \times \lambda w$$，$$\lambda$$是scaled factor，而这个部分在PyTorch里就被叫做weight decay。所以说，使用L2 regularization term等价于对于parameters，在每一步更新的时候，减去当前值成比例的一小部分（正是因为如此，所以叫weight decay）。

在PyTorch里，我们可以通过直接在loss里加上term来做regularization。在每一步计算loss之后，不管loss function是什么，我们将此时模型的parameters的值的平方和（L2）或者绝对值（L1）求和，再乘以一个factor，加到loss里，之后再进行backpropagation：

```python
# In [1]:
def training_loop_l2reg(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs+1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            
            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())    # 将.pow(2.0)换成.abs()则是L1 regularization
            loss = loss + l2_norm * l2_lambda
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    if epoch == 1 or epoch % 10 == 0:
        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train / len(train_loader)))
```

然而，PyTorch里的SGD optimizer已经有了weight_decay这个argument，它会直接在更新的时候做weight decay，如上面所说的那样。


**NOT RELYING TOO MUCH ON A SINGLE INPUT: DROPOUT**

很早以前一个有效的控制overfitting的方法就被提出来了：dropout。它的原理很简单：在训练的时候，每次都随机的将某些neuron的输出直接置为0。

上述的这样的操作，在每次training的iteration的时候，实际上都使用了略微不同的模型，从而使得模型里的那些neuron在记住training set数据这件事上变得更难做到。另一个等价的角度是，dropout打乱了模型生成的特征，类似于data augmentation的操作，但这次是针对neural network结构的。

在PyTorch里，我们可以通过nn.Dropout这个PyTorch module来实现它，而其一般被使用在activation function和下一层的linear layer或者convolutional layer之间。我们需要给它一个argument来表明我们想要置零的这个操作出现的概率有多大。在convolutional neural network里，我们还可以用特殊的nn.Dropout2d或者nn.Dropout3d来做，它们每次会将某些channels直接置零：

```python
# In [1]:
class NetDropout(nn.Module):
    def __init__(self, n_chans1 = 32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv1_dropout = nn.Dropout(p=0.4)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3, padding=1)
        self.conv2_dropout = nn.Dropout2d(p=0.4)
        self.fc1 = nn.Linear(8 * 8 * n_chans1 // 2, 32)
        self.fc2 = nn.Linear(32, 2)
    
    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * self.n_chans1 // 2)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out
```

dropout一般只在training的过程里有用，而在validation的时候，就不需要它了，PyTorch设置模型的状态来改变它：model.train()和model.eval()，而任意一个nn.Module都具有这两个methods。

**KEEPING ACTIVATIONS IN CHECK: BATCH NORMALIZATION**

在dropout提出之后，Google提出了batch normalization，它可以让我们增加训练的速度，使得训练的结果不那么依赖初始化的参数值，而且还可以有regularization的作用，所以说它替代了dropout。

batch normalization背后最主要的思想是，在activation layer之前，将这个层的输入rescale一下，从而一个minibatch的这些输入，满足某种分布。回忆一下learning的机制，以及activation function的作用，这个操作可以使得activation function的输入不会过于远离activation function的saturated range，从而不会使得training变缓或者停滞。

在实际操作上，batch normalization使用改变mean和standard deviation的方法来改变中间层activation function的输入（这些操作都是以minibatch作为一个整体）。batch normalization的regularization效果是因为对于单个的sample来说，其结果被由一个minibatch计算来的mearn和standard deviation影响力，而这个minibatch的选取是随机的。

在PyTorch里，batch normalization由nn.BatchNorm1D，nn.BatchNorm2D，nn.BatchNorm3D这三个PyTorch modules来实现，取决于输入的dimension。因为batch normalization的目的是将activation function的输入进行scale和shift，所以它一般都放在convolution layer或linear layer之后，activation function之前：

```python
# In [1]:
class NetBatchNorm(nn.Module):
    def __init__(self, n_chans1 = 32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_chans1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3, padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_chans1 // 2)
        self.fc1 = nn.Linear(8 * 8 * n_chans1 // 2, 32)
        self.fc2 = nn.Linear(32, 2)
    
    def forward(self, x):
        out = self.conv1_batchnorm(self.conv1(x))
        out = F.nn_maxpool2d(torch.tanh(out), 2)
        out = self.conv2_batchnorm(self.conv2(out))
        out = F.nn_maxpool2d(torch.tanh(out), 2)
        out = out.view(-1, 8 * 8 * self.n_chans1 // 2)
        out = torch.tanh(self.fc1(out))
        out = self.fc1(out)
        return out
```

正如dropout，batch normalization在training和validation过程里是不一样的。实际上，在inference的时候，我们不希望一个input的output收到其它sample的input的影响。所以，在inference的时候我们仍然要normalization，但normalization的参数是固定下来的。

随着minibatch一直喂给模型，具有batch normalization的模型就会一直计算每个minibatch的mean和std，而且对于整个数据集的mean和std也记录着并一直更新，一旦我们使用model.eval()，那么记录下来的整个数据集的mean和std就停止更新，并直接用在normalization层里，作为inference时候使用的参数。一旦我们想要再继续更新整个数据集的mean和std了，就使用model.train()，那么就又恢复了。


#### 8.5.3 Going deeper to learn more complex structures: Depth

之前，我们说一个模型的width是我们将模型变得更大学习能力更强的第一个dimension。而第二个我们可以调节的dimension就显而易见了：depth。既然这是一本关于deep learning的书，那么depth毋庸置疑也是我们需要考虑的内容。毕竟，更深的模型会比浅的模型更好，真的是这样么？也不一定，取决于具体情况。深度增加了，模型拟合复杂函数的能力变强了。对于computer vision来说，一个浅层的模型能够学习到一个人在图像里的形状，而一个深层的模型能够识别这个人的身份。深度模型使得模型能够处理层结构的信息，使得模型能够按顺序层层递进学习知识。

还有另一种考虑depth的方法：增加网络的深度也就是增加网络能够处理input的operation sequence的长度。类比于算法来说，更长的指令可以描述的更加具体，可以以不同的层级来描述不同抽象程度的行为。而对于neural networks来说，那就是每层所学习到的features的抽象程度是不一样的。

**SKIP CONNECTIONS**

一开始设计deep neural networks的时候遇到了一些问题，更深的网络一般来说training更难以收敛。让我们来回忆一下backpropagation，然后想一想其在很深的neural network里是怎么工作的。loss function关于parameters的gradients，特别是那些前面的层的parameters，需要被乘以非常多的前面的导数（链式法则）。而这些乘上的值有的可能特别小，从而使得gradient很小，或者特别大，使得其它的值都被吞了。一般的情况就是，非常深的网络的前面层的那些parameters，loss function对于这些parameters的gradients容易vanish，从而这些层的parameters无法得到充分更新。

2015年的时候，residual networks（ResNet)被提出来，其是一个利用了简单的设计就使得很深的neural network每层都能得到充分的学习。从此开始，computer vision开始使用很多层的deep neural networks来解决各种问题。ResNet用的技巧很简单，就是使用一个shortcut将输入不通过某层直接和输出连接，如figure11所示。

![resnet]({{ '/assets/images/DLP-8-11.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 11 The architecture of our network with three convolutional layers. The skip connection is what differentiates NetRes from NetDepth.*

我们来看看ResNet里用到的shortcut connection的一个简单的例子，我们先重新设计一个普通的neural network：

```python
# In [1]:
class NetDepth(nn.Module):
    def __init__(self, n_chans1 = 32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(n_chans1 // 2, n_chans1 // 2, kernal_size=3, padding=1)
        self.fc1 = nn.Linear(4 * 4 * n_chans1 // 2, 32)
        self.fc2 = nn.Linear(32, 2)
    
    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = F.max_pool2d(torch.tanh(self.conv3(out)), 2)
        out = out.view(-1, 4 * 4 * self.n_chans1)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out
```

我们再来看一个简单的ResNet的结构:

```python
# In [1]:
class NetRes(nn.Module):
    def __init__(self, n_chans1 = 32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(n_chans1 // 2, n_chans1 // 2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(4 * 4 * n_chans1 // 2, 32)
        self.fc2 = nn.Linear(32, 2)
    
    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out1 = out
        out = F.max_pool2d(torch.tanh(self.conv3(out)) + out1, 2)
        out = out.view(-1, 4 * 4 * self.n_chans1 // 2)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out
```

在上述ResNet里所用的结构叫做identity map，所以它是怎么解决deep neural network gradient vanish的问题呢？

考虑一下backpropagation，我们可以认为这种skip connection给比较靠前的layers的parameters提供了一个更快的到达loss的路径，从而有了一条避免经过大量乘法的反传路径。

具有skip connection结构的deep neural network对比同样width，depth的同结构neural network，其loss function要更加平滑。


**BUILDING VERY DEEP MODELS IN PYTORCH**

如果我们想构建非常深的网络，方法可以是定义一个block，比如(Conv2d, ReLU, Conv2d) + skip connection，然后在一个for loop里定义这个neural network。我们将会构造一个figure12里描述的neural network。

![deep]({{ '/assets/images/DLP-8-12.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 12 Our deep architecture with residual connections. On the left, we define a simplistic residual block. We use it as a building block in our network, as shown on the right.*

我们先构造一个PyTorch module，代表一个block，包含了convolution，activation和skip connection：

```python
# In [1]:
class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3, padding=1, bias=False)   # batch normalization会抵消bias的作用，所以不用设置bias
        self.batch_norm = nn.Batchnorm2d(num_features = n_chans)
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')            # 我们使用.kaiming_normal_的初始化设置，这是ResNet那篇论文里的初始化设置
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)                            # 将batch_norm初始化为能够生成mean=0, std=0.5的output distribution的值
        torch.nn.init.zeros_(self.batch_norm.bias)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x
```

因为我们想要构建deep neural network，所以我们也加入了batch normalization来避免gradient vanish。我们现在用上面的block来构建一个100个block的neural network。

我们使用nn.Module来构建这个100个block的neural network。首先，在__init__里，我们建立一个nn.Sequential，包含一个list的ResBlock的instances。nn.Sequential可以保证上一个PyTorch module的输出可以作为下一个PyTorch module的输入。而且nn.Sequantial还能保证这个list里所有的PyTorch module的参数对于这个100个block的neural network来说都是可见的。之后，在forward method里，我们利用这个nn.Sequantial来让input通过100个block最终输出output。

```python
# In [1]:
class NetResDeep(nn.Module):
    def __init__(self, n_chans1 = 32, n_blocks = 10):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.resblocks = nn.Sequential(*(n_blocks * [ResBlock(n_chans=n_chans1)]))
        self.fc1 = nn.Linear(8 * 8 * n_chans1, 32)
        self.fc2 = nn.Linear(32, 2)
    
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 8 * 8 * self.n_chans1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out
```

我们上述的neural network会比浅层的更加难以收敛一点，对于初始化的值也更加脆弱一点，这都是按常理应该的。


**INITIALIZATION**

我们简要的讲一下initialization。initialization对于训练neural networks来说是一个很重要的trick。但是由于一些历史原因，PyTorch的很多默认的initialization都不是很理想，现在正在逐步改进中。同时，我们也可以通过自行定义初始化的值来改进这个过程。



#### 8.5.4 Comparing the designs from this section

我们在figure13里总结了我们这一节里所介绍的模型设计方法对于training和validation accuracy的影响。我们可以观察出一些性质：对于overfitting来说，dropout和weight decay要比batch norm有更加严格的statistical description，而且从结果也能看出来，batch norm的network的training accuracy都快到100%了。所以相对来说，我们会认为weight decay和dropout为regularization的方法，而batch norm则更多侧重于使得训练更加容易，以及使得loss收敛的更加容易和迅速。

![compare]({{ '/assets/images/DLP-8-13.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 13 The modified networks all perform similarly.*


#### 8.5.5
neural networks结构变化的非常快。我们介绍这些老的结构可以让我们对于理解PyTorch neural network结构有帮助。而后面的几个chapters可以帮助我们掌握如何将ideas用PyTorch code来写出来。


### Conclusion

我们最终设计了一个能够区分bird和airplane图片的neural network。但我们还有很多还没有做，比如说，如何从一张大的输入image里将含有bird或者airplane的部分裁剪下来作为输入。生成那样的box对于我们的网络来说是做不到的。

另一个问题是我们对于非bird和airplane的图片并没有提供处理办法。比如说，如果提供一张cat的图片，然后模型以很高的确信度判断为bird。这种对于离training distribution很远的samples仍然有很高确信度的输出的情况，叫做overgeneralization。这在我们的输入并不能有很高的确信度的时候是个常见的问题。

在这一章里，我们利用PyTorch构建了一些能够使用的不同结构的neural networks。我们首先利用我们的intuition来构建了convolutional neural network。之后我们又研究了width变化和depth变化的model，并且控制了overfitting。

现在我们已经有了PyTorch这样一个有力的工具，我们在Part2里的内容和Part1里介绍的方式不一样，Part1里是每章介绍若干个小问题，而Part2里则是利用很多chapters来将一个现实的大的问题分解为小部分，并解决。


### Summary

* convolution也是一种linear operation，其可以被用在feed forward network里。使用convolutions可以使得模型具有更少的parameters，而且能够检测locality特征，具有translation invariance的性质。
* 将多个convolution layers + activation layers堆叠起来，在中间再加上max pooling layers，就会得到逐层减小的feature images。所以后面层的convolution对于初始输入image的作用区域就会很大。
* 任何nn.Module的subclass可以递归的收集它内部submodules的parameters并将它们全部返回。这个特性可以用于optimizer里的参数更新。
* functional API提供了nn.Modules的counterpart，其并没有internal state。这可以用在并没有需要学习的parameters的计算里。
* 学习好的模型可以将其参数存下来，并且很轻松的再加载出来。






        
        


                

















































---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*
