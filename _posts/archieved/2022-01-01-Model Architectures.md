---
layout: post
comments: false
title: "Model Architectures"
date: 2021-11-29 01:09:00
tags: paper-reading
---

> This post is a summary of general machine learning model architectures.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---


## Convolutional neural network architectures

### 1. [AlexNet: ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
*ALex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton* 

*NIPS 2012*

AlexNet是深度学习浪潮的奠基作之一。

这篇文章里的模型赢得了ILSVRC 2012年的classification任务的冠军。

**1. title**

标题很简洁明了，用什么方法、模型（deep convolutional neural network）解决了什么问题（ImageNet classification）。

**2. authors**

Hinton大佬的文章，那肯定是值得看的。

**3. abstract**

我们设计了一个很大、很深的cnn来解决ImageNet LSVRC-2010的120万张图片分为1000个类的任务。在测试数据上，top1和top5的错误率都远远比其它的方法要好。

>这种写法其实是很少见的，第一句话说我做了什么，第二句话就直接展示效果并且说十分好。

这篇文章里的神经网络是很大的，有6千万个参数以及650000个神经元，而且有五个卷积层，三个全连接层，最后还有个长为1000的softmax。为了使得训练更快，我们设计了方法，使得神经元是位于non-saturated区域的，并且用了高效的GPU设计方法来训练。我们还用了dropout来使得全连接层不至于过拟合。我们最后将训练后的模型再应用到ILSVRC-2012的比赛里，达到了top-5比第二名低很多的错误率。

>这个摘要主要就说了我们用了什么方法做了什么事情，介绍了一下方法、模型具体的设计细节，之后就说了在实验上比其他方法远远要好。其实不太像是摘要，反而像是技术报告（technic report）的写作方式。但毕竟是大佬的文章，结果就是要比其他方法好的非常多。

**4. Introduction**

>一篇文章的第一段一般都是在讲一个故事，我们在做一个什么问题，这个问题为什么重要等。

我们要做object detection这个问题。为了提升它的性能，我们需要收集更大的数据集，学习更好的模型，以及使用技术来使得模型不要过拟合。

>用一些正则的方式使得模型不要过拟合在当时是个很重要的方向，但实际上现在有很多研究表明设计合理的网络结构要比使用正则方法要更加重要，这仍然是一个未解决的研究方向。

再之后就是介绍了各种数据集，因为文章的标题是解决ImageNet的数据集上的任务，所以最后引出ImageNet，吹一波ImageNet很好。

为了能够在上百万张照片里学习1000个种类，我们需要一个具有很大的learning capacity的模型。之后就引入了我们要用CNN来解决这个问题，但是CNN设计的很大并不容易，因为会overfitting而且可能会训练不动。

>这种写法是有问题的，因为当年的主流并不是CNN，而是其它的方法。所以这里一点都不提到其它的方法，而直接说解决这个问题就用CNN，其实是一个不客观，且比较窄的视角。

再之后说明CNN比较难训练，但对于这样比较大的数据集，确实可以训练一个比较大的CNN。而且有了GPU的帮助，可以使得训练高效很多。

接下来说了这篇论文的贡献。这篇论文在ImageNet上训练了一个最大的神经网络，并在ILSVRC-2010和ILSVRC-2012这两个任务上都取得了很好的结果。我们还实现了在GPU上高效的部署。而且我们提出的新的模型能够产生一些新的比较特殊的feature，能够使得效果很好而且减少训练的时间。而且我们做了一些防止过拟合的方法。最后说明网络的结构，有五个卷积层，三个全连接层，而且深度很重要：移除掉任何一层都不行。

>因为这篇文章提出了新的结构，有了新的不常见的feature来解决classification的问题，所以才让这篇文章有了很大的价值，成为了奠基作。而不是仅仅通过很多技术上的操作使得模型取得了第一名，但实际上并没有思想或者结构上的创新，只能让其他人觉得复杂，自叹不如，或者没有能力做如此复杂的工作，但并不具有启发性。而这篇文章提出了新的内容，相当于为后人挖了大坑。

最后介绍了一下用了GTX 580 GPU来训练。

**5. Dataset**

>因为这篇文章是在ImageNet这个数据集上解决其classification的任务，所以得介绍一下这个数据集。

数据集介绍的部分就不说了，主要介绍了数据集的内容，以及ILSVRC这个竞赛的内容。

（但这部分最后一段很重要）ImageNet实际上每张图片的分辨率是不一样的，也就是说图片是没有裁剪好的。这篇文章里的做法简单粗暴，直接将短边rescale到256，而长边等比例缩小之后，直接在rescale的图片中心区域裁剪出256$$\times$$256大小的部分。除了再将每个通道的平均值减掉这个操作以外，并没有任何别的pre-processing操作，整个模型是在raw-RGB图片上操作的。

>当时大多数的方法都需要从原始输入图片里抽取特征再用模型进行处理，比如说sift特征等等。但是这篇文章不需要抽取任何特征，实际上这是很大的一个进步，但这篇文章并没有把其作为一个卖点。这种方法实际上是end-to-end，也就是说原始数据直接输入模型，没有任何特征提取，结果直接输出。文章没有重点提到这个实际上是历史的局限性，之后的研究表明end-to-end是深度神经网络一个很大的卖点，应该要予以强调。

**6. Architecture**

**6.1 ReLU nonlinearity**

运用到神经网络里，饱和的那些非线性函数$$tanh(x)$$、$$sigmoid(x)$$等实际上要比非饱和的那些非线性函数比如说$$ReLU(x)$$使得训练慢很多。从fig 1就能看出，虚线是使用tanh的，实
是使用ReLU的，横轴是epoch数，纵轴是training error，可以看出来效果差别挺大。
  
![RELU]({{ '/assets/images/ALEXNET-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. 我们这里用CIFAR-10数据集，对于一个4层的CNN进行了测试，用的是tanh和ReLU两种不同的非线性函数。而对于每种情况，我们设置的learning rate不一样，learning rate设置成为
每种情况下，都选择使该种情况下训练最快的值。我们可以看到，ReLU使得训练快了很多。*

**6.2 Training on Multiple GPUs**

介绍的如何用多个GPU训练的，主要都是技术、工程上的细节。和ML、CV关系不大。

**6.3 Local Response Normalization**

实际上这个东西也不重要，后续也没有什么人用。
  
>提到了一点，ReLU不是饱和型的非线性函数，所以在输入ReLU之前实际上并不需要对输入做什么操作（比如说将输入集中到非饱和区域等）。

**6.4 Overlapping Pooling**

将传统的不overlap的pooling改成了overlap的pooling，会使得效果好很多。

**6.5 Overall Architecture**

用了五个卷积层，三个全连接层，再加上一个1000-way的softmax，如fig 2所示：
![Model Structure]({{ '/assets/images/ALEXNET-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. 模型的结构*
  
>输入是一个很扁的很宽的图片$$224 \times 224 \times 3$$，然后通过网络将高宽不断变小，而深度不断增加。随着网络的增加，慢慢将空间信息压缩，即高宽逐渐减小，到了最后一个卷积层
已经变成了$$13 \times 13$$，也就是说认为这里一个像素已经可以代表前面很大一块的像素。而随着深度的增加，通道数也在逐渐的增加，可以认为每个通道是在表示一种特定的特征，比如说19个通道，我可以认为它能够识别输入的图片的192种不同的特征，比如说某个通道用来识别边，某个通道用来识别圆等等。所以说，空间信息在逐渐压缩，而语义信息在逐渐增加（通道数就是语义信息）。而最后，到了全连接层。所以说，输入的$$224 \times 224 \times 3$$的图片，最后通过网络变成了一个$$4096$$长度的向量，最后再用一个线性分类器做分类。实际上，这个$$4096$$的向量具有很好的语义信息，所以效果很好，也可以从后面的fig 3看出。
>整个机器学习，都可以看成知识压缩的过程，原始的数据不管是图片、文字还是视频，最后都通过一个向量来表示，而这个向量所含有的语义信息，可以让机器进行识别。从而可以在上面做各种各样的事情。这也是神经网络精髓之所在。


**7. Reducing Overfitting**

**7.1 Data Augmentation**

输入是$$256 \times 256 \times 3$$大小的图片，而文章会随机裁剪$$224 \times 224 \times 3$$大小的部分作为输入，引入了随机性，增强了模型的generalization的能力。还对RGB通道做了PCA的操作。

**7.2 Dropout**

将很多个模型融合在一起找到效果最好的那个，这个方法是很常见的。但是对于deep neural networks来说，本来训练就很贵了，这么做是不现实的。这篇文章用了dropout的操作，也就是说随机的将模型里某些输出直接设置为0。这个操作可以使得我们每次都会得到不一样的模型，但这些模型的权重实际上是共享的，从而最后等价于很多个模型做融合。
  
>但后来的工作表明dropout实际上并不是在做模型融合，更像是正则项。之后的工作里说明dropout实际上等价于一个L2的正则项。

**8. Details of Training**

介绍了用SGD方法来训练模型。使用了weight decay，也使用了momentum。

还介绍了一下模型参数的初始化操作。

最后介绍了一下learning rate的设置方法。

**9. Results**

介绍了在ImageNet上的classification任务和其他方法的对比，有ILSVRC-2010和ILSVRC-2012两个比赛。

**9.1 Qualitative Evaluations**

我们从fig 3可以看到，模型的效果很不错。而从fig 3的右侧发现，feature向量相近的那些输入图片，内容确实是相近的，这其实是模型之所以效果好的很重要的点，这说明模型确实学习到了有效的语义信息，对于语义信息相似的图像，能生成距离相近的feature向量，从而在feature空间中能够很容易的分类。
![test]({{ '/assets/images/ALEXNET-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 3. 左图显示了在Iamgenet数据集里找的一些图片，以及它们通过模型输出的top-5的预测结果。而右图的第一列是随机选取的图片，对于每一行来说，第一列的图片输入模型后可以得到最后的4096长的feature向量，而我们寻找到那些图片，其feature向量与此向量的距离最近。*
  
>神经网络的可解释性一直是一个研究热点，而此处也提及到了一部分。

**10. conclusion**

>这篇文章是没有结论的，只有discussion。指明了未来可能还需要做些什么事情。但一篇文章的结论conclusion通常来说是和abstract的对应，所以说没有conclusion是很少见的，也是不推荐的。

我们这篇文章的结果表明深的、大的CNN对于很难的任务是效果很好的。如果我们的模型去掉一层，那么结果会变差，这说明深度也是有必要的。

>其实这里论证不太严谨，因为去掉一层结果会变差也可能是其它的参数没设置好，实际上去掉一层，再改改参数，还是能达到一样的效果的。但是歪打正着，这个结论是正确的，深度确实是很必要的。

我们这篇文章所使用的方法还没有用到任何的unsupervised pre-training的方法，并没有预先transform数据，而且我们所解决的是一个supervised learning的问题。

>实际上机器学习在长时间的范围内研究的都是unsupervised learning，Hinton还有LeCun等人一直都认为unsupervised learning才是主流方向。但这篇文章的出现，使得supervised learning火了起来，一直到最近Bert还有GAN的出现，才让unsupervised learning重新回到人们关注的视野里。实际上在这篇文章之前，neural network一直做的是unsupervised learning的工作，之所以不做supervised learning是因为比不过SVM，所以只能做一些SVM做不了的工作。但这篇文章的出现说明深的、大的neural network可以在supervised learning打赢其它方法了，所以引起了轰动和热潮。



### 2. [VGGNet: Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
*Karen Simonyan, Andrew Zisserman*

*ICLR 2015*

**1. Title**

和AlexNet、ResNet的标题一样的风格，都是直接说明什么模型（方法）解决什么问题。

**2. Authors**

Andrew Zisserman是Oxford VGG实验室的leader，很强的组。

**3. Abstract**

这篇工作我们探究了CNN的深度对于大规模image recognition任务精度的影响。我们的主要贡献是十分详尽的研究了网络深度的作用，而我们的网络使用的是很小的convolutional filters（$$3 \times 3$$），我们的结果显示深度到16-19层的时候，任务的精度能有很大的提升。这些结果是我们参加ImageNet Challenge 2014的模型的基础，我们的模型赢得了localisation的第一名，以及classification的第二名。我们同时也说明我们的方法对于其它的数据集仍然有很好的效果。我们将两个效果最好的CNN模型公布了，希望能够帮助之后在CVi领域使用深度视觉特征的研究。

**4. Introduction**

CNN最近在大规模的image和video recognition任务中获得了巨大的胜利，也是因为大规模的公开的数据集比如说ImageNet，以及高性能的计算资源比如说GPU，才使得这种成功成为可能。在ILSVRC-2014中，deep visual recognition architecture已经展露了头角，在前几年的比赛里，ILSVRC-2011的冠军使用的是high-dimensional shallow feature encodings，ILSVRC-2012年的冠军使用的是deep CNN（也就是AlexNet）。

随着CNN在CV领域逐渐变得常见，有很多人尝试改进AlexNet以获得更高的精度。比如说，ILSVRC-2013年的冠军就是将AlexNet改进为第一层使用更小的convolutional filter以及更小的stride。另一条改进AlexNet效果的路则是利用不同scale的输入image来训练。在我们这篇文章里，我们解决CNN结构的另一个重要的问题：深度。为了达到这个目的，我们将架构的其它参数固定，然后通过增加更多的convolutional layer来使得网络加深，这在训练上和计算上都是可行的，因为我们使用了$$ 3 \times 3$$的filters（$$ 3 \times 3$$的filters配合padding和stide=1可以使得输出的feature map和输入的feature map长宽不变，从而可以无限的加深下去）。

结果是，我们获得了一个精度更高的CNN模型，不仅在ILSVRC classification和localisation任务上取得了sota的效果，还可以用在其它的image recognition数据集上，仍然可以获得很好的效果，即使是很简单的设计（比如说将deep features直接利用SVM分类，不需要任何的fine-tuning）。我们公开了两个效果最好的模型为了将来的研究。

这篇文章剩下的部分结构如下。在section5里，我们描述了我们设计的CNN模型。image classification的训练和测试细节再section6里。我们的框架和其它人的模型的对比再section7里。section8总结整篇文章。


**5. ConvNet Configurations**

为了在一个公平的设定下研究增加CNN的深度带来的影响，我们所有的CNN层都使用同一个结构。在这一个section里，我们先描述我们的CNN模型的整体结构，然后再描述一些细节的设置。我们的设计细节最后再和之前的工作进行对比。


**5.1 Architecture**

在我们的训练过程中，我们的CNN输入是一个固定大小的$$224 \times 224$$的RGB image。我们做的唯一的pre=processing就是在训练集里，将所有的图片都减去它们的平均值（element
-wise）。这个image之后通过一系列的convolutional layer，在这些convolutional layer里，我们都使用的是$$3 \times 3$$的filters（这是能够获取一个pixel的上下左右信息的最小
filter size）。stride一直设定为1，而padding的设置使得每层convolutional layer的输入和输出的长宽是一样的，也就是说对于$$3 \times 3$$的filter，padding是1。而max-pooling在
某些convolutional layer后会出现，整个CNN一共有五个max-pooling，都是使用的$$2 \times 2$$的大小，stride是2。
  
在Convolutional layers之后，接了三个fully connected layers：前两个都是有4096个通道，而第三个是1000个通道，因为ILSVRC的分类是1000类，所以用这个作为输出，正好每个通道表示每个类。最后一层是一个soft-max layer，将第三个fully connected layer的输出转换为表示为每个类的概率。
  
对于所有的隐藏层，我们都用ReLU作为activation function。我们所有网络都没有用到local response normalisation（LRN）的技术。

**5.2 Configurations**

我们这篇文章里所使用的CNN的结构配置总结在Table1里，每列表示一个网络。再之后我们就使用A-E来表示这些网络。所有的网络设计都按照5.1里描述的那样，它们仅仅在深度上有区别：从11层（8层CNN，3层fc）到19层（16层CNN，3层fc）。convolutional layer的宽度也不大（也就是每个convolutional layer输出的channel数），从64开始，每次遇到max-pooling之后就增加一倍，直到到达512为止。
![TABLE1]({{ '/assets/images/VGG-1.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Table 1*
  
Table2总结了每个网络的参数的数量。尽管我们的网络深度很大，但我们网络的参数的数量并不比那些有着更大convolutional filters的更浅的网络要多。
![TABLE2]({{ '/assets/images/VGG-2.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Table 2*

**5.3 Discussion**

我们的CNN结构和在ILSVRC-2012和ILSVRC-2013竞赛里表现最好的那些模型都很不一样。他们的网络的第一层用stride=4的$$11 \times 11$$的filter，或者用stride=2的$$7 \times 7$$的filter，然而我们这篇文章里用的是很小的stride=1的$$3 \times 3$$的filter。很容易看出来，两个$$3 \times 3$$的filter堆叠起来，就等价于一个$$5 \times 5$$的filter。三个$$3 \times 3$$的filter堆叠起来，就等价于一个$$7 \times 7$$的filter的感受野。所以说，我们通过堆叠三层$$3 \times 3$$的filter而不是一层$$7 \times 7$$的filter，得到了什么？首先，我们的层与层之间还有非线性层，而$$7 \times 7$$的只有一个非线性层，这能使得我们的决策函数更加复杂。其次，我们减少了参数的数量：假设我们的$$3 \times 3$$的层的输入和输出的channel都是$$C$$，从而对于一共是27$$C^2$$个参数；而同时，一个$$7 \times 7$$的filter的参数是49$$C^2$$。

注意到我们还使用了$$1 \times 1$$大小的convolutional layer，这样的层可以在不影响其他层卷积感受野的情况下，增加模型的非线性程度，也可以改变通道数，因为这个层之后也是有activation function的。

之前也有人使用过小的convolutional filters，但是它们的网络并不深，而且没有在ILSVRC数据集上进行测试。Goodfellow关于识别街景图片里的数字的文章使用了深的CNN（11层）并且说明深的网络效果更好。GoogLeNet在我们这篇文章的同时也研究了小的convolutional filters和深的CNN。



### 3. [ResNet: Deep Residual Learning for Image Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
*Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun*

*CVPR 2016*

**AlexNet(2012) $$\rightarrow$$ ResNet(2016)**

**1. title**

deep residual learning是文章提出的方法，而image recognition则是要解决的问题，同样也是简洁明了的标题。

**2. Authors**

四个人都是大佬，这是在微软研究院所做的工作。

**3. Abstract**

训练深的神经网络非常不容易。我们使用深的残差网络使得训练比之前容易很多。我们将结构设计为，让每一层去学习相对于这一层输入的残差（因为结构设计为$$output = x + f(x)$$，从而$$f(x) = output - x$$即是残差）。我们通过大量的实验来证明我们提出的residual networks很容易训练，对于增加的深度来说也是一样。在ImageNet数据集上我们使用了152层深度的网络，比VGG深了8倍。最后我们在ILSVRC 2015的竞赛里达到了3.57%的test error而获得了冠军。我们还在CIFAR-10数据集上演示了如何训练100和1000层的网络。

对于很多视觉的任务来说，深度是很重要的。我们仅仅是将分类器用到的feature换成deep residual network所学习到的feature，就使得COCO object detection的准确率提升了很多。我们还赢得了ImageNet detectino， ImageNet localization，COCO detection和COCO segmentation比赛的冠军。

**4. Introduction**

深度CNN在image classification领域获得了很多突破。深度networks通过综合low/mid/high level的features，以end-to-end的方式实现。而这些不同level的features可以通过加深层数来使得它们非常的丰富。很多最近的工作表明网络的深度是很重要的，并且在ImageNet数据集上的任务做得好的网络都用了很深的结构，从16到30层不等。很多其他的视觉任务也从深度网络种获利。

但随之而来的出现了一个问题：学习一个好的深度网络就是简单的将层堆叠在一起就可以了吗？一个最显著的问题就是，在网络很深的时候，梯度容易爆炸或者消失，这就使得网络收敛变得很困难。而现有的方法就是在网络初始化和中间层normalization上做工作，这样能使得训练几十层的网络变得略微可行（比如说Batch normalization等），网络可以训练了，能够收敛了。

但虽然现在深的网络能够训练了，实际上网络的性能随着深度的增加而变差了，如fig 1所示。而这样一个现象并不是因为overfitting，这是因为此时训练误差也变大了（overfitting指的是训练误差小，而测试误差大）。

![Hard to Train]({{ '/assets/images/RESNET-1.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 1. 左侧是20层和56层普通的网络在CIFAR-10数据集上的training error，而右侧是test error。可以看到，深的网络的training error和test error都更大。*

>所以说用了一些方法之后，深的网络是收敛了，但是并没有训练到很好的结果，也就是属于欠拟合的状态。

上述的现象说明深的网络并不是那么好训练的。我们考虑一个浅的网络，以及它的深的改造（也就是多加一些层进去）。如果我们所添加的这些层所学习到的都是identity map，其它的层和浅的网络是一样的，那这个深的网络的效果是不会比浅的网络差的。但理论上是可以的，现实中却很难让深度网络学到这样的结果，是很难做到的。

>这篇文章提出的方法就是显式的构造一个结构，使得浅的网络的深的改造不会比浅层网络更差。

在这篇文章里，我们通过介绍deep residual learning framework来解决这个问题。我们并不是让堆叠的layers构成的网络来学习一个要学的mapping，而是显式的构造结构从而让每层学习residual mapping。对于我们要添加的层来说，输入为$$x$$，而我们希望能够学习到的mapping是$$H(x)$$，我们并不直接让网络来学习$$H(x)$$，而且让他来学习$$F(x)=H(x)-x$$，从而这样的话，我们希望学习到的mapping $$H(x)=F(x)+x$$。我们假设优化redisual mapping要比优化初始的mapping要容易。在最极端的情况，如果浅层网络就已经足够了，那实际上所添加的网络就是identity mapping，那这个时候residual mapping就是0。

>这里就是residual learning的核心的思想。见fig 2。

![Residual Block]({{ '/assets/images/RESNET-2.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 2. residual learning：一个模块.*

在模型设计里实现上述的residual结构，即$$F(x)+x$$可以通过shortcut来实现，如fig 2所示。

>但实际上shortcut的结构在很久以前就提出过了。

shortcut的结构很简单的就实现了我们的residual learning的结构，而且其就是简单的identity mapping，所以不增加参数也不增加计算复杂度，而且也不影响反向传播算法的使用。

我们在ImageNet上进行了实验证明了之前所说的过深的网络可能会造成的问题，并且也验证了我们提出的方法的有效性。1) 我们用了residual结构的很深的网络也能够很容易的被训练，而没有用residual结构的网络在网络加深的过程中训练error会增加。2) 我们的residual neural network可以享受由深度网络带来的性能上的提升，从而在ILSVRC比赛里打败了之前的那些网络。

相同的现象在CIFAR-10这个数据集上也得到了验证，从而说明我们的方法并不是仅仅局限于ImageNet这一个数据集的。

我们设计了一个152层的residual neural network在ILSVRC-2015的比赛里赢得了第一名。而且我们还将residual neural network用于ImageNet detection, ImageNet localization,
COCO detection, and COCO segmentation in ILSVRC & COCO 2015比赛均获得了第一名。这些证据足以说明我们的residual neural network这个框架是十分有效的。


>introductino是abstract的扩充，也是对文章整个过程的一个描述。这篇文章的introduction写的很标准。读者看完之后就能了解文章最核心的内容是什么。

**5. Related Work**

**5.1 Residual Representations**

**5.2 Shortcut Connections**


**6. Deep Residual Learning**

**6.1 Residual Learning**

假设$$H(x)$$是我们想要通过几层堆叠的layers要拟合的mapping，$$x$$是这些堆叠的layers的输入。如果假设多层的layers的堆叠能够拟合任意复杂的mapping，那么它也可以拟合residual mapping，也就是$$H(x)-x$$（假设输入和输出的维度是一样的）。所以说，我们不让堆叠的layers去拟合$$H(x)$$，而是显式的让它们去拟合residual mapping，$$F(x)=H(x)-x$$。从而我们所想要拟合的原mapping就可以表示为$$H(x)=F(x)+x$$。虽然说堆叠的layers都有能力去学习$$H(x)$$和$$F(x)$$，但是对于neural networks来说，学习的难易程度是不一样的。

上述的过程是由我们在introduction里的内容而启发的。如果我们所添加的layers在学习之后就是identity mapping，那么添加了这些layers至少不应该使得training error变大。但实际上，很深的neural networks的效果是会变差的，也就是说training error是会变大的。发生这种现象表明想要深的网络训练好，是很困难的。而使用residual的结构，则会使得训练简单很多。


**6.2 Identity Mapping by Shortcuts**

我们每过几层就采用residual learning的结构。而每个residual block就如fig 2所示。公式化来说，我们的residual block是这样的：
  
$$y = F(x, \{W_i\}) + x$$
  
$$x,y$$分别是这个block的输入和输出，而$$F$$代表residual mapping，$$W_i$$是参数。
  
上述公式里的shortcut既没有增加新的参数，也没有增加计算复杂度。这不仅在实践中会高效很多，也使得我们在实验中进行对比的时候，能控制其他变量不变，只是将shortcut的部分去掉，从而对比很公平。
  
在上述公式里我们可以看到，最后的结果$$y$$是residual mapping $$F$$和$$x$$的element-wise的addition，从而$$x$$和$$F$$的维度得是一样的。如果它们的维度不一样，比如说input和output的channels发生了变化，我们就引入linear projection $$W_s$$来解决问题：
  
$$y = F(x, \{W_i\}) + W_sx$$
  
residual mapping $$F$$的设计是多样的。我们这篇文章里$$F$$的设计用了2层或者3层 layers，而更多的层也是可以的。但是如果只有一层，那么其实residual block就变成了linear layer：$$y = Wx + x$$，这样是不行的。
  
而且上述的这些论述不仅仅是对于layers是MLP时候，layers是CNN的时候，同样也是适用的。


**6.3 Network Architectures**

我们测试了多种不同的plain/residual neural networks，对于我们的结论并没有什么不一样。用于ImageNet的plain和residual network结构分别如下：

**Plain Network**

我们的plain network的结构如fig 3中间那幅图所示。

**Residual Network**

我们在plain network的基础上加入shortcut connections，从而将网络改成了residual neural networks。对于输入和输出维度相同的residual blocks，shortcut connections就直接是elementwise addition。而对于不同的情况，比如说fig 3里的虚线，可以用zero padding的方法，也可以用$$ 1 \times 1$$的convolution来使得feature map的通道数相符。

![network]({{ '/assets/images/RESNET-3.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 3. ImageNet的示例结构。左侧：VGG-19模型。中间：34层的plain network。右边：34层的residual network。*

**6.4 Implementation**

输入的图片的短边随机从256到480的范围内采样，长边则是按比例计算。之后再从已经scale的图片里随机裁剪$$224 \times 224$$大小的图片，之后再随机的选择是否需要水平翻转。再之后将图片的平均值减去（elementwise）。而且在每个convolution之后，activation之前还采用了batch normalization。使用SGD进行优化，batch_size选取为256。learning rate从0.1开始并且在error到达平台期后手动减小至1/10。使用了weight decay-0.0001和momentum=0.9。并没有使用dropout。
  
>dropout常见于MLP或者fully connected layer里，convolution layer一般不用。
  
在测试的时候，我们从测试图片里随即裁剪10张进行测试。而且还对scale不同的image进行了测试再将结果取平均。
  
>这些都是刷榜的技巧，使得结果更好。


**7 Experiments**

**7.1 ImageNet Classification**

我们在ImageNet 2012 classification dataset上验证我们的方法，这个数据集有1000个类。模型在128万张图片上进行训练，并在50000张图片上进行验证。最后在100000张测试图片上测试最后的结果并报告。我们验证了top-1和top-5的error rate。
 * Plain networks
 对于plain networks，我们验证了18层和34层的效果。34层的plain network在fig 3中间进行了描述。而18层的plain network是相似的。从Table 1能看到模型的细节。
 ![table]({{ '/assets/images/RESNET-4.PNG' | relative_url }})
 {: style="width: 600px; max-width: 100%;"}
 *Table 1. ImageNet上所使用的plain networks的结构。conv3_1，conv4_1，conv5_1通过stride=2实施了降采样。*
 
 fig 4显示了18层和34层的plain network和residual network在training error和validation error上的区别。我们再一次发现，34层的plain network还不如18层的好，这说明34层的plain network欠拟合。
 
 ![plain]({{ '/assets/images/RESNET-5.PNG' | relative_url }})
 {: style="width: 600px; max-width: 100%;"}
 *Fig 4. 左边：plain network。右边：residual network。*
 
 我们认为上述的这种plain network欠拟合的问题，并不是因为梯度消失导致的。因为我们的网络里都加入了batch normalization。我们猜测可能是因为深的neural networks有着指数降低的收敛率，从而使得training error的降低很困难。但这个问题的原因还需要未来的工作来解决。

 * Residual Networks
 从fig 4我们可以看到residual结构很好地解决了深度neural network欠拟合的问题，因为对于34层的ResNet，其training error和validation error都要比18层的ResNet低，说明他确实学习到了更优越的性能并做到了generalization。

 * Identity vs. Projection Shortcuts
 有不同的shortcut的方法。(1) 只用zero-padding的方法来对于维度不同的情况处理 (2) 只使用projection，也就是$$1 \times 1$$的convolution来对维度不同的情况处理 (3) 只使用projection来处理，但对于维度相同的情况仍然也加上。
 
 >注意，在输入和输出维度不同的residual block里，虽然使用projection能够使得通道数很容易的匹配，但是实际上输出的高宽也是输入的一半，所以可以使用stride=2的$$1 \times 1$$的convolution来实现。

 * Deeper Bottleneck Architectures
 我们对于更加深的网络，比如说ResNet-50，101，152都采用了一种bottleneck的设计，这样可以减少计算复杂度，使得训练更快一些。我们可以在fig 5看到设计的结构。对于高通道数的输入，先用$$1 \times 1$$的convolution将通道数降低，再做普通的convolution，最后再用$$1 \times 1$$的convolution将通道数升回去。
 
 ![bottleneck]({{ '/assets/images/RESNET-6.PNG' | relative_url }})
 {: style="width: 600px; max-width: 100%;"}
 *Fig 5. 左边：普通的residual block。右边：加了bottleneck结构的residual block。*


**7.2 CIFAR-10 and Analysis**


**7.3 Object Detection on PASCAL and MS COCO**


>为什么ResNet训练的效果很好？实际上可能是因为它解决了梯度消失的问题。对于plain network来说：

>$$ \frac{df(g(x))}{dx} = \frac{df(g(x))}{dg(x)} \frac{dg(x)}{dx} $$

>梯度都是很小的，所以说这两个乘积项本身都很小，乘在一起就更小了。而网络很深的时候，误差反向传播到很前面的网络的时候，就容易发生梯度消失的问题。

>对于residual network来说，情况就不一样了：

>$$ \frac{df(g(x)) + g(x)}{dx} = \frac{df(g(x))}{dg(x)} \frac{dg(x)}{dx} + \frac{dg(x)}{dx} $$

>两项梯度相加，就很有可能使得梯度不是那么小了，从而就可以训练的动了。

**8. Conclusion**

这篇文章是没有结论的。因为文章内所要说的结果太多了，超出了会议规定的文章最大页数。这种写法是不建议的，最好还是要有conclusion，使得文章具有完整性。


## Transformer-related architectures

### 1. [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
*Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin*

*NIPS 2017*

**1. Title**

这个标题很有意思，开创了一个潮流，之后很多论文都用类似的标题。

**2. Authors**

这篇论文有8个作者，且对文章的贡献都是均等的，而且还有额外的解释说明了每个作者的贡献。这是很少见的，但是这是Google的论文，他们经常做这种骚操作。

**3. Abstract**
主流的序列转录模型（输入和输出都是序列，比如说机器翻译，输入是一种语言序列，输出是另一种语言序列）主要都是用复杂的recurrent或者convolutional的neural network，而且都是有encoder和decoder的结构。而且表现很好的那些模型，在encoder和decoder之间是有attention机制进行连接的。我们这篇文章提出了一个简单的架构，只用attention机制，而没有任何recurrent或者convolutional的结构，叫做Transformer。我们在两个机器翻译任务上测试了Transformer，发现它效果更好，而且并行度更好所以训练的时间也更短。我们的模型在WMT 2014 English-to-German的翻译任务上达到了28.4 BLEU。在WMT-2014 English-to-French的翻译任务上，通过8个GPU训练了3.5天达到了41.8的BLEU。


>这篇文章一开始是针对机器翻译而写的，所以整篇文章、摘要都只提到了机器翻译的实验。之后因为Bert等文章使得Transformer这篇文章火了，才受到大量人的关注。但这篇文章一开始就是机器翻译领域的文章。

**3. Introduction**
RNN，包括LSTM，gated RNN等，在序列转录任务里包括Language modeling，machine translation等具有重要的地位，基于RNN的模型也在这些任务里获得了最好的结果。recurrent language models和encoder-decoder architecture是最主流的两个模型。

RNN计算一个序列的时候，是从左往右挨个词来看。对于第t个词，会计算一个输出$$h_t$$，也叫做它的隐藏状态，而$$h_t$$是由前面一个词的隐藏状态$$h_{t-1}$$和当前的词来决定的。这样就可以将历史信息传递过去，这也就是RNN为什么能够处理时序信息的关键。这个时序的特征使得并行计算是不可行的，这对于很长的序列是很致命的。另一个缺点是时序信息是按串联的方法依次传递下去的，所以对于很早的历史信息就可能会丢失掉。解决这个问题的办法就是使用更大的$$h_t$$，但是这会导致计算量和内存量的需求变大。

attention机制已经在很多任务的模型里被应用了，对于input和output序列来说，attention机制可以表示它们之间或者内部的依赖关系，而不需要顾及依赖关系双方的距离有多远。但attention机制一直是与RNN一起使用的。

>在RNN里用的attention机制主要是将encoder的信息有效的传递给decoder。

在这篇文章里，我们提出了Transformer，不再使用recurrence结构而是只使用attention机制来构建input和output之间的依赖关系。我们的Transformer可以进行并行计算，使得训练时间大大减短，而且效果更好。

>这个introduction写的比较短，可以认为是abstract的扩充。这么写是因为后面的主题内容过多，而NIPS是一个篇幅较短的会议，所以只能压缩其他部分的内容。


**4. Background**

为了减少序列计算的成本，很多基于CNN的架构被提了出来，它们都是使用并行计算来构建input和output之间的依赖关系。但是因为CNN的卷积核是有大小的，CNN的卷积操作是在数据上按照卷积核的大小以一定步长滑动的，所以说如果两个相距很远的词想要被学习到之间的关系，就需要很多的卷积操作才可以（因为CNN是卷积层的叠加，后面的卷积操作基于前面的卷积操作）。从而CNN的这种特性使得学习距离很远的词之间的依赖关系变的更加的复杂。使用Transformer结构的话，模型一层就可以看到所有的词，就不会有CNN那样的问题。但CNN的一个优势是有很多的通道，而每个通道都可以代表不同的模式，而我们也想在Transformer里实现这样的效果，所以我们使用Multi-head attention的结构，希望能够模仿这种多通道的效果。

self-attention，有时候也叫做intra-attention，是一种attention机制，它作用在一个单一序列的不同的位置上，从而计算出这个序列有效的representation。self-attention机制也并不是这篇文章提出来的，很早以前就在很多地方被使用过了。

Transformer是第一个只依赖于self-attention机制而不需要用到任何recurrent或者convolutional结构来计算input和output的representations的模型。


**5. Model Architecture**

目前比较好的序列转录模型都有一个encoder-decoder的结构。encoder将输入的序列$$(x_1,...,x_n)$$映射到$$z = (z_1, ..., z_n)$$，$$z$$的长度也是$$n$$，而且每个$$z_i$$是一个向量，是$$x_i$$的feature。而decoder则是将$$z$$作为输入，而输出一个$$(y_1,...,y_m)$$，$$m$$和$$n$$是不一定一样的。而且$$(y_1,...,y_m)$$里的$$y_i$$是一个一个依次生成的。这样的一个生成的方式叫做auto-regressive，也就是说在每一步都需要前面已经生成的$$y$$作为额外的输入，配合$$z$$来生成新的$$y$$。

而Transformer也是采用了这样一种encoder-decoder的结构，Transformer将self-attention和point-wise fully connected layers堆叠在一起实现的encoder和decoder。如fig 1左右两部分所示。

![architecture]({{ '/assets/images/TRANSFORMER-1.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 1. Transformer：模型结构。*
 
**5.1 Encoder and Decoder stacks**
 
**Encoder**: encoder是由$$N=6$$个完全一样的层堆叠构成的。而每个层都有两个子层。第一个子层是multi-head self-attention机制，第二个子层实际上就是个MLP。对于每个子层我们还有残差连接，而且每个子层还有个layer normalization。也就是说，每个子层的输出为： LayerNorm(x + Sublayer(x))，其中Sublayer(x)是子层里的function（也就是前面说的multi-head self-attention或者是MLP）。为了简便残差连接，这个模型里所有的子层，以及初始的input embedding，输出的大小都是$$d_{model} = 512$$。

>这种简单的设计使得模型调参很简单，因为encoder部分就两个超参数，$$N$$和$$d_{model}$$。

>通过和BatchNorm对比一下解释LayerNorm是什么，以及为什么我们在这种输入是变长的模型里不使用BatchNorm。假设我们的输入是2维的，row表示batch，column表示feature，也就是每一行是个样本，每一列是个特征。BatchNorm是对于每个batch，将每列特征的均值变成0，方差变成1，这是通过计算整个batch对于每一列的均值和方差，之后再减去均值，除以方差的根号来实现的。最后还有可学习的参数，$$\lambda$$和$$\beta$$，来对normalization的值做一个线性转换，使得其表达性更强。而LayerNorm和BatchNorm很像，也是用来做归一化的，但区别在于LayerNorm是将这个2维输入的每一行进行归一化，也就是对于每个输入，其内部进行归一化。如fig 2所示。

![LN]({{ '/assets/images/TRANSFORMER-2.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 2. BatchNorm和LayerNorm的2维例子，row是batch，column是feature。*

>但是对于Transformer或者正常的RNN而言，输入是3维的，因为输入的是一个batch的序列，而每个序列里有很多个元素，每个元素又可以被表示为一个embedding向量，从而输入是个$$batch \times sequence \times features$$的三维数据。对于BatchNorm来说，我们仍然对于每个feature都做归一化，如fig 3蓝色方框所示，也就是对于每个feature而言，对于所有的batch和所有的sequence的元素求归一化。而对于LayerNorm来说，我们对每个batch做归一化，也就是说对于每个batch来说，我们对所有的feature和所有的sequence内的元素求归一化，如fig 3黄色所示。

![EX]({{ '/assets/images/TRANSFORMER-3.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 3. BatchNorm和LayerNorm的3维例子。*

>所以说为什么要用LayerNorm而不是BatchNorm呢？因为很多时候，我们一个batch里面的sequence里包含的元素的个数是不等的，如fig 3所示。那么如果用BatchNorm，如fig 4蓝色部分所示，在计算方差和均值的时候，有很多部分是0，是没有用的。那么在各个batch之间，这种sequence内元素长度的变化，会导致方差和均值计算的不稳定，很波动。而且还有个问题就是如果在测试的时候遇到了一个sequence，里面包含的元素的数量比所有的训练时候所看到的sequence内的元素数量都要多，如fig 4上面那条蓝色的长条所示，那么这个时候用之前所计算的均值和方差来处理这个数据就是不合理的。而对于LayerNorm来说就不存在这个问题，因为LayerNorm计算均值和方差是在每个sequence内部计算的，所以不存在其它的sequence还能影响它，从而长度也不会影响。

![E4]({{ '/assets/images/TRANSFORMER-4.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 4. BatchNorm和LayerNorm生成的结果的样子。*


**Decoder**: decoder和encoder是很像的。decoder同样是由$$N=6$$个相同的层堆叠而成的。每个层有和encoder一样的两个子层，而且decoder还多了个子层，同样也是multi-head attention，但这回不是self-attention，而是利用了encoder的输出。decoder里的每个子层也用了残差连接的结构，并且跟上了layernorm。因为我们采用的是自回归的方式，所以decoder里的multi-head self-attention的结构也需要更改一下，将还没有看到的那些输出遮住，也就是我们所说的masked multi-head attention。

接下来我们来看每个子层是什么样的。

**5.2 Attention**

一个attention function可以被描述为将一个query和一个key-value对的集合映射到一个output，其中query，key，value和output都是向量。output是values的一个加权和，而每个value的权重则是通过用compatibility function来衡量query和key获得的。而不同的attention的结构里，compatibility function用的不一样，也就是衡量相似度的方式不一样。

**5.2.1 Scaled Dot-Product Attention**

我们这篇文章里用的attention叫做scaled dot-product attention，如fig 5左边所示。

![att]({{ '/assets/images/TRANSFORMER-5.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 5. 左：scaled dot-product attention。右：multi-head attention（有多个并行计算的attention layers）。*

我们attention的输入里的query和key的维度都是$$d_k$$，而value的维度是$$d_v$$（从而output的维度也是$$d_v$$）。我们的做法是对于query和每个key做dot product，再除以$$\sqrt(d_k)$$，然后再用一个softmax function来获取每个value的权重。

在具体操作中，我们针对一系列的query同时进行计算。将这些query表示为Q，key和value也同样表示为K和V，Q, K, V都是矩阵。从而，output的矩阵就是：

$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt(d_k)})V$$

两种最常见的attention functions是additive attention和dot product attention。我们这里用的就是dot product attention，除了还额外除以了$$\sqrt(d_k)$$。而addtive attention是通过一个只有一个隐藏层的MLP实现的， 它可以处理query和key不等长的情况。这两种attention function在理论上复杂度差不多，但在实践上因为矩阵乘法有高效算法，所以后者会快很多。

当$$d_k$$不大的时候，上述两种attention function的效果差不多，但$$d_k$$大的时候，如果不除以$$\sqrt(d_k)$$，additive attention的效果要更好。我们猜测是因为$$d_k$$很大的时候，计算出来的dot product有的会很大，从而到了softmax function的饱和区，会在反向传播的时候产生很小的gradient，影响训练效果。所以我们除以了$$\sqrt(d_k)$$。
（因为softmax的作用本来就是达到一个效果，在大的地方输出接近1，小的地方输出接近0，如果已经达到了这种效果了，那么softmax就难以再训练了）

而对于mask而言，上述计算依然进行，只不过在进入softmax之前，对于那些还不应该被看到的output的value对应的权重，将其都置成非常大的负数，比如说-1e10，这样在进入softmax之后，其权重就几乎是0，从而被排除在外。

**5.2.2 Multi-Head Attention**

与其我们只用一个key，value和query的维度都为d_{model}的attention function，我们发现将query，key和value用不同的可学习的linear projection映射到$$d_k$$，$$d_k$$和$$d_v$$维度$$h$$次，从而学习$$h$$个output的效果更好。这些$$h$$个attention function还可以并行操作，从而计算的更快。而这$$h$$个output连接在一起，维度是$$d_k \times h$$，再经过一次linear projection，从而得到$$d_v$$维度的output。如fig 5右侧所示。

>为什么要用multi-head的结构呢？因为我们可以发现attention function其实没有什么可以学习的参数，相似度就是内积。但我们有时候希望能学习到更多的模式，而这些模式可能需要不一样的计算相似度的方法。如果用的是additive attention的结构，其实那里面还有一些可以学习的参数，但这里没有。从而这里的做法是给$$h$$次低维的投影，希望其能够找到有效的投影模式，从而学到有效的内容。而投影是有参数可以学习的，从而增加了模型的学习能力。有点像CNN里的多个通道的意思。

multi-head attention使得模型能够从不同的subspace里学到不同的信息。

$$MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O$$

其中$$head_i = Attention(QW_I^Q, KW_i^K, VW_i^V)$$，而这些linear projections都是可学习的矩阵$$W_i^Q \in R^{d_{model} \times d_k}$$，$$W_i^K \in R^{d_{model} \times d_k}$$，$$W_i^V \in R^{d_{model} \times d_v}$$，$$W^O \in R^{hd_v \times d_{model}}$$。

在这篇文章里，我们使用$$h=8$$。使用$$d_k=d_v=d_{model}/h=64$$。因为我们做multi-head的时候降维度了，所以其实multi-head的计算量和single-head的差不多。

**5.2.3 Applications of Attention in our Model**

整个Transformer模型用三种不同的方式使用multi-head attention。

* 在encoder-decoder的attention子层里，query是从前面一个decoder的子层来的，而key和value则是由encoder的输出提供。这种设计可以使得decoder里的每个位置都能获取输入的sequence所有位置的信息。这模仿了在sequence-to-sequence模型里的典型的encoder-decoder attention的机制。
* encoder本身包含了self-attention子层。在一个self-attention子层里，key，value和query都是一样的，也就是前一个子层的输出。从而encoder的每一个位置都可以获取encoder其它位置的信息。
* 相似的，decoder里也有self-attention子层，它允许decoder的每个位置都能获取该位置之前的所有位置的信息。因为这是个auto-regressive的模式，所以说每个位置后面的信息它是获取不到的。我们通过在softmax之前将对应的权重设置为很大的负数来做到mask out。


**5.3 Position-wise Feed-Forward Networks**

encoder和decoder的每个层，除了有attention子层以外，也有还有一个fully connected network，其对于输入的每个词单独作用（也就是说有每个子层只有一个MLP，其作用在每个词上，也就是每个position上，而不是一个整体的大的MLP，这就是position wise名字的来由）。而这个MLP有两个线性层，并且其中还有ReLU：

$$FFN(x) = max(0, xW_1 + b_1)W_2 + b_2$$

虽然说在同一个子层里对于不同的position（词）我们使用的是同一个MLP，也就是同一个linear transformation（with ReLU），但不同的层其内部的MLP的参数还是不一样的。另一种描述这个MLP的方式就是kernel size是1的convolution。对于这个FFN来说，input和output的维度都是$$d_{model}=512$$，内部隐藏层的维度是$$d_{ff}=2048$$。

>attention的作用其实就是将序列里的信息抓取出来，做一次汇聚。因为在汇聚之后每个position其实就已经有序列整体的信息了，所以说不同position的MLP可以是同一个。

>RNN和Transformer的区别：见fig 6。Transformer和RNN一样，都是用一个MLP来做语义空间的转换，而且RNN也是用了同一个MLP来处理不同position（实际上RNN就只有一个MLP在不断更新），但不一样的是如何传递序列信息。RNN是把上一个时刻的输出作为下一个时刻的一个输入，而Transformer是通过attention结构来获得全局的序列信息，再用MLP进行转换。它们的关注点都在于如何有效使用序列信息，但实现的方式不一样。


**5.4 Embeddings and Softmax**

和别的sequence tranduction模型类似，我们使用需要学习的embedding来将encoder和decoder的输入的tokens都映射到$$d_{model}$$维的向量。我们同样使用需要学习的linear transformation和softmax function将decoder的输出转换为预测下一个token的概率。在我们的模型里，encoder和decoder的tokens的embeddings层的权重和decoder之后的linear transformation的权重是共享的。在embedding层，我们将这个层的参数乘以$$\sqrt(d_{model})$$。

>因为embedding最后容易将每个输入的norm都学成l2norm接近于1，所以乘上一个系数，以便它的值的范围和positional encoding的范围差不多，不至于过小。


**5.5 Positional Encoding**

>attention其实并没有任何时许信息，那么即使打乱一个句子的词的顺序，也会是一样的效果，这是不应该出现的，所以我们需要加入时序信息。RNN是本身就有时序结构，而Transformer的做法是直接在输入里加入时序信息。

因为我们的模型并没有recurrence或者convolution的结构，所以为了我们的模型能够利用sequence的时序信息，我们需要加入一些sequence里的tokens的相对或者绝对的位置信息。为了达到这个目标，我们在encoder和decoder的输入的embeddings上加上了positional encoding。positional encoding和embedding一样也是$$d_{model}$$维的，所以两者可以相加。positional encoding有很多种。

在我们这篇文章里，我们使用不同频率的sine和cosine function来表示positional encodings：

$$PE_{(pos, 2i)} = sin(pos/10000^{2i/d_{model}})$$

$$PE_{(pos, 2i+1)} = cos(pos/10000^{2i/d_{model}})$$

其中pos是position，i是dimension。也就是说，每一个dimension，都对应着一个三角函数。


**6. Why Self-Attention**

在这一节里我们将会比较self-attention层和recurrent以及convolution层之间各个不同的方面。

首先是每个层的计算复杂度。另一个是可以被并行的计算的总和。第三个是长距离的两个position之间需要信息交互需要的计算量。学习长距离的依赖关系对于很多sequence transduction任务来说都是很具有挑战性的。影响模型学习长距离依赖关系的一个重要的点就是距离远的两个position之间的信息传递最少要经过多少计算。任意的两个position之间的信息交换所需的计算量越小，模型学习这种长距离的依赖关系的能力就越强。结果如table 1所示。

![tab]({{ '/assets/images/TRANSFORMER-7.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Table 1. *

**Experiment**和**Result**就不讲了，因为都是NLP领域的实验，和我的方向没什么关系。


**7. Conclusion**
我们介绍了Transformer，这是第一个仅仅依赖于attention机制的序列转录模型，将所有的recurrent layers都换成了multi-headed self-attention。

在机器翻译这个任务上，Transformer要比那些recurrent或者convolutional layers结构的模型训练起来要快很多。在WMT-2014 English-to-German和English-to-French的任务上确实效果很好。

我们对于这种只利用了attention机制的模型十分有信心，认为它可以被应用在别的任务之上。我们打算将Transformer应用到text以外的数据上，包括images，audio，video等。而使得生成不是那么的时许化也是另一个未来的研究方向。



### 2. [BERT: Pre-training of Deep Bidirectional Transformers for Language understanding](https://arxiv.org/pdf/1810.04805.pdf)

[CODE](https://github.com/google-research/bert)

*Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova*

*Arxiv 2018*

**Transformer $$\rightarrow$$ BERT**

>在CV领域，很早之前就可以使用CNN在一个很大的数据集，比如说ImageNet上学习一个模型，而这个模型可以为很多的CV任务提升他们的性能，也就是说这是一个pre-trained的模型。但是在NLP领域，在BERT之前，一直都没有一个训练好的NLP网络，使得其再训练好之后能够帮助后面一批的NLP任务。从而NLP任务都是针对每个任务单独构造网络然后单独训练。BERT的出现使得NLP领域有了这样以一个在很大的数据集上训练好的pre-trained模型，其能够应用于后续很多的NLP任务当中，既简化了这些NLP任务的训练，又提升了它们的性能。所以说BERT以及后续的改进使得NLP领域在过去几年内有了个质的飞跃。

**Title**

BERT是起好了一个名字，这样方便之后的人引用或者提及。pre-training意味着我们这篇文章的目的是训练一个pre-trained的模型，用于后续其它一批任务上。deep bidirectional transformers解释了所用的主要的网络结构，而language understanding则解释了这个pre-trained模型之后可以应用在哪些任务上。

所以说这个标题告诉我们，BERT这个模型是一个深的双向的transformer，是用来做预训练的，针对的是一般的语言理解的任务。

**Authors**

是Google AI语言组的人，一作从有想法到跑第一次代码成功只花了几周的时间，之后写完论文也就花了几个月的时间。


**Abstract**

我们提出了一个新的language representation模型，叫做BERT，是Bidirectional Encoder Representations from Transformers的简称。和最近的language representation模型（ELMO和GPT）不一样，BERT通过在每一层里都联合左右两侧的上下文来在没有标签的文本上学习预训练的bidirectional representation。结果是，预训练的BERT模型再加上一个附加的output层就可以对一系列任务都有sota的效果，比如说question answering，language inference等，而不需要对具体任务来调整模型的结构。

>ELMO使用的是RNN的架构，所以对于不同的下游任务模型结构要做不同的修改，而GPT虽然也使用的Transformer，但是GPT是单向的预测，并没有用到双向上下文信息。

BERT在概念上很简单，在实验上效果很好。其在11个NLP任务上达到了sota的效果。


**1. Introduction**

语言模型预训练对于提升很多NLP任务性能都有显著的效果。这些NLP任务包括sentence-level的任务比如说natural language inference，paraphrasing等，它们通过整体分析句子来预测sentences之间的关系，也包括token-level的任务比如说named entity recognition，question answering等，它们在token level生成更加细粒度的输出。

现在有两种将预训练的语言特征应用到下游任务的方法：feature-based和fine-tuning。feature-based方法，比如说ELMo，使用task-specific架构，将预训练的特征作为补充特征来使用。fine-tuning方法，比如说GPT，尽量对于各种任务都使用同样的模型结构，在各个不同任务的数据集上来微调已经预训练好的模型参数。这两种方法在预训练时是一样的，使用的是一样的目标函数，也就是使用unidirectional language models来学习语言特征。

>unidirectional language models就是经典的语言模型，也就是从过去的词和句子来预测接下来的词或者句子。

我们认为现有的这些方法限制了预训练语言特征的效果，特别是对于fine-tuning方法。最主要的一个限制点是标准的语言模型是单向的（unidirectional），这限制了可以被用来做预训练的模型种类。比如，在GPT里，作者使用了一个从左到右的模型结构，其中每个token在Transformer结构的网络里只能获取它之前的tokens的信息。这样的限制对于sentence-level的任务来说是sub-optimal的，而对于使用fine-tuning方法的token-level任务来说也是很不好的，比如说question answering，因为双向信息会格外的重要。

在这篇文章里，我们改进了fine-tuning based方法，提出了BERT：Bidirectional Encoder Representations from Transformers。BERT通过使用了一个masked language model (MLM)减轻了之前提到的单方向的限制。masked language model随机的在输入里遮住一些tokens，而目标则是根据上下文来预测被遮住的tokens。和从左到右的预训练模型不一样，MLM的目标函数使得所学习到的representations能够融合左右两侧的信息，也就允许我们预训练一个bidirectional Transformer。除了MLM，我们还使用了一个next sentence prediction任务来联合预训练text-pair representations（这个任务是给定两个句子，预测这两个句子是不是相邻的，这个任务可以让模型学习句子层面的信息）。我们这篇文章的贡献总结如下：

* 我们阐述了双向预训练模型对于语言representations的重要性。和GPT使用单向语言模型学习representation不同，BERT使用MLM来学习双向语言representations。这和ELMo也不一样，ELMo是仅仅将从左到右和从右到左分别学习到的语言representations简单的连了起来。
* 我们说明这样的预训练representations使得很多针对具体任务设计具体模型结构的努力变得不再需要了。BERT是第一个在很多sentence-level和token-level的下游任务上都获得了sota效果的预训练模型，效果超过了很多根据具体任务设计具体模型结构的方法。
* BERT在11个NLP任务上达到了sota的效果。


**2. Related Work**

预训练的语言representations有着很长的历史，所以只能简要的介绍最著名的那些。

**2.1 Unsupervised Feature-based Approaches**

学习words的能够广泛使用的representations已经是一个研究了几十年的方向，包括non-neural和neural方法。预训练好的word embeddings是现在NLP系统里不可或缺的一部分，比之前hand-crafted的embeddings效果要好很多。为了预训练word embedding向量，很多方法应用了从左到右的语言模型目标函数。

这些方法之后被扩展到更粗粒度的情况，比如说sentence embeddings，或者paragraph embeddings。为了训练sentence representations，以前的工作有的使用目标函数来排列下一个sentence的候选，有些工作基于之前的sentences的representations生成下个sentence，还有些工作使用denoising autoencoder。

ELMo以及其后续跟进工作将传统的word embedding拓展到一个新的高度。ELMo利用一个从左到右和一个从右到左的语言模型来获取上下文相关的features。每个token的上下文相关的representations就是将从左到右的模型的特征和从右到左的模型的特征连起来而获得。和目前已有的task-specific模型相结合起来，ELMo在多个NLP任务上取得了sota的效果，包括question answering，named entity recognition，sentiment analysis等。还有研究工作提出使用LSTM来从左侧和右侧上下文预测单个词。和ELMo类似，它们的模型是feature-based的而且并没有深度双向。

**2.2 Unsupervised Fine-tuning Approaches**

正如feature-based方法，fine-tuning based方法也是从无标签的数据上预训练word embeddings。

最近，生成上下文相关的token representations的sentence或者document encoders也是从非标签的文本上预训练或者从监督的下游任务上训练的。GPT在很多sentence-level的任务上获得了sota的效果。从左到右的语言模型以及auto-encoder目标函数被用来预训练这样的模型。


**3. BERT**

在这一节里我们介绍BERT和它详尽的细节。我们的框架有两个步骤：pre-training和fine-tuning。在pre-training过程中，模型在不同的pre-training任务上使用非标注的数据进行训练。在fine-tuning过程中，BERT模型在预训练后的参数被下游的任务的监督数据进行进一步的微调。每一个下游的任务都会微调出不同的模型。fig1使用question answering作为例子来解释了这个过程。

![1]({{ '/assets/images/BERT-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 1. BERT的overall pre-training和fine-tuning过程。除了输出层，pre-training和fine-tuning的网络结构是一样的。对于不同的下游任务，也是使用同样的预训练好的网络参数。在fine-turning过程中，所有的参数都被微调了。$$\left[CLS \right]$$是加在所有输入的前面的特殊的符号，$$\left[ SEP \right]$$是一个特殊的separator token（比如，分开questions和answers）。*

BERT的一个特殊的地方是它在不同的任务里的模型结构都是一样的。在预训练模型框架和最后的下游任务模型框架之间存在很小的差别。


*Model Architecture*

BERT的模型结构是一个多层的双向Transformer encoder，在TensorFlow的tensor2tensor库里收录。因为Transformer太火了，而我们的模型基本上和原Transformer一模一样，就不再赘述。

在这篇文章里，我们将层数（也就是Transformer的blocks数）定义为$$L$$，隐层大小为$$H$$，self-attention的head个数为$$A$$。我们在两个模型上测试结果：$$BERT_{BASE}$$，$$L=12, H=768, A=12$$，总参数为1.1亿；$$BERT_{LARGE}$$，$$L=24, H=1024, A=16$$，总参数为3.4亿。

$$BERT_{BASE}$$的模型大小（参数个数）设计为和GPT差不多，为了能够更合理的比较。


*Input/Output Representations*

为了使得BERT能够处理一系列下游任务，BERT的输入用一个token sequence来表示，这个sequence既能表示单个sentence，又能表示一对sentences构成的pair（比如说，$$\langle Question, Answer \rangle$$）。一个sentence可以是任意长度的连续文本，而并不是我们平常意义里的一个sentence。一个sequence表示的是BERT的token sequence输入，可能是一个sentence也可能是两个打包在一起的sentence pair。

>Transformer那篇文章的输入是一个序列对，因为既有encoder又有decoder所以可以实现。但BERT只有一个encoder，为了使得输入仍然可以是一个序列对，就只能使用上面这种方法。

我们使用有30000个token vocabulary的Wordpiece embeddings（[Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf%20(7.pdf)）（Wordpiece这种切词方法使得我们的词典并不是很大，从而使得嵌入层的参数没那么多，如果词典很大的话，那可学习参数都跑到嵌入层去了，就无法好好学习Transformer里的参数了）。每个sequence的第一个token都被设定为一个特殊的classification token，记为$$\left[ CLS \right]$$。这个token最终对应的模型的hidden state被认为是aggregated sequence representation被用来做classification任务（也就是这个token对应的输出最终表示的是一整个sequence的信息，因为Transformer使用自注意力机制，所以每个词都能看到所有其它词的内容，而无所谓这个词的位置在哪，所以将这个token放在第一个位置也是可以的）。sentences pairs会被pack到一起成为一个sequence。我们使用下述方法来区分一个sequence里的两个sentences。首先，我们用一个特殊的token，$$\left[ SEP \right]$$来区分它们。其次，我们给每个token都加上一个可学习的embedding来表示这个token是属于sentence A还是sentence B。正如fig1所示，我们将输入的embedding记为$$E$$，特殊token，$$\left[ CLS \right]$$的最终的hidden vector记为$$C \in R^{H}$$，第$$i$$个输入token的最终的hidden vector记为$$T_i \in R^{H}$$。

对于某个token，它输入给BERT的向量表示是这个token本身的embedding，加上它在哪个句子对应的embedding，再加上position encoding而获得。这个构造的可视化见fig2。

>在Transformer里，positional encoding是手动设置的，但是在BERT里，token本身的embedding，还是属于哪个句子的embedding，或者是positional embedding都是学习得来的。

![2]({{ '/assets/images/BERT-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 2. BERT输入representation。每个输入的token的embedding是token embedding，segmentation embedding和positional embedding加起来获得的。*

**3.1 Pre-training BERT**

跟ELMo和GPT不一样，我们并不使用传统的从左到右或者从右到左的语言模型来预训练BERT。我们使用两个非监督的任务来与训练BERT。fig1的左边展示了pre-training的过程。

*Task 1: masked LM*

直觉上，认为一个双向的模型肯定比一个单向的从左到右的模型或者只是简单的将从左到右和从右到左模型的输出连起来的模型要强大是很正常的。但实际上，标准的conditional language model只能是从左到右或者从右到左训练，因为双向会使得每个词非直接的看到自己，从而模型可以在多层结构的上下文环境中很容易的预测target word。

为了能够训练双向的representations，我们只是将一些输入的tokens遮起来，然后让模型来预测这些遮起来的词。我们将这个过程称为masked language model（MLM）。之后，模型对应于被遮住的tokens的最后的hidden vectors被喂给一个output softmax over the vocabulary，来预测这个被遮住token到底是哪个词，正如标准的LM里所做的那样。在我们的所有的实验里，我们遮住每个sequence里随机的15%的tokens。和denoising auto-encoders不同，我们仅仅是预测被遮住的tokens，而并不是要重构整个原输入。

尽管这样做让我们可以获得一个双向预训练模型，一个缺点是模型的输入数据在预训练和fine-turning的时候会不一样，因为fine-tuning的时候并没有$$\left[ MASK \right]$$ token。为了缓解这个问题，我们在预训练的时候再加入一些随机性。当第$$i$$个token被选中作为mask token时，我们80%的概率将其作为mask token，10%的几率将其替换为一个随机token，10%的几率不改动。然后$$T_i$$再和最初的token利用cross-entropy loss计算差别。

*Task 2: Next Sentence Prediction (NSP)*

很多重要的下游任务比如说question answering（QA）以及natural language inference（NLI）基于的是两个sentences之间的关系，这并不能直接被language model所获得。为了训练一个能够理解sentence关系的模型，我们为一个binarized next sentence prediction任务来作预训练。当选择sentence A和sentence B作为预训练例子的时候，50%的时候，B是A的下一个sentence（标注为IsNext），50%的情况是随机选的（标注为NotNext）。正如我们在fig1中所示，$$C$$就被用于next sentence prediction (NSP)。尽管结构很简单，实验证明这样做对于后续的QA和NLI任务效果提升都很有作用。


**3.2 Fine-tuning BERT**

fine-tuning是十分自然地，因为Transformer里的self-attention机制使得BERT能够建模很多下游的任务（不管输入是单个文本或者文本对），fine-turning的时候将输入输出换掉就行。对于那些使用文本对的任务，一个常见的方式是在使用bidirectional cross attention之前先独立的encode文本对。BERT并不这样做。BERT使用self-attention机制来将这两个stages统一到了一起，直接使用self-attention机制来对pack到一个sequence的两个sentences构成的pair进行处理，等价于在两个sentences之间进行bidirectional cross attention。

>BERT和基于encoder和decoder的架构有什么不一样，这样的架构比如说Transformer。对于BERT来说，其直接将两个句子对打包在一起放进去了，所以self-attention就使得BERT能够直接看到两个句子之间的关系，但是在encoder-decoder的架构里，encoder一般是看不到decoder里的信息的。所以说BERT处理这种任务效果会更好。但也付出了代价，BERT就无法做机器翻译这种类型的任务了。

对于每个任务，我们很简单的将该任务的输入和输出放到BERT里，来微调BERT的系数。对于输入来说，预训练时候的sentence A和sentence B就类似于（1）paraphrasing任务里的sentence pair；（2）entailment任务里的hypothesis-premise pairs；（3）question answering里的question-passage pairs；和（4）text classification或者sequence tagging里的一个degenerate text pair。对于输出来说，对于token-level的任务（比如说sequence tagging或者question answering），token representation被喂给一个output layer，而对于classification-level的任务（比如说entailment或者sentiment analysis），$$\left[ CLS \right]$$ representation被喂给一个output layer。所以说不管怎么样，在BERT输出之后加一个层就可以做各种不同的任务了。


**4. Conclusion**

最近由于language models的transfer learning所带来的实验上效果的提升已经证明了种族的非监督的预训练已经是很多语言模型的不可或缺的一部分了。特别是，这些预训练模型使得某些难以获得数据的任务也能够从很深的网络中获利。我们主要的贡献是进一步将上述发现拓展到bidirectional的模型架构，允许一个预训练模型在不改变结构的情况下处理一批下游任务。


**写作总结**

BERT使用的是encoder，而GPT使用的是decoder，BERT使用了双向性的结构，得到了很多好处，但也失去了一些，比如说做机器翻译就不好做了，做文本的摘要提取也不好做了，也就是那些生成类的任务都不好做了。但是分类相关的问题在NLP里更加常见一点，所以BERT受到了非常大的追捧。BERT很符合深度学习的发展，BERT在一个很大的数据集上预训练好，然后在后续一系列的任务上微调之后都能获得很好的效果，这样一个预训练的模型就显得很重要。对于NLP群体来说，BERT这样一个很大的模型在很大的数据集上预训练好了的模型，就很重要了。


### 3. GPT-1, GPT-2和GPT-3

GPT-1：[Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

GPT-2: [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

GPT-3: [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)

GPT系列和BERT系列的模型在今天的自然语言处理界已经可以说是无人不知无人不晓。尤其是GPT2出来的时候，openai放话说因为该模型的功能太强大，担心被有心之人滥用所以选择不开源，炒足了噱头，引起了巨大的媒体轰动。虽然过了几年回头看，觉得该团队对该模型有些过于自信，但无可否认的是该系列的模型在刚刚发布的时候，对于各项任务的处理都有优异的表现甚至是STATE OF THE ART的级别。有意思的是，GPT1的论文在投稿的时候并不是一帆风顺，甚至几番被拒稿并且从未被任何顶会接受。其中一个原因便是GPT1的模型在架构上几乎没有任何的创新。但为什么每次新的GPT模型放出后都受到一众大佬的研究与热议，而具体文章的（开创性）贡献在哪呢。

**GPT-1**

**1. Introduction**

在CMU博士后研究员刘鹏飞的一篇的综述（[Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://arxiv.org/pdf/2107.13586.pdf)）里，他介绍了自然语言学界经历过的四种任务处理范式。他认为古早时期的第一种范式便是语言学家需要手工设计一系列特征模板，来输入模型。模型对任务的处理结果高度依赖于特征模板的设计，间接地便高度依赖于领域专家的知识。举个例子，如果有学过自然语言处理的经典算法课的可能对条件随机场CRF模型不陌生。业界甚至有一个专门的库CRF++帮助你自动生成大量的随机模板输入模型进行训练从而避免对领域专家的需要。可是当第二范式神经网络学派开始流行以后，用预训练后的词嵌入表征加上模型架构的调整，便取得了相似甚至远超过第一范式的效果后，需要大量人工介入的第一范式便渐渐式微了。在这个时期我们可以看到大量的工作在词嵌入上，比如NNLM，CBOW,SKIP-GRAM,GLOVE,ELMO等。也可以看到大量的工作在模型架构上，比如BI-LSTM, SEQ2SEQ架构在神经机器翻译领域NMT的应用等。而真正开启第三范式，在超大的文本数据集上预训练一个通用的模型，接着再对下游的特定任务微调的PRETRAIN-FINETUNE的范式，则是我们今天本文的主角，GPT1模型。相比于第二范式而言，第三范式的优点在于更进一步减少了人工的参与。不再需要对于每个任务采取不同的模型架构，而是用一个取得了优异泛化能力的模型，去针对性地对下游任务进行微调。

如上所述，预训练的想法早在第二范式便已普及开来。但是不同的是词嵌入捕捉的更多是词义维度上的信息，如何捕捉更多的语言学信息是个大难点。难点如论文所述主要有两点：如何为所谓的可迁移的语言学能力或信息制定目标函数？即我们的优化目标是什么(如语言模型好坏，机器翻译结果或辩论的一致性)？其次学习到的能力将以何种形式迁移？是否需要如ELMO论文里展现的将学到的上下文词嵌入(contextual embeddings) 和输入拼接，并且定制下游模型的架构？或者又如UMLFit那样需要非常细致复杂的训练调整？这两个问题该论文都提出了自己的解答，并在未来不断地被其他第三范式的论文扩展深化。

**2. 模型框架**

模型的训练主要由两个阶段构成（注意这也是第三范式的基础框架，之后的一系列基于TRANSFORMER的大模型预训练如bert/albert/roberta或者transfromer-xl,xl-net等都没有脱离这个框架）：第一阶段是在大量的充足语料上用无监督的方式训练一个语言模型（并且在这个阶段根据优化目标融入各式的语言能力，这点在这篇开山之作里似乎没有体现。但对模型进行多任务的预训练，将十分有利于模型在下游任务里的表现提升。第二阶段是根据下游的任务进行有监督式的微调。之所以叫微调是因为在这个阶段用的数据量远远小于第一阶段，并且基本没有更改模型架构和引入过多新的参数。

第一阶段的优化目标如下所示：

![gpt]({{ '/assets/images/GPT-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}

这是一个相当经典的自回归语言模型, 并且他是生成式(Generative)的无监督方式预训练(Pre-Train)模型。至此GPT名字的由来便完全解释清了。但是如果看过CBOW和SKIP-GRAM论文的同行，可能如我一样，看到这行公式的第一反应便是，如果用一个自回归的仅依赖于前文的滑动上下文窗口建模语言模型，那左右双向的上下文语言模型建模可不可以实现？而事实上BERT论文就是在做这件事，甚至它的名字就是讲这件事。

那么第一阶段的模型架构长什么样呢？答案十分简单，是个多层堆叠的Transformer里的解码器. 如下文所解释：

![gpt2]({{ '/assets/images/GPT-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}

注意有别于基础Transformer用的三角函数来做位置嵌入，该论文用的是可学习的位置矩阵来表征位置信息。在实际应用中，这两种方式似乎效果差别不大。

至此，第一阶段讲解完成。也为上文提到的捕捉更抽象的语言函数应如何制定目标函数的难题提供了一个回答。在后面我将分享的一系列阅读笔记里我们会看到各种各样的优化目标。而第一阶段的简单易懂，也侧面佐证了为什么据闻刚开始时该论文的发表并不顺利。因为虽然他的效果十分惊人并且开始了新的学界范式，但在未出名时这个框架给人的感觉便是创新性不足，犹如一个简单的缝合怪。但真理和科学往往便是这样前进的。

第二阶段的微调是以有监督的形式进行。将文本输入放进第一阶段训练所得的模型，取最后一层的隐层输出，加一个简单的线性层做映射，最后以SOFTMAX转化为概率形式，并顺势得到我们需要最大化的似然函数表达形式。如下所示：

![gpt3]({{ '/assets/images/GPT-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}

这个形式已经简单到了令人发指的地步，该论文或许想要强调该预训练语言模型的强大，所以特意采用了特别简单的微调架构。但事实上，如果想要对下游任务做更好的优化，往往需要更复杂的做法（例如融合先验知识的prompt-tuning, 对特定领域做预训练的TAPT等）。我们紧接着将第二个优化目标L2与第一个优化目标L1以加权形式结合，进行梯度回传。具体来说，两个阶段的架构如下图所示：左边是我们提到的多层堆叠的解码器，模型架构没有任何修改。右边则是根据不同任务简单地添加了个形式稍有不同的线性层。

![gpt4]({{ '/assets/images/GPT-4.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}

至此，第二阶段的大意也在几句话间讲解完毕。最后值得一提的是我们需要对不同任务简单的构造成一个遍历式的单项文本(traversal style). 比如对于QA问答里的多选题，会根据选项多寡如上图所示构造成几个不同的输入文本对，然后单独做处理最后映射成概率。

**3. 实验分析和ablation analysis**

在这个章节里，文章主要提到了两点。第一，文章发现模型的泛化能力与模型堆叠的解码器层数直接相关。并且截至发文，模型对数据集依然欠拟合！即模型依然可以继续扩大继续训练。有意思的是，这直接揭示了未来几年里学界不断比拼算力资源，在越来越大的数据集上用越来越多的参数大力出奇迹的风潮（最主要的就是OPENAI自己，从GPT1的几亿参数到GPT2的十倍参数到GPT3的1750亿参数。一方面虽然思路简单粗暴，但却是一个综合的系统工程。有兴趣可以了解Nvidia-megatron和Microsoft-deepspeed的相关论文了解大规模预训练的工程难点和优化目标）。第二，文章测试了不做任何微调（zero-shot）的情况下，模型在各个任务里的表现以揭示为何大模型的预训练是有益于捕捉语言的本质信息的。例如在做情感分析的时候，在句尾拼接一个VERY，来检测后面的词是POSITIVE的概率高还是NEGATIVE的概率高。作者发现，模型对于各个下游任务的表现随着预训练的阶段不断稳定地提高！这表明第一阶段的预训练里，模型的确学得了泛化的语言能力！并且相同的测试在LSTM上表现远逊色于在transformer上的表现。这表明transformer的确是更有效的架构，并且lstm的高方差显示了其架构的归纳偏置不如transformer. 在随后的剥离测试里，作者发现在抛弃第一阶段的预训练后，模型表现大幅下降。其次用LSTM模型所取得的结果全面逊色于用transformer的结果。

读到这里，是否GPT这篇文章的贡献就结束了？其实不是，我认为在zero-shot里，作者的情感分析用的那个模板方法和PROMPT METHOD有异曲同工之妙！开头我们提到CMU博士刘鹏飞认为自然语言学界有四大范式，那么第四范式是什么呢？即预训练，提示(PROMPT)加预测！这也是接下来第二第三篇GPT论文工作的方向！不同于大家认为的平平无奇，私以为GPT这篇文章不仅开启了第三范式的先河，并且极为开创性地大胆预见了第四范式的存在，并为我们吹响了黎明的号角！

**GPT-2**

在GPT1里面，文章提到未来可以继续拓展的几个方向。其中一个便是模型在ZERO-SHOT的设定下，模型的表现与堆叠的解码器层数有直接的正相关性。这一点在GPT2里面得到了直观的体现，GPT2的模型参数达到了十五亿，是GPT1的十倍大小，而模型的表现的确也取得了长足的进步。第二个拓展的方向便是继续研究大规模语料下的生成式预训练模型为什么对下游任务有帮助。文章认为对单任务单领域的训练是模型缺乏泛化能力的主要原因（这一点在后来的一系列关注元学习的预训练文章里不断被验证。感兴趣地可以去看看上文提到的INSTRUCTION TUNING）。并且进一步认为对于之前的预训练加微调的范式依然不是最优的语言模型状态。他虽然仅需要少量的微调和些许的架构改动，但能不能有一种模型完全不需要对下游任务进行适配就可以表现优异？这篇文章便是在往这个方向努力。这也是为什么文章叫做Language Models are Unsupervised Multitask Learners的原因。

文章（相比于GPT1）的不同主要体现在以下几个方面，首先模型运用了更大规模的新数据集。新数据集是在REDDIT论坛上有人点赞过的文章，他们称为WEBTEXT。其次，文章对GPT1的模型架构进行了微调。具体来说层归一化(LAYER NORMALIZATION)被放到了每一个解码器的前端，并且在最后的一层隐层输出做了一个层归一化。

>关于这一点其实值得展开讲讲。文章只是粗略地提到了这样做的好处，却没有深入分析其中的区别。[这篇文章](https://kexue.fm/archives/8747)说前置层归一化和后置层归一化，对于模型预训练的稳定性和微调有着重要区别。

参数的初始化方式也更改了，把每一个残差链接层的参数按照残差层的个数进行了缩放。此外值得一提的是，在GPT2里，对语句的分词用了与GPT1里不同的方式。在这里他们用了BYTE-LEVEL VERSION OF BYTE PAIR ENCODING. 对字节级别的信息进行编码作为输入。这样基本词汇表就是256个。接下来再对文本语料进行不断融合，找到不同的BYTE-PAIRS。

总结起来，GPT2最主要的贡献是探索了更大规模的模型在ZERO-SHOT的情况下的表现，即没有使用任何微调，仅靠预训练+提示+预测就在8/9个任务里达到了SOTA.


**GPT-3**

GPT3和GPT2相比，延续了一贯的大力出奇迹的思路。继续把模型扩大了百倍以上达到了惊人的1750亿的参数级别！并且继续探索了在不对下游任务进行适配（模型结构更改和参数更新）的情况下，模型的表现。

具体来说，如下图所示，GPT3不做任何FINE-TUNE。只重点考察了在ZERO-SHOT(只有任务描述），ONE-SHOT（任务描述+单个例子）和FEW-SHOT(任务描述+多个例子)的表现。

![gpt5]({{ '/assets/images/GPT-5.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}

具体对于我们从业者来说，我们在该论文里能学到的并不太多。其中可能可以关注的是该文章为了验证模型泛化能力从而设计多个下游任务验证的实验并在相关任务的各个细致测评的细节。另一方面可能是该论文是如何以一种类似PROMPT的方法继续挖掘大模型的潜力的思路。关于PROMPT的一些常见手法，大家可以参见开头提的综述。

以上总结了GPT系列的三篇论文所提出的创新点和概念。从一开始的大模型预训练的引导者，到后面转为在无适配的FEW-SHOT泛化能力方向的引导者。我认为这系列的文章除了我们常调侃的大力出奇迹之外，也有极大的开创性和贡献。


### 4. [iGPT: Generative Pretraining from Pixels](http://proceedings.mlr.press/v119/chen20s/chen20s.pdf)

*Mark Chen, Alec Radford, Rewon Child, Jeff Wu, Heewoo Jun, David Luan, Ilya Sutskever*

*ICML 2020*

**Abstract**

受到NLP领域的非监督representation学习的进展所启发，我们探究了相似的模型是否能够从图像中也学习到有意义的representations。我们训练了一个sequence Transformer来以自回归的方式预测pixels，而不需要有任何这个2D图像结构上的信息。尽管在低分辨率的ImageNet上不使用标签进行训练，我们发现一个GPT-2规模的模型可以学到很强的图片representations，我们使用linear probing，fine-tuning和low-data classification来验证这个representation的表现。在CIFAR-10这个数据集上，我们使用一个linear probe获得了96.3%准确率，比使用监督学习方法的ResNet效果要好。使用fine-tuning的话可以达到99.0%的准确率，和现在最好的监督学习框架类似。


**1. Introduction**

非监督的预训练在深度学习发展过程中扮演了核心角色。从2000年开始，Deep Belief Network，Denoising Autoencoder等就被广泛用在CV和speech recognition领域的神经网络的设计上。研究者们认为能学习到数据分布$$P(X)$$的模型对于监督学习模型$$P(Y \vert X)$$也能提供有效的features。然而，一些高级的技术比如说piecewise linear activation functions，normalization strategies等使得对于很多任务，我们并不需要很好的预训练也能获得很好的结果。还有一些CV研究对于非监督的representations的有效性提出了质疑。这些工作使得主流视觉研究认为sota的结果可以通过直接用模型encode prior structure并且使用大量的监督数据来学习representations来得到。还有些研究甚至表明unsupervised预训练对于模型的效果是有害的。

从而，非监督的预训练在其它的领域繁荣发展起来。其将很多NLP任务的sota结果往前推进了一步。有意思的是，现在的主流非监督representation学习框架，BERT，使用的目标函数是预测被污染的输入数据，很像denoising autoencoders，这是CV最初玩的东西。

作为一个更高维度的，有更多噪音的，以及信息更加冗余的数据格式，图像要比文本更加难以进行generative modeling。最初针对图片的generative pre-training方法已经提出了数十年，而且其对于NLP产生了很大的影响，这类方法值得再重新被考虑，并且和最近的self-supervised方法进行比较。我们重新在图片上衡量generative pre-training方法，现在有了更加灵活的模型结构、更加高效和trackable的训练目标函数和更多的计算资源，generative pre-training方法和其它的self-supervised方法相比也具有了竞争力，在低分辨率的非监督representation learning任务中达到了sota的效果。

有一点值得注意的是：我们的模型使用了dense connectivity的方式，并没有encode图片的2D位置信息，却要比那些encode了的方法效果还要好。

![igpt]({{ '/assets/images/IGPT-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 1. 方法的一个overview。首先，我们预处理raw images，将其resize到一个低的分辨率，并reshape到一个1维的sequence。之后，我们在两个pre-training目标函数里选一个，auto-regressive next pixel prediction或者是masked pixel preduction。最后，我们通过linear probes或者fine-tuning来衡量所学习到的representations的质量。*

**2. Approach**

我们的方法是有一个pre-training stage跟着一个fine-tuning stage。在pre-training stage，我们研究了auto-regressive和BERT里使用的目标函数。我们使用sequence Transformer来预测pixels而不是language tokens。

衡量representation质量的一个方法是为image classification做fine-tuning。fine-tuning在pre-training的基础上在模型上加上一个small classification head，用来优化一个classfication的目标函数从而来改变模型所有的weights。pre-training可以被看作一个良好的initialization或者是一个和early stopping相结合的regularzer。

另一种衡量representation质量的方法使用pre-trained模型作为一个feature extractor。给定有标签的数据$$(X,Y)$$，模型应用在$$X$$上来获得features $$f_X$$。之后一个linear classifier在$$(f_X,Y)$$上被训练。之所以使用linear probing是因为好的features应该能够线性分割开所有的类别。而且，linear probing还将feature的质量和模型结构之间的关系分隔开：在fine-tuning的时候，一个模型可能会因为其模型结构更适合这个下游任务而效果更好，而并不是因为其pre-trained的更好。

我们先介绍对于图片来说，auto-regressive和BERT的目标函数各是什么。然后，说明我们所使用的Transformer decoder的结构细节。最后，我们描述Transformer如何被用于fine-tuning，以及features如何被用于linear probing。


**2.1 Pre-training**

给定一个无标签的数据集$$X$$，里面都是高维数据$$x=(x_1, \cdots, x_n)$$，我们可以选择一个集合$$\left[ 1, \cdots, n \right]$$的permutation $$\pi$$，然后以自回归的方式来预测数据的density $$p(x)$$：

$$p(x) = \Pi_{i=1}^n p(x_{\pi_i} \vert x_{\pi_1}, \cdots, x_{\pi_{i-1}}, \theta)$$

当我们处理图片的时候，我们选择上述的permutation为identity permutation，$$\pi_i = i$$，$$1 \leq i \leq n$$，这也叫做raster order。我们训练一个模型来最小化数据的negative log-likelihood：

$$L_{AR} = \mathbb E_{x \sim X} \left[ -log p(x) \right]$$

我们也考虑了BERT的目标函数，从$$\left[ 1, \cdots, n \right]$$里采样一个子序列$$M$$，每个index有0.15的概率独立的出现在$$M$$中。我们称$$M$$为BERT mask，然后我们训练模型最小化基于没被遮住的数据$$x_{\left[1,\cdots,n\right] \ M}$$的被遮住的数据$$x_{M}$$的negative log-likelihood：

$$L_{BERT} = \mathbb E_{x \sim X} \mathbb E_{M} \Sigma_{i \in M} \left[ - log p(x_i \vert x_{\left[1,\cdots,n\right] \ M}) \right]$$

在pre-training时，我们选择$$L_{AR}$$和$$L_{BERT}$$里的一个作为目标函数来在pre-training数据集上最小化目标函数。


**2.2 Architecture**

Transformer decoder的输入是一个离散tokens的sequence $$x_1, x_2, \cdots, x_n$$，然后对于每个位置输出一个d维的embedding。decoder是$$L$$个blocks堆叠而成的，第$$l$$个输出中间层embedding $$h_1^l, \cdots, h_n^l$$，也是$$d$$维的。我们使用GPT-2（[Language models are unsupervised multitask learners](http://www.persagen.com/files/misc/radford2019language.pdf)）里的Transformer decoder block，其对于输入tensor $$h^l$$计算如下：

$$n^l = layer_norm(h^l)$$

$$a^l = h^l + multihead_attention(n^l)$$

$$h^{l+1} = a^l + mlp(layer_norm(a^l))$$

也就是说，在attention和mlp操作之前都要先经过layer_norm，并且模型的结构是残差链接的结构。我们发现这样的结构使得大型的Transformer也能够很快的实现。

sequence elements之间的仅有的mixing过程发生在attention操作里。为了之前所介绍的auto-regressive目标函数能够被使用正确，我们使用Transformer里类似的方式将sequence里后续的elements遮住。在使用BERT目标函数的时候，我们并不需要attention mask，因为在对输入sequence使用content embeddings之后，$$M$$位置的embeddings都被置零了。

额外的，我们并没有加入位置编码（positional encoding），我们希望模型能学习它们的position embeddings，positions之间任何的位置关系都需要被模型在训练的时候学习到。但是对于使用auto-reggresive目标函数的模型，因为raster order本身就带有顺序信息，所以其实使用这个目标函数某种程度上加入了一些positional encodings。

在Transformer decoder block的最后一层之后，我们使用一个layer norm：$$n^L = layer_norm(h^L)$$，并且学习一个从$$n^L$$到logits的映射，来对于每个sequence element参数化表示其的conditional distribution。当使用BERT目标函数训练的时候，我们直接将被遮住部分的logits忽略就行。


**2.3 Fine-tuning**

在fine-tuning的时候，我们沿着sequence dimension对$$n^L$$进行average pooling，来为每个输入的图片获取一个$$d$$维的vector：

$$f^L = \langle n_i^L \rangle_i$$

也就是说$$n^L$$是一个$$n \times d$$tensor，而沿着sequence维度进行average pooling就是沿着sequence维度计算平均值，从而得到一个$$1 \times d$$的tensor，也就是一个$$d$$维向量，$$f^L$$。

我们学习一个从$$f^L$$到class logits的映射，我们使用的是一个cross entropy loss $$L_{CLF}$$。

尽管对于fine-tuning stage来说，仅仅最小化$$L_{CLF}$$也能为下游任务获取不错的效果，但是从实验来看，同时优化$$L_{GEN}+L_{CLF}$$效果更好，其中$$L_{GEN} \in \lbrace L_{AR}, L_{BERT} \rbrace$$。


**2.4 Linear Probing**

linear probing和fine-tuning的方法差不多，除了一点：average pooling不一定在最后一层出现，而是在中间层出现，$$f^l = \langle n_i^l \rangle_i$$，$$0 \leq l \leq L$$。实验表明，对于linear probing来说，最好的features一般是在中间层。和fine-tuning一样，我们将features映射到class logits。因为在linear probing的时候，features被认为是固定的，所以说这个时候我们只能最小化$$L_{CLF}$$，因为之前的模型参数已经不能再动了。


**3. Discussion and Conclusion**

我们的结果表明generative image modeling是一个学习高质量的非监督图片representations的有效方法。简单的预测pixels任务为很多低分辨率的数据集学习到了sota的representations。对于高分辨率的数据集，我们的模型仍然具有竞争力。

然而，我们的实验仍然说明了还有几个方向可以改进。我们现在使用self-attention机制来建模低分辨率的图片。而其它的自监督模型使用的是基于CNN的encoders，其很容易就能处理高分辨率的图片。而且我们发现我们需要很大的模型才能学习到高质量的representations。iGPT-L和具有差不多效果的其它的模型相比，多了2-3倍的参数。

尽管dense self-attention在NLP领域被广泛使用，但是其计算复杂度以及内存占用率很高，因为其随着sequence长度以平方的方式变化。



### 5. [BEIT: BERT Pre-Training of Image Transformers](https://arxiv.org/pdf/2106.08254.pdf)

[CODE](https://aka.ms/beit)

**Abstract**

我们介绍了一个self-supervised vision representation模型，BEIT，是Bidirectional Encoder representation from Image Transformers的缩写。和NLP领域的BERT论文一样，我们提出一个masked image modeling的任务来预训练vision Transformers。在pre-training的时候，每个图片有两个views，也就是image patches（比如说每个patch是$$16 \times 16$$大小），和visual tokens（比如说离散的tokens）。我们先将原始的图片tokenize维visual tokens。然后随机的遮住某些image patches，然后将其喂给Transformer。pre-training的目标函数是基于被污染的image patches来还原原始的visual tokens。在BEIT进行完pre-training之后，我们在预训练好的encoder之后直接接上task layers来进一步在下游任务上fine-tune模型参数。在image classification和semantic segmentation任务上的实验表明我们的方法达到了sota的效果。


**1. Introduction**

Transformer已经在CV领域获得了令人瞩目的效果（ViT）。然而，实验显示vision Transformers需要比CNN更多的训练数据。为了解决这个需要大量数据的问题，self-supervised pre-training是一个能够使用大规模数据的可靠的方法。有好几个line of research已经在研究vision Transformers了，有constrastive learning（[An Empirical Study of Training Self-Supervised Vision Transformers](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_An_Empirical_Study_of_Training_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf)，[Self-Supervised Learning with Swin Transformers](https://arxiv.org/pdf/2105.04553.pdf)）和self-distillation（[Emerging Properties in Self-Supervised Vision Transformers](https://openaccess.thecvf.com/content/ICCV2021/papers/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf)）。

BERT在NLP领域获得了很大的成功。BERT里的masked language modeling任务首先随机在一个text里mask一部分tokens，然后利用Transformer来encode被污染的text从而恢复这些被mask的tokens。受到BERT的启发，我们将这个denoising auto-encoder的想法用到预训练vision Transformer里，这还没有被vision团体研究过。然而直接将BERT里的预训练照搬到image上是不可行的。首先，对于vision Transformer的输入单元来说，并没有已经存在了的vocabulary。这样我们利用Transformer来预测被mask的tokens，就不能直接使用一个softmax classifier来对所有可能的输出结果计算概率值。而对于NLP来说，language vocabulary是存在的，这样auto-encoding的prediction就可以很容易地进行。一个直接的解决办法就是将这个问题转变为一个regression问题，让模型来预测被遮住patches的原始像素值。然而，这样的pixel-level recovery任务会让模型在预测短程dependency以及高频细节上浪费过多的capacity（[Zero-shot text-to-image generation](http://proceedings.mlr.press/v139/ramesh21a/ramesh21a.pdf)解释了原因）。我们的目的是克服上述vision Transformers的pre-training的困难。

在这篇文章里，我们提出了一个self-supervised vision representation模型，BEIT，Bidirectional Encoder representation from Image Transformers。受到BERT的启发，我们提出一个pre-training任务，叫做maksed image modeling（MIM）。正如fig1所示，MIM使用每张图片的两个views，也就是image patches和visual tokens。我们将图片分为一系列的patches，作为Transformer的input representation。我们将图片tokenize为一系列离散的visual tokens，这是通过discrete VAE的latent codes实现的（[Zero-shot text-to-image generation](http://proceedings.mlr.press/v139/ramesh21a/ramesh21a.pdf)）。在pre-training过程中，我们随机mask image patches里的一部分，然后将剩下的部分喂给Transformer。模型学习的是恢复原图片的visual tokens，而不是原图片被遮住patches的原始像素值。

>但实际上Maksed auto-encoder就是还原的被遮住patches的原始像素值

我们先进行self-supervised learning，然后再两个下游任务上fine-tune已经预训练好的BEIT：image classification和semantic segmentation。实验结果表明BEIT要比其它的self-supervised learning方法的效果好。而且，BEIT是监督预训练的相互补充。BEIT的效果还可以进一步的利用监督数据在ImageNet上微调而变的更好。ablation study表明我们提出的方法对于BEIT能够很好的预训练图片数据都是很重要的。除了在下游任务上的表现，fine-tuning的收敛速度和稳定性也变得好了很多。而且，self-supervised BEIT可以在预训练的时候学习合理的semantic regions。

我们的贡献总结如下：

* 我们提出了一个masked image modeling任务来在self-supervised方式下预训练vision Transformers。我们也从variational autoencoder的角度给出了理论性的解释。
* 我们预训练BEIT然后使用下游任务来fine-tune模型，下游任务包括image classification，semantic segmentation等。
* 我们表明self-supervised BEIT的self-attention机制学习到了semantic regions以及object的boundaries，而且这都是在没有任何标注的情况下完成的。


**2. Methods**

给定一张输入图片$$x$$，BEIT将其encode为contextualized vector representations。如fig1所示，BEIT在self-supervised的方式下使用masked image modeling(MIM)任务来预训练。MIM基于encoding vectors来恢复被遮住的image patches。对于下游任务（比如说image classification，semantic segmentation），我们在预训练好的BEIT上加上task layers然后在特定的数据集上fine-tune模型参数。

**2.1 Image representations**

在我们的方法里，图片有两种representations，image patches和visual tokens。这两个representations作为预训练时候网络的输入和输出形式。

**2.1.1 Image patch**

参照ViT里的做法，2D图片被分割为一系列的patches，从而一个标准的Transformer能够直接将其作为输入。我们将图片$$\pmb x \in \mathbb R^{H \times W \times C}$$ reshape为$$N = HW/P^2$$个patches $$x^p \in \mathbb R^{N \times (P^2C)}$$，其中$$C$$是通道数，$$(H,W)$$是输入图片的分辨率，$$(P,P)$$是每个patch的分辨率。image patches $$\lbrace x_i^p \rbrace_{i=1}^N$$进一步被展平为向量，然后再被linearly projected，和BERT里的word embeddings类似。image patches有着原始像素的信息，作为BEIT的输入features。

在我们的实验里，我们将每张$$224 \times 224$$的照片分为$$14 \times 14$$的grid，从而每个patch大小都为$$16 \times 16$$。

**2.1.2 Visual token**

和NLP里的方法类似，我们通过一个image tokenizer将图片表示为一系列离散tokens组成的sequence。我们将图片$$\pmb x \in \mathbb R^{H \times W \times C}$$ tokenize为$$\pmb z  = \left[ z_1, \cdots, z_N \right] \in \mathcal V^{h \times w}$$，其中vocabulary $$\mathcal V = \lbrace 1, \cdots, \vert \mathcal V \vert \rbrace$$包含离散token的下标。

使用[Zero-shot text-to-image generation](http://proceedings.mlr.press/v139/ramesh21a/ramesh21a.pdf)里的方法，我们通过discrete variational autoencoder（dVAE）来学习这个image tokenizer。在visual token学习的过程中有两个modules，tokenizer和decoder。tokenizer $$q_{\phi}(\pmb z \vert \pmb x)$$将图片$$\pmb x$$利用一个visual codebook（也就是vocabulary）映射到discrete tokens $$\pmb z$$上。decoder $$p_{\psi}(\pmb x \vert \pmb z)$$学习基于visual tokens $$\pmb z$$来恢复原输入$$\pmb x$$。reconstruction目标函数被写为$$\mathbb E_{\pmb z \sim q_{\phi}(\pmb z \vert \pmb x)} \left[ log p_{psi}(\pmb x \vert \pmb z) \right]$$。因为latent visual tokens $$\pmb z$$是离散的，所以模型是non-differentiable的。Gumbel-softmax relaxation被用来训练模型参数。而且在dVAE训练的时候，对于$$q_{\phi}$$使用了一个uniform的prior。

我们将每张图tokenize为一个$$14 \times 14$$的grid（和image patches的时候分的grid大小一样）。visual tokens和image patches的数目是一样的。vocabulary大小为$$\lvert \matchcal V \rvert=8192$$。在这篇文章里，直接使用了[Zero-shot text-to-image generation](http://proceedings.mlr.press/v139/ramesh21a/ramesh21a.pdf)里的公开的[image tokenizer](https://github.com/openai/DALL-E)。


**2.2 Backbone Network: Image Transformer**

正如ViT里一样，我们使用标准的Transformer作为backbone network。从而我们实验的结果就可以和之前的方法直接进行比较了。

Transformer的输入是一系列image patches $$\lbrace \pmb x_{i}^p \rbrace_{i=1}^N$$。这些patches然后经过linearly projection变为patch embeddings，$$\pmb E \pmb x_{i}^p$$，其中$$\pmb E \in \mathbb R^{P^2C) \times D}$$。我们再在输入的sequence前面加一个特殊的token $$\left[ S \right]$$。我们同时也加上了标准的1维可学习的position embeddings到patch embeddings上，$$\pmb E_{pos} \in R^{N \times D}$$。输入向量$$\pmb H_0 = \left[\pmb e_{\left[S\right]}, \left[ \pmb E \pmb x_{1}^p, \cdots, \pmb E \pmb x_{N}^p \right] + \pmb E_{pos} \right]$$被喂给Transformer的encoder。encoder包含$$L$$层Transformer blocks，从而$$\pmb H^l = Transformer(\pmb H^{l-1})$$，其中$$l=1,\cdots,L$$。最后一层的输出向量$$\pmb H^L = \left[ \pmb h_{\left[S\right]}^L, \pmb h_1^L, \cdots, \pmb h_N^L \right]$$作为image patches的encoded representations，其中$$\pmb h_i^L$$是第$$i$$个image patch的representation向量。


**2.3 Pre-Training BEIT: Masked Image Modeling**

我们提出一个masked image modeling (MIM)任务来预训练BEIT。我们随机mask一部分image patches，然后让模型来预测被遮住的那部分patches对应的visual tokens。

![beit]({{ '/assets/images/BEIT-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 1. BEIT pre-training的一个overview。在pre-traininng之前，我们使用autoencoding方式的reconstruction来学习一个image tokenizer，将一个图片根据所学习到的vocabulary来tokenize为一系列离散的visual tokens。在pre-training的过程中，每个图片都有两个views，image patches和visual tokens。我们将image patches的一部分随机遮住（图中的灰色patches就表示被遮住的patches）然后将它们替换为一个特殊的mask embedding $$\left[ M \right]$$。然后将这些image patches喂给一个vision Transformer。pre-training任务的目的是基于被遮住的image patches来预测相对应的visual tokens。*

fig1显示了整个方法的过程。正如2.1里所说，给定一张图片，我们将其分割为$$N$$个image patches $$\lbrace \pmb x_{i}^p \rbrace_{i=1}^N$$，同时也将其tokenize为$$N$$个visual tokens $$\\brace z_i \rbrace_{i=1}^N$$。我们选择随机遮住40%的image patches，其中被遮住的位置记为$$\mathcal M \in \lbrace{1, \cdots, N \rbrace$$。然后我们将被遮住的patches的embeddings换为一个可学习的embedding，$$\pmb e_{\left[M\right]} \in \mathbb R^D$$。被污染的image patches $$x^{\mathcal M} = \lbrace \pmb x_i^p: i \notin \mathcal M \rbrace_{i=1}^N \Cup \lbrace \pmb e_{\left[M\right]}: i \in \mathcal M \rbrace_{i=1}^N$$被喂给一个$$L$$层的Transformer（如2.2里所述）。最后的输出向量$$\lbrace \pmb h_{i}^L \rbrace_{i=1}^N$$被认为是输入image patches的encoded representations。对于每个被遮住的position，$$\lbrace \pmb h_i^L: i \in \mathcal M \rbrace_{i=1}^N$$，我们使用一个softmax classifier来预测相对应的visual tokens：$$p_{MIM}(z^{'} \vert x^{\mathcal M}) = softmax_{z^{'})(\pmb W_c \pmb h_i^L + \pmb b_c)$$，其中$$x^{\mathcal M}$$是输入的corrupted图像，$$\pmb W_c \in \mathbb R^{\vert \mathcal V \vert \times D}$$，$$\pmb b_c \in \mathbb R^{\vert \mathcal V \vert}$$。pre-training的目标函数是最大化在给定corrupted图片的基础上正确的visual tokens $$z_i$$的log-likelihood函数：

$$max \Sigma_{x \in \mathcal D} \mathbb E_{\mathcal M} \left[ \Sigma_{i \in \mathcal M} log p_{MIM}(z_i \vert x^{\mathcal M}) \right]$$

其中$$\mathcal D$$是训练图片组成的集合，$$\mathcal M$$表示随机选取的mask位置，$$x^{\mathcal M}$$表示根据$$\mathcal$$ mask之后的corrupted图片。

MIM任务受到BERT里所使用的maksed language modeling的高度启发，而BERT里使用的MLM是现在NLP领域最成功的预训练目标函数。然而，直接使用pixel-level auto-encoding来做vision pre-training会使得模型过于在意短程dependency和高频细节。BEIT通过预测离散的visual tokens来克服了这个问题，visual tokens将细节总结为更高层次的抽象特征了。ablation study表明我们这种方法要比pixel-level的auto-encoding效果好很多。


**2.4 From the perspective of Variational Autoencoder**

BEIT的预训练可以被看作一个variational autoencoder的训练。$$x$$是原始的输入图片，$$\tilde x$$是被mask后的图片，$$z$$是visual tokens。考虑log-likelihood $$p(x \vert \tilde x)$$的evidence lower bound（ELBO），也就是从corrupted图片中恢复原图片的log-likelihood：

$$\Sigma_{(x_i, \tilde x_i) \in \mathcal D} log p(x_i \vert \tilde x_i) \geq \Sigma_{(x_i, \tilde x_i) \in \mathcal D} (\mathbb E_{z_i \sim q_{\phi}(\pmb z \vert x_i) \left[ log p_{\psi}(x_i \vert z_i) \right] - D_{KL} \left[q_{\phi}(\pmb z \vert x_i), p_{\theta} (\pmb z \vert \tilde x_i) \right]) \tag{1}$$

上述式子右边的第一项是visual token reconstruction的目标函数，$$q_{\phi}(z \vert x)$$表示的是用来获取visual tokens的image tokenizer；$$p_{\psi}(x \vert z)$$表示的是给定visual tokens我们能获取到的原输入图片的函数；$$p_{\theta}(z \vert \tilde x)$$表示的是基于masked的images来获取visual tokens的函数，也就是我们的MIM任务代表的函数。

我们通过两步来学习这个模型。首先，我们获取image tokenizer，将其当作一个discrete variational autoencoder。这一步里我们最小化reconstruction loss $$-\mathbb E_{z_i \sim q_{\phi}(\pmb z \vert x_i) \left[ log p_{\psi}(x_i \vert z_i)\right]$$，使用的是一个uniform prior。然后，我们控制$$q_{\phi}$$和$$p_{\psi}$$不变，来学习prior $$p_{\theta}$$。我们将$$q_{\phi}(\pmb z \vert x_i)$$使用一个可能性最大的visual token简要表示为一个one-point distribution：$$\hat z_i = argmax_z q_{\phi}(z \vert x_i)$$。从而，公式1右侧可以写为：

$$\Sigma_{(x_i, \tilde x_i) \in \mathcal D} (\mathbb E_{z_i \sim q_{\phi}(\pmb z \vert x_i) \left[ log p_{\psi}(x_i \vert z_i) \right] + logp_{\theta}(\hat z_i \vert \tilde x_i)) \tag{2}$$

其中第一项是visual token reconstruction，而第二项是masked image modelling，也就是我们的BERT预训练的目标函数。


**2.5 Pre-training setup**

BEIT的模型结构和ViT里的一样。我们使用一个12层的Transformer，hidden size是768，并且有12个attention heads。feed-forward网络的中间层大小为3072。我们使用$$16 \times 16$$的image patch大小。我们直接使用[Zero-shot text-to-image generation](http://proceedings.mlr.press/v139/ramesh21a/ramesh21a.pdf)里训练好的image tokenizer。visual tokens的vocabulary大小为8192。

我们在ImageNet-1K上预训练BEIT，其有大约120万张图片。我们还使用了random resized cropping，horizontal flipping，color jittering等方法来做数据增强。我们使用$$224 \times 224$$大小的输入分辨率。从而visual patches和visual tokens的数目都是$$14 \times 14$$。我们最多mask 40%的patches。

预训练会运行800个epoches，batch size为2000。optimizer使用的是Adam。learning rate是1.5e-3，10个epoches用来热身，使用的是cosine learning rate decay。weight decay是0.05。训练使用16张V100显卡大约进行了5天时间。


**2.6 Fine-tuning BEIT on downstream vision tasks**

在BEIT上预训练之后，我们在Transformer上加一个task layer，然后再下游任务上进一步微调模型参数，如同BERT里的方法一样。我们使用image classification和semantic segmentation作为下游任务。

*Image classification*

对于image classification这个任务，我们直接是用一个简单的linear classifier作为task layer。我们先使用average pooling来aggregate representations，然后将aggregated之后的global representation喂给一个softmax classifier。然后category probabilities就被计算为：$$softmax(avg(\lbrace \pmb h_i^L \rbrace_{i=1}^N \pmb W_c))$$，其中$$\pmb h_i^L$$是第$$i$$个image patch的对应的encoding向量，$$\pmb W_c \in \mathbb R^{D \times C}$$是projection矩阵，$$C$$是labels的个数。我们maximize正确标签数据的likelihood就可以fine-tuneBEIT和softmax classifier的参数。

*Semantic segmentation*

对于semantic segmentation，我们使用预训练好的BEIT作为backbone encoder，然后使用几个deconvolution层作为decoder来产生segmentation。这个模型也是end-to-end被fine-tuned的。


**3. Conclusion**

在这篇文章里，我们描述了vision Transformer的一个self-supervised预训练的框架，在下游任务上获得了很好的效果，包括image classification，semantic segmentation等等。我们所提出的方法类似于BERT，但是是应用在图片上的。

在将来，我们希望之后的工作基于以下几个方向：

* 设计具有scalability的BEIT预训练模型，从而使得BEIT预训练达到BERT预训练在NLP领域的效果
* 为多模态预训练设计一个统一的方法，从而可以使用相似的目标函数和相似的模型来对texts和images共同进行预训练。


### 6. [An Image is worth 16 $$\times$$ 16 workds: Transformers for image recognition at scale](https://openreview.net/forum?id=YicbFdNTTy)

还有一个[Arxiv版本](https://arxiv.org/pdf/2010.11929.pdf)，内容略有不同。

*Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenbron, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby*

*ICLR 2021 Oral*

>这篇文章是2021年里CV领域最重要的文章，它打破了自从2012年AlexNet提出以来CNN在CV领域的统治地位，其将一个NLP领域的模型Transformer应用到CV领域，在大规模非监督数据学习的基础上也能取得和CNN甚至超过CNN的预训练效果。而且ViT的出现打破了NLP和CV领域使用不同结构模型的壁垒，从而使得多模态也找到了新的方向。跟随ViT之后的研究层出不穷，已经有几百篇了，有将其拓展到别的任务的，有对模型本身进行改进的，有对模型进行分析的，还有对目标函数或者训练方式进行改进的。ViT开启了CV的一个新时代。现在几乎所有的CV任务，基于ViT的模型都取得了最好的结果，比如ImageNet上的classification就是用传统ViT达到了目前最好的效果，而Swim Transformer（ICCV 2021最佳论文）在COCO数据集上的目标检测任务也达到了最佳效果。而且ViT的模型结构以及特性和CNN有很大的不同，所以是很重要的开创性的工作。

**Abstract**

当Transformer架构已经成为NLP任务的标准架构时，Transformer在CV领域的应用仍然没有太大的发展。在视觉领域，attention要么和CNN一起被使用，要么就修改CNN结构里的卷积将其替换为attention机制，但是CNN的整体结构是没有改变的（比如说ResNet有很多层，我们只是将每层里面的卷积操作给替换成自注意力机制，而并没有改变大的模型层次结构）。我们将会表明完全依赖CNN是没有必要的，一个应用在图像patches上的Transformer在image classification任务上就可以表现得很好。当在大量的数据上预训练之后再转移到一些针对具体下游任务的中小型数据集上（比如IamgeNet，CIFAR-100，VTAB等）时，Vision Transformer（ViT）具有和CNN sota相媲美的效果。而且Vision Transformer需要更少的训练资源（但实际上计算量还是很大的）。


**1. Introduction**

self-attention架构，特别是Transformers，在NLP领域现在是最主流的。现在最主要的方法是在一个大的text corpus上进行预训练，然后再在特定任务的数据集上fine-tune。多亏了Transformer架构计算上的高效性和可扩展性（scalability），我们可以训练很大的Transformer架构的模型（超过1000亿个参数）。随着模型和数据集的增长，目前还没有看到性能饱和（也就是过拟合）的发生。

>将Transformer结构直接应用到图像领域的难处。Transformer的输入是一个序列，然后利用自注意力机制计算序列里两两元素的关系。所以第一个难点是如何将一个2D的图片表示为一个1D的序列。最直观的方法就是将每个像素点都当作一个元素，直接将图片拉直为一个序列送进Transformer里计算。但是这样做的话序列长度太长，计算复杂度太高（目前NLP领域的Transformer的输入序列长度大多也就是几百，如果直接将一个$$224 \times 224$$的图片拉直输入，那计算复杂度太高了，目前无法计算）。

在CV领域，卷积架构仍然占据着主导地位，比如AlexNet，LeNet，ResNet等。受到NLP领域成功的启发，一些工作开始尝试将CNN架构和self-attention机制结合起来，[Non-local Neural Networks](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf)（这篇文章的想法是既然直接将图片拉直送人Transformer计算复杂度太高，那就将中间层的feature map拉直送入Transformer，一般来说对于CNN，feature map的大小是会随着网络深度的增加而减小的，如果一个$$14 \times 14$$的feature map拉直，长度也就只有196，如果channels个数也不多，那这个计算复杂度还是可以接受的），[End-to-end object detection with Transformers](https://arxiv.org/pdf/2005.12872.pdf)。有些工作尝试将CNN里的卷积计算换掉，[Stand-alone self-attention in vision models](https://arxiv.org/pdf/1906.05909.pdf)（这篇文章的想法是既然整张图拉直计算复杂度太高，那就只将某个窗口内的图拉直送入Transformer，这样的想法就有点像CNN的做法了，因为CNN也是每个卷积操作只关注局部信息），[Stand-alone axial-attention for panoptic segmentation](https://arxiv.org/pdf/2003.07853.pdf)（这篇文章的做法是将2D的图片当作两个1D的向量的乘积，然后在这两个1D的向量上分别做Transformer，这样计算量就大大降低了）。这些将CNN里的卷积替换掉的模型，虽然理论上很高效，但是还并不能被高效用在目前硬件设备上。因此，在大规模的image recognition任务上，经典的类ResNet架构还是sota。

>自注意力早就在CV领域被研究了，而且甚至使用自注意力完全替代卷积操作。所以这篇文章就从如何设计一个具有可扩展性的大规模的应用在CV领域的Transformer架构这个角度出发来写的。

被Transformer在NLP领域的scalability的成功所启发，我们尝试直接将一个标准的Transformer模型应用在image上，做尽可能少的改动。为了做到这点，我们将一张图片分为patches，然后将这些patches的linear embeddings排成sequence作为Transformer的输入。image patches被当作NLP里的tokens（words）。我们以监督学习的方式来在image classification任务上训练这个模型。

>强调以监督方式来训练模型是因为Transformer在NLP领域基本上都是以非监督的方式进行训练的，要么就是单方向的generative language modelling（GPT），要么就是双方向的masked language modelling（BERT）。

当没有使用很强的regularization，在中等大小的数据集上训练时，比如说ImageNet，Transformer的结果比ResNet的结果要低几个百分点。这似乎说明Transformer在image上这样使用的效果并不好：Transformers并没有CNN固有的一些归纳偏置（inductive biases），比如translation equivariance和locality，因此在有限的数据集上训练的模型泛化效果并不好。

>归纳偏置（inductive bias）实际上就是先验知识，也就是我们已经知道的假设。CNN的两个归纳偏置，locality指的是相邻的区域一般具有相似的特征，靠得越近的区域相关性就越强。另一个归纳偏置叫做平移等变性（translation equivariance），写成公式表示就是$$f(g(x)) = g(f(x))$$，也就是先做$$f$$操作还是先做$$g$$操作，结果是不变的。将$$f$$理解为卷积，而$$g$$理解为平移。因为卷积操作相当于将一个模板沿着图片位置滑动，所以说即使图片内的物体进行了平移，并不改变计算的结果。

所以CNN有着这两个很强的归纳偏置，而Transformer什么也没有，所有的知识都需要Transformer自己去数据里学。

但是，当我们在更大的数据集上训练的时候（1400万-3亿张图片），情况就有所变化了。Transformer在images上的效果要比很多任务的sota效果要好。


>这个introduction的写法还是很好的，首先说明Transformer在NLP领域应用的效果很好，而且目前都还没有饱和的现象，那这么好的模型应用到CV领域会怎么样呢？然后说明这个方向肯定已经有很多人研究过了，但是这些研究要么就是将自注意力机制和CNN结合起来，要么就是替换CNN里的卷积操作为自注意力操作，并没有改变整体的结构。从而说明前面那些工作和这篇文章的本质上的区别。之后再引入这篇文章里提出的Vision Transformer。最后一段介绍一下很好的效果。


**2. Related Work**

Transformer是用来做machine translation而被提出的，现在已经在很多NLP任务里是sota的架构了。大型的Transformer-based模型通常在大的corpora上预训练，然后在具体任务上微调。BERT使用了一种denoising的self-supervised预训练任务，而GPT line of work使用language modeling（基于之前的词或者句子预测下一个词或者句子）作为其预训练任务。

naive的直接将self-attention机制用到images上就会变成让每个pixel来注意其它的pixels。这会造成pixel数量平方级别的计算，这scale到实际任务是不现实的。因此，为了将Transformer用到image领域，很多研究者尝试了一些近似的办法。[]()对于每个pixel，只将self-attention应用在其邻域内的pixels上，而不是考虑所有的pixels。这样的local multi-head dot-product self-attention blocks可以完全替代convolutions操作。

最接近我们这篇文章的是这篇文章，[On the relationship between self-attention and convolutional layers](https://arxiv.org/pdf/1911.03584.pdf)，其从image中获取$$2 \times 2$$的patches，并在这些patches上应用self-attention。这篇工作和我们这篇文章非常的相似，但是我们的工作更进一步的阐明大规模的预训练使得Transformers和sota的CNN框架效果差不多甚至更好。进一步的，这篇文章使用了较小的patches，只有$$2 \times 2$$，使得模型更加适用于小分辨率的图片，而我们的方法对于中等分辨率的图片仍然可以适用。

还有很多工作尝试将CNN和self-attention相结合，比如，[]()将CNN得到的feature map和self-attention得到的feature map结合起来用于image classification，或者说在CNN得到的feature maps上再继续用self-attention得到更好的features，在object detection，video processing，image classification，unsupervised object discovery和unified text-vision tasks上都有使用。

另一个相关的工作是image GPT (iGPT) [Generative pretraining from pixels](http://proceedings.mlr.press/v119/chen20s/chen20s.pdf)，其在减小了图片的分辨率和color space之后再在图片pixels上使用Transformers。模型是以一种非监督学习的方式被当做一个generative model来训练的，所得到的representation之后可以被fine-tune，也可以直接被用来做classification，在ImageNet上的classification有72%的准确率。这个准确率要比这篇文章提出的ViT的准确率要低不少。

>实际上在CV领域，一般来说生成模型都会比判别模型效果差。但MAE这篇文章就做到了生成模型要比判别模型的效果要好，而且迁移到其它任务上（比如目标检测）进行微调之后效果也很好。

>这个related work写的非常全面彻底。


**3. Method**

在模型设计上我们尽可能使用原始的Transformer的设计。因为其已经有了很成熟的代码库和高效实现了。

![1]({{ '/assets/images/VIT-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 1. 模型总览图（Model Overview）。 我们将图片分为固定数量的patches，将其每个都进行linearly embedding，再加上position embedding是，喂给一个标准的Transformer encoder。为了能够实现classification，我们在sequence之前加了一个多余的可学习的classification token。*

**3.1 Vision Transformer (ViT)**

整个模型的样子如fig1所示。标准的Transformer的输入是一个1维的sequence，由token embeddings组成。为了处理2维图片，我们将图片$$\pmb{x} \in R^{H \times W \times C}$$ reshape到一个由展平的2D patches构成的sequence $$\pmb{x_p} \in R^{N \times (P^2 \dot C)}$$，其中$$(H,W)$$是原图片的分辨率，$$C$$是原图片的通道数，$$(P,P)$$是每个patch的分辨率，$$N = HW/P^2$$是patches的数量，其也是sequence的长度。Transformer在它的层之间使用的都是一个不变的latent vector size $$D$$，所以我们将patches展平，并使用一个可训练的线性投射（公式1）将其映射到一个$$D$$维的向量。我们将这个patch经过投射之后得到的$$D$$维向量成为patch embedding。

和BERT里的class token类似，我们在上述patch embeddings构成的sequence的头部加上一个可学习的embedding，$$z_0^0 = x_{class}$$，其在经过$$L$$个Transformer模块（也就是层）之后的值$$z_L^0$$用来表示image representation $$\pmb y$$，如公式4所示。在pre-training和fine-tuning的过程中，$$z_L^0$$都和一个classification head相连，这个classification head在pre-training的时候是具有一个隐层的MLP，而在fine-tuning的时候只是一层线性层。

position embeddings也被加入到patch embeddings当中，为了保留位置信息。我们使用标准的可学习的1D positional embeddings（也就是BERT里用的那种位置编码），因为更复杂的2D positional embeddings并没带来什么效果提升。现在这个embedding vectors sequence可以作为Transformer encoder的输入了。

>比如说输入的图片是$$224 \times 224 \times 3$$大小，选择的patch大小为$$16 \times 16$$，这样的话，就会有$$224 / 16 = 14$$，一共$$14 \times 14 = 196$$个image patches，而每个patch的大小为$$16 \times 16 \times 3$$。然后将每个patch展平，就是一个$$16 \times 16 \times 3 = 768$$大小的向量。然后使用一个大小为$$768 \times 768$$的projection矩阵，$$E$$，将这$$196$$个长度为$$768$$的向量进行线性投射。从而得到了大小为$$196 \times 768$$的tokens的embeddings，再加上开头的特殊的$$\left[ CLS \right]$$token的embedding，就是$$197 \times 768$$的token embeddings。最后再加上positional encodings，我们的positional encodings也是一个$$197 \times 768$$的矩阵，其中每一行代表每个位置的positional encoding，这个矩阵也是可以学的。所以最终输入Transformer的东西，就是一个$$197 \times 768$$的矩阵。
>接下来就是输入Transformer的encoder里，encoder会有$$L$$个blocks，对于每个block，首先经过layer norm，不改变tensor的大小，然后对于ViT-base来说，其自注意力机制的heads个数为12，所以会将$$197 \times 768$$先利用12个线性投射，获得12个大小为$$197 \times 768 / 12 = 197 \times 64$$大小的tensors，这些tensors分别作为自注意力里的$$q,k,v$$进行自注意力计算，得到了12个大小为$$197 \times 64$$的输出，然后再concatenate起来，重新得到了$$197 \times 768$$大小的tensor，进行残差连接后再经过一次layer norm，然后经过一个mlp，一般扩大4倍，从而变成$$197 \times 3072$$，然后再计算回来，变回$$197 \times 768$$，再进行一次残差链接，从而输出。这样重复进行$$L$$个block，最终输出结果。

Transformer encoder有multiheaded self-attention层（MSA）（公式2）、MLP blocks（公式3）以及每层计算完都会进行LayerNorm，并且在每个block之间还有residual连接。

$$\pmb{z_0} = \left[x_{class}; x_p^1 \pmb{E}; x_p^2 \pmb{E}; \cdots ; x_p^N \pmb{E} \right] + \pmb{E_{pos}} \tag{1}$$

其中$$\pmb{E} \in \mathbb{R}^{(P^2 \dot C) \times D}, \pmb{E_{pos}} \in \mathbb{R}^{(N+1) \times D}$$

$$\pmb{z_l^{'}} = MSA(LN(\pmb{z_{l-1}^{'}})) + \pmb{z_{l-1}^{'}} \tag{2}$$

其中$$l=1, \cdots, L$$。

$$\pmb{z_l} = MSA(LN(\pmb{z_l^{'}})) + \pmb{z_l^{'}} \tag{3}$$

其中$$l=1, \cdots, L$$。

$$\pmb y = MLP(z_L^0) \tag{4}$$

*归纳偏置（Inductive bias）*

我们发现vision transformer相比于CNN少了很多image-specific inductive bias。在CNN里，locality，二维的领域结构以及translation equivariance在模型的每一层里都存在。而在ViT里，只有MLP是local和translation equivariant的，self-attention层是global的。ViT除了在一开始的位置编码用了图片这个数据的2维特征之外，就再也没有使用任何图片的归纳偏置了，也就是说并没有用任何这个输入数据是2维图片的domain knowledge。而且位置编码也是随机初始化，然后在训练过程中学习到的，并没有携带任何2D信息。

>正是因为ViT并没有用到domain knowledge，所有的知识都得自己学，所以其在中小数据集上效果不如CNN，但是在大型数据集上，其学习全局知识的优势就能体现出来了。所以说CNN模型比较data efficient，而ViT模型更加的general。

*Hybrid Architecture*

作为原始图片patches的一个替代，输入Transformer模型的sequence也可以从一个CNN的feature map获得。在这个hybrid模型里，patch embedding projection $$\pmb E$$被用在从feature map上获取的patches上。


**3.2 Fine-tuning and higher resolution**

我们在大型数据集上预训练ViT，之后再在下游任务的小数据集上微调。在微调的时候，我们将pre=trained prediction head移除，加上一个初始化为0的$$D \times K$$的feedforward层，其中$$K$$表示的是下游任务的classes。一般来说，在分辨率更高的图片上微调效果会更好，这个时候我们保持patch大小不变，这样就会有更多的patches，ViT可以处理长度变化的patch embedding sequence，所以并没有问题，但这个时候预训练的positional encoding可能就不好使了，我们可以根据预训练的positional encoding embeddings在图片中的位置，使用2D插值来计算高分辨率图片的positional encoding embeddings。如果使用了插值来进行新的位置编码的计算，那这里也算是引入了图片的2D信息的归纳偏置了。


**4. Experiments**

主要对比了ResNet，ViT和它们的混合模型的representation学习能力。为了了解每个模型想要学习好到底需要多少数据，他们在不同大小的数据集上进行预训练，然后在很多的下游任务数据集上进行测试。当考虑到预训练的计算代价的时候，ViT的表现很好，其能在很短的时间内在很多下游的任务上表现很好。

最后文章还做了自监督的实验，也就是不用监督数据，而使用自监督数据进行预训练，这个方法还是比较有潜力的。

>文章使用自监督方法的效果并没有使用监督方法的预训练效果好，然而后续很火的MAE就是沿着这条研究方向，使用自监督方法预训练模型达到甚至超过了ViT监督方法的效果。

**4.1 Setup**

*数据集*

预训练使用的数据集有ImageNet-1k，也就是有1000个类的ImageNet数据集，大约130万张图片，ImageNet-21k，也就是有21000类的ImageNet数据集，大约1400万张图片，还有Google的JFT数据集，18000类，大约3亿张图片。而下游任务使用的是分类任务，使用很常见的CIFAR，Oxford-pet，Oxford-flower这些数据集。

*模型变体*

一共使用了三种模型，和Transformer那篇文章对应的，也就是ViT-base，ViT-Large和ViT-Huge。而且patch的大小也和模型结构有关。

**4.2 Pre-training data requirement**

![5]({{ '/assets/images/VIT-5.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 2. 图片里方块就表示的是各种ResNet，比如最靠左侧的一列，下面的方块可能就是ResNet52，上面的方块就是ResNet152，从而图中灰色的区域就表示了ResNet所能达到的效果范围。而从图中可以看出来，对于小的数据集ImageNet-1k，ViT的效果全面不如ResNet。因为Transformer没有归纳偏置，需要更多的数据来学习的更好。而在ImageNet-21k这个数据集上，ViT和ResNet的效果就差不多了。而在JFT上训练的时候，ViT的效果比ResNet全面要好。*

这个图表示的意思就是，如果你的数据集比较小的话，那还是ResNet的效果会比较好，如果想用ViT的话，至少得准备ImageNet-21k那么大的数据集。如果你有较大的数据集的话，用ViT的效果会跟好一些，而且ViT的训练相对来说会快很多。

**4.3 Inspecting vision transformer**

为了理解vision transformer是如何处理图片数据的，我们来分析其内部的representations。vision transformer的第一层是将展平的patches线性投射到一个满足transformer尺寸的embedding上（也就是$$E$$），也就是一个更低维度的空间里。fig3左侧展示了所学习到的embedding filters的几个最主要的部分。这些部分有点像表示patches里的低维结构信息的basis functions。

>我们可以看到，第一层，也就是$$E$$学到的特征很像那些gabor filter，和CNN第一层所学习到的特征很像，都是颜色、纹理特征。

在这个linearly projection之后，一个可学习的position embedding被加在patch representation上。fig3中间显示的是模型尝试去学习encode图片中patches之间的distances，和positional encoding类似，也就是相近的patches就有类似的positional encodings。而且，能看出来显示出了行列的结构，同一行/列的patches有着类似的embeddings。

![2]({{ '/assets/images/VIT-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 3. 左侧：RGB embedding filters（初始的28个主成分）。中间：position embeddings。右边：head的注意区域大小和网络深度的关系。每个点表示的是一层里16个head中的某个head的mean attention distance accross images。*

self-attention允许ViT在整张图片上整合全局信息，即使对于很低的层也是这样。我们来看看这个网络到底将这个特性实现了多少。基于attention weights，我们来计算在图片中的mean attention distance accross images，见fig3右侧，也就是每个点能关注到周围多大的区域。这个attention distance和CNN里的receptive field有点像。对于这张图来说，使用的是ViT-L/16，只有24个层，所以说横轴是24，而每层都有16个heads，也就是每一列都是16个圆点。纵轴的mean attention distance的计算就是，对于两个pixels，我们计算这两个pixels所在的patches之间的自注意力的weights乘以这两个pixels之间的距离，再将所有的这样的pixels对对应的这个值加起来。我们发现，有些heads在很浅的层就已经能注意到距离很远的pixels了，表明模型在一开始确实就能注意到整张图的全局信息了。而随着网络越来越深，所有的heads的注意的pixels之间的距离都很远了，就意味着网络学习到了语义概念，而不是根据临近的像素点进行判断。

我们还发现，模型会注意到那些有助于classification的图片区域，如fig4所示。

![3]({{ '/assets/images/VIT-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 4.*

**4.4 使用自监督方式预训练ViT**

仿照BERT里的masked language modelling的设计，对于图片来说，先将其分割为patches，然后将一些patches mask掉，利用其它的patches来还原被mask的patches。但是这样的方法预训练出来的效果和表现最好的监督方式预训练出来的效果还是差了4个百分点。作者还提到了还可以利用constrastive learning的方式来进行自监督预训练ViT，其可以是未来的研究方向。

>因为constrastive learning是2020年CV领域最火的自监督方法。MoCo v3和DINO都是使用constrastive learning的方式来训练的ViT。

**5. Ablation Analysis**

**5.1 关于使用$$\left[CLS\right]$$ token还是使用全部sequence token进行average pooling效果的对比

对于CNN来说，比如说ResNet50，再最后一层卷积层得到的feature map大小比如说是$$14 \times 14 \times 128$$，这个时候要进入classifier了，就可以使用global average pooling，在$$H$$和$$W$$的维度上都降为1，也就是说在average pooling的filter大小为$$14 \times 14$$，从而得到了一个$$128$$维的向量，就可以方便的连接几层MLP，再传入softmax做分类了。上述global average pooling的操作相当于flatten的操作。而在ViT里，为了做后续的classification，我们添加了一个$$\left[CLS \right]$$的token，然后利用这个token最后的输出向量，接上MLP和softmax来进行分类判断。其实也可以不使用这个$$\left[CLS\right]$$token，而直接对于输入的image patches序列tokens的输出，进行一波average pooling（沿着sequence的维度），从而将一个sequence的tokens的features，压缩到一个token的feature，再接上MLP和softmax来进行分类判断。文中使用前一种方法只是因为BERT使用的是这个方法，文章也对这两种方法的结果进行了对比，实验证明效果差不多。

但是要注意，这两种情况下学习率是不一样的，如果使用同样的学习率，会导致效果不好，见下图：

![4]({{ '/assets/images/VIT-4.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 5. class token和global average pooling classifiers效果的对比，效果差不多，但得使用不同的learning rate。*


**5.2 不同的positional encoding对于实验效果的影响**

有三种不同的positional encoding方法：

* 1D的位置编码，也就是BERT里使用的方法，也就是假如有$$14 \times 14=196$$个patches，那么就利用$$196 \times D$$大小的矩阵来表示这些patches的positional encoding，其中$$D$$就是输入embedding的维度，对于上面的例子来说就是768。
* 2D的位置编码，因为图片patches其实是在图片里有2D位置信息的，这样对于$$14 \times 14=196$$个patches，先用$$196 \times D/2$$大小的矩阵来表示其横向位置信息，然后再用$$196 \times D/2$$大小的矩阵来表示纵向位置信息，再将这两个矩阵concatenate起来。
* 相对的位置编码

实验表明，这三种位置编码最后导致的实验效果是差不多的。


**6. Conclusion**

我们探索了将Transformer用在image recognition领域的可行性。和以往使用self-attention的方法不同，除了在一开始切割image patches和positional encoding的时候用了一些图像的归纳偏置，我们并不为架构引入image-specific inductive biases。也就是我们不需要对vision领域有什么domain knowledge，直接将一张图片理解为patches的sequence，然后用标准的Transformer来处理。但这简单的，但是scable的方法和在大数据集上预训练结合起来效果很好。因此，Vision Transformer达到了或者超过一些image classification的sota结果。

虽然结果很鼓舞人，但是还有很多问题没有解决。一个是如何将ViT应用到其它的CV任务上，比如说detection（DETR就做了这个工作）和segmentation。而且，如何将文章中的large scale supervised预训练变成self-supervised预训练也是很重要的。

>实际上MAE就做到了。


### 7. [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/pdf/2111.06377.pdf)

*Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, Ross Girshick*

*[CVPR 2022 Oral]*

**Transformer $$\rightarrow$$ Bert(使用了Transformer架构，但是使用self-supervised的学习方法), Transformer $$\rightarrow$$ ViT(在CV中使用Transformer架构), ViT $$\rightarrow$$ MAE (正如Bert和Transformer的关系一样，MAE就是使用self-supervised的学习方法的CV中的Transformer架构)**

**Transformer $$\rightarrow$$ Bert $$\rightarrow$$ MAE $$\leftarrow$$ ViT $$\leftarrow$$ Transformer**

**Title**

标题里经常用两个词，scalable和efficient。如果做的算法很快，就可以是efficient，或者real-time，如果做的模型很大的话，就可以是scalable。vision learners说明不依赖于具体的模型种类，是一个backbone就行。autoencoders里的auto表明的是“自”的意思，不是“自动”的意思。“自”模型的特点就是标号$$y$$和样本$$x$$来自于同一个东西，比如说在NLP里，用前后的词来预测词，用来预测的标号和样本都是词，是同一种东西，所以是“自”模型（在Transformer里提到了这个是auto-aggressive模型）。在NLP里，因为“自”模型比较常见，所以只用encoder而省略前面的auto也可以，但在CV里，标号来自于图片本身的情况较少，所以需要加上auto，强调一下和其他工作的区别。

这个标题的格式，在GPT里也用到了，"a" are "b" "c", 其中"a" and "c"是名词，"b"是形容词。这种句式实际上把结论放在了标题里。而且这种说法比较客观。

**Abstract**

这篇文章说明masked autoencoders(MAE)对于CV来说是一个scalable的self-supervised learner。我们的MAE方法很简单：将输入图片的随机的patch遮起来，然后重构这些被遮住的地方。这个模型基于两个核心设计。首先，我们开发了一个非对称的encoder-decoder结构，encoder仅仅在那些可以看到的image patches上操作（并没有mask tokens的信息），而decoder从encoder给出的可见的image patches上学到的latent representations结合mask tokens来重构原图片。decoder的设计是很轻量级的。其次，我们发现，将原输入图片遮住很大一部分，比如说75%，会使得问题变成一个不那么简单并且有意义的self-supervisory任务（如果只遮住一小部分，那模型可能学会插值就能获得不错的效果，但如果遮住很大一部分，就会迫使模型学习那些更好的特征）。将这两点结合起来，使得我们能够高效且准确的训练大型模型。这个模型主要是用作迁移学习的，是一个预训练模型，可以用于一系列下游的CV任务。


**Figures**

![MAE]({{ '/assets/images/MAE-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 1. MAE architecture。在pre-training过程中，很大一部分随机挑选的image patches会被遮住（比如75%）。从而encoder实际上只需要对很小一部分的image patches进行编码。mask tokens在encoder编码之后被用进来和encoder所得到的features一起按顺序排列好传给一个轻量级的decoder来重构原始输入图片。在pre-training之后，decoder就不需要了，encoder就可以直接被用于获取图片的有效特征（这个时候就不需要遮盖了，这个encoder就是一个特征提取器），用于下游任务。*

![IMAGENET-TEST]({{ '/assets/images/MAE-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 2. 在ImageNet验证集上测试MAE的效果。这些用于测试的图片都是没有参预训练的。左侧一列是随机去掉80%的image patches部分的图片，中间是MAE重构的效果，右边一列是真实的原图片。*

![COCO-TEST]({{ '/assets/images/MAE-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 3. 在COCO验证集上测试MAE的效果，使用的MAE模型是在ImageNet上训练的（也就是和figure2里用的模型参数一样）。注意到最右侧两个例子，虽然说还原的不对，但是语义上能说得通。*

![PORTION-TEST]({{ '/assets/images/MAE-4.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 4. 使用遮盖比例75%的数据集来训练的MAE在ImageNet验证集上的重构效果，这时候我们采用了更高的遮盖比例。这说明我们的方法可以泛化。*


**Discussion and Conclusion**

简单的并且可以scale的方法是整个深度学习的核心。在NLP里，简单的self-supervised方法最近很火。但在CV领域，现有的pre-training框架还主要是监督的（也有一些self-supervised的工作进展）。在这片文中丽，我们发现autoencoder就可以做出一个很好的pre-training框架。现在，我们没准可以像在NLP里一样成功的在CV领域使用self-supervised方法了。

但是，我们要注意到images和languages是有天然区别的两类signals，而且这样的区别一定要被谨慎对待。images仅仅是被记录的光亮强度，并没有像language一样可以被语义分割为一个个的单词。我们并没有尝试在image中遮住objects，而是随机的遮挡住patches，这些patches并不一定会构成有意义的语义块。同样的，MAE也是重构像素值，而并不是重构有语义的块。但是，实验表明我们的模型仍然能够推出复杂的重构，这表明其学习到了很多视觉概念，也就是语义。我们认为者是因为MAE内部的hidden representations涵盖了这些语义。这可能是将来的研究方向。


**1. Introduction**

深度学习目睹了一系列capacity和capability增长的模型架构的诞生。这样越来越大的模型就需要越来越多的数据，特别是带有标签的数据。

这个问题最近在NLP领域被self-supervised pre-training方法的提出而解决了。他们的方法，基于GPT里的autoaggressive语言模型以及BERT里的masked autoencoding，实际上概念上很简单：将原数据一部分移除掉，然后让模型来学习预测这部分被移除掉的数据。这些方法现在让训练有数千亿参数的可以泛化的NLP模型都变得可行了。

masked autoencoders，是一种更加general的denosing autoencoders（在图片中人为添加噪音，然后通过去噪的方式来对图片的特征进行更好的提取），这个想法在CV领域也不那么新鲜。BERT在NLP领域获得了很大的成功，但是autoencoding方法在视觉领域的应用距离NLP里的成功还很远。于是我们提出了问题：是什么使得在vision和language里使用maksed autoencoding不一样呢？我们希望能从以下几个角度来回答。

1）直到最近，NLP和CV的模型结构都不一样。在视觉领域，CNN是最主流的模型。卷积是在regular grids上操作的，而且将mask tokens或者positional embeddings这种indicators加进去不是那么的容易。而这种模型结构上的差异，在ViT提出之后就不是问题了。

2）language和vision之间的information density不一样。languages是人类产生的信号，其具有高度的语义性，而且information-dense。让一个模型对一个句子只需要能够预测若干个缺失的单词，这个模型就需要有很复杂的language理解能力。但images是natural signals，而且有着很大的spatial redundancy，比如说，一个缺失的patch可以只需要有一些高层的很简单的信息就可以被其领域的patches重构回来。为了克服这个不同点，并且鼓励模型学习有用的features，我们说明一个很简单的策略在CV里很有效：随机遮挡很高比例的patches。这个策略很大程度上减少了redundancy，而且构造了一个具有挑战性的self-supervisory任务，其需要对图片有整体性的理解而不仅仅是low-level的image statistics。从之前的fig 2-4可以看出MAE的重构效果。

3）autoencoder的decoder，其的作用是将encoder给出的latent reprensentation再映射回输入，在language和image两个领域的重构任务也不一样。在视觉领域，decoder重构的是像素，所以其输出是低语义层次的，比普通的识别任务输出的语义层次要低。而对于language，deocder预测的是缺失的words，其含有丰富的语义信息。在BERT里，decoder可以随便选（BERT里就是一个MLP），但是在images领域，decoder的设计决定了encoder所学习到的latent representation的语义层次的高度。

由这些想法所启发，我们为visual representation learning提出了MAE这个模型。我们的MAE对于输入的图片随机遮盖patches，并且在像素空间内重构这些被遮盖的patches。MAE有一个非对称的encoder-decoder的设计。我们的encoder仅仅在那些可见的patches上操作（并没有mask tokens），而decoder是轻量级的，从encoder给出的latent representation和mask tokens一起来重构输入，如fig1所示。这样的设计大大减小了模型的计算量。在这种设计下，一个非常高比例遮挡的策略可以获得双赢：不仅让encoder能学习到更有效地representation，其还只需要encode一小部分patches。这可以大大减小训练的时间以及训练占用的内存，从而可以使MAE方法scale到更大规模的模型上。

我们的MAE学习到泛化效果很好的高capacity的模型。使用MAE pre-training，我们可以仅仅使用ImageNet-1K数据集就训练那些十分需要数据的模型，比如说ViT-Large/-Huage（[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/pdf?id=YicbFdNTTy)），并且泛化效果还更好。我们同时还在object detection，instance segmentation，semantic segmentation等后续任务上测试了MAE在transfer learning领域的效果。都要比利用其它的方法包括监督方法进行预训练的效果要好。这些结论和在NLP里发现的结果差不多，所以希望autoencoder在CV领域也能有在NLP领域那样的发展。


**2. Related works**

*Masked language modeling*

Masked语言模型和它们的autoregressive counterparts，比如BERT和GPT，在NLP领域是非常成功的pre-training模型。这些方法遮住输入sequence的一部分然后训练模型来预测这部分。这些方法的scaliability也很好，而且这些模型产生的预训练的representations对于下游任务来说泛化效果也很好。


*Autoencoding*

autoencoding是一个学习representations的经典方法。其有一个encoder来将输入映射到一个latent representation上，并有一个decoder来重构这个输入。比如说，PCA和k-means就是autoencoders。Denoising autoencoders（DAE）是一类autoencoders，其先给输入加上噪音，再学习如何重构原始的，没有被污染的输入。一系列方法都可以看作使用不同的噪音的DAE，比如，masking pixels，或者移除color channels等。我们的MAE也是一种DAE，但是和经典的DAE有着很大的不同。


*Masked Image Encoding*

masked image encoding方法通过遮盖的方式污染图片，然后再来学习图片的representations。最早的工作是将masking当作一种噪声，所以是DAE的一种特殊情况。context encoder使用CNN来学习那些缺失的部分。由NLP里的成功所启发，最近的方法都基于Transformers。iGPT在pixels构成的sequence上操作，像GPT一样预测被遮住的pixels。


*Self-supervised Learning*

非监督学习的方法在CV领域非常火热，通常集中研究于各种pre-training的任务。最近，contrastive learning非常的流行，[A Simple Framework for Contrastive Learning of Visual Representations](http://proceedings.mlr.press/v119/chen20j.html)，[Momentum Contrast for Unsupervised Visual Representation Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)，[Representation Learning with Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf?fbclid=IwAR2G_jEkb54YSIvN0uY7JbW9kfhogUq9KhKrmHuXPi34KYOE8L5LD1RGPTo)，[Unsupervised Feature Learning via Non-Parametric Instance Discrimination](https://openaccess.thecvf.com/content_cvpr_2018/html/Wu_Unsupervised_Feature_Learning_CVPR_2018_paper.html)，其在两张或者多张图片间对相似性进行表示。contrastive learning的方法太过于依赖data augmentation。autoencoding采用了概念上不同的另一个方向，也显示出了不一样的效果。


**3. Approach**

我们的MAE是一个简单的autoencoding的方法，其在给定原输入部分的patches的情况下重构原输入。和所有的autoencoders一样，我们的方法有一个encoder来将观察到的数据映射到一个latent representation，也有一个decoder来从这个latent representation重构原始的输入。和经典的autoencoders不一样，我们采用一种非对称的设计，encoder只在那些没有被遮住的patches上操作，不考虑mask tokens，并且使用一个轻量化的decoder，从mask tokens和latent representations来重构原始的输入。fig1描述了整个流程。

*Masking*

和ViT里的方法一样，我们先将图片分成regular的不重合的patches。然后我们随机挑选出一些patches，然后把剩下的都mask掉。我们的patch采样方式很简单：不重复的随机挑选，使用的是均匀分布（也就是每个patch被选中的概率相等）。

具有很高的遮挡比例的随机采样很大程度上消去了图片的redundancy，因此创建了一个任务，其不能够简单的只是通过领域patches插值就能解决。我们采用的是uniform distribution，其可以避免潜在的center bias（也就是更倾向于遮住图像中心的区域）。而且，这样很高比例的遮挡可以使我们能够scale我们的encoder到很大的模型，因为输入给encoder的计算量大大减小了。

*MAE encoder*

我们的encoder就是一个ViT，但是输入仅仅是那些没有被遮住的patches。正如在标准的ViT模型里一样，我们的encoder通过一个线性projection加上positional embeddings来embed patches，然后再用一系列Transformer模块来处理。然而，我们的encoder只需要在一小部分的输入图片patches上进行计算，而且并没有使用mask tokens。这使得我们训练很大的模型和很大的数据集成为可能。

*MAE decoder*

MAE的decoder的输入是full set of tokens，包括（1）encoded没被遮挡的patches；（2）mask tokens。由fig1可见。每个mask token是一个shared，learned向量，也就是每个被遮住的patch都用同一个向量来表示，而这个向量的值也是通过学习得到的。我们给这个集合里所有的tokens都加入了positional encodings；如果没有这个，mask tokens就没有它们在image中的位置信息了。decoder也有另外一些Transformer blocks。

MAE的decoder只用于在pre-training的时候做image reconstruction任务，只有encoder在生成后续所用的representation时会被用到。因此，decoder的结构可以被灵活的设计，和encoder的结构可以没有关系。

*Reconstruction target*

我们的MAE通过预测每个masked patch的像素值来重构输入。decoder的输出的每个element都是一个向量的像素值，表示的是一个patch。decoder的最后一层是一个linear projection，其输出的通道数等于每个patch里像素的个数。decoder的输出被reshape成为一个重构的image。我们的loss计算的是重构的image和原始输入之间的MSE。

我们还研究了另一个reconstruction target，也就是每个masked patch的normalized像素值。也就是说，我们计算一个patch里的所有像素的mean和deviation，然后normalize这个patch。使用normalized pixels能够改善encoder所学习到的representation的质量。

*Simple implementation*

我们的MAE pre-training可以被高效的部署。首先，我们对于每个input patch都生成一个token（这个token也就是通过linear projection和一个added positional embedding得到的）。然后我们随机打乱tokens再排成一列，直接去除掉这列后面那部分tokens（去除比例就是masking ratio）。这个过程为encoder产生了一小部分tokens（也就是没被遮住的那部分），也就等价于不重复的随机挑选patches。在encoding之后，我们将encoded patches构成的list通过加上mask tokens来延长，然后再unshuffle整个list（因为之前为了随机选patches而shuffle了，现在逆过来这个过程），从而tokens就对应了它们正确的位置。decoder被应用到这个list上（加上了positional embeddings）。

**4. The writing style of this paper**

这篇文章的introduction很长。第一是因为他使用了大量的图片，这是好事。第二个原因是因为他使用了一种问问题，回答问题，再引出论文想法的写法，这种写法是很好的。如果不这么写的话，那就是将abstract进行扩充，那其实最后就得到了这里introduction最后的两段话。因为这篇文章的技术细节并不复杂，所以作者采用了一种回到更本质的问题的方法，也就是将Bert从NLP用到CV的时候会有什么问题，然后再一一回答，之后引出MAE这个算法。也就是将这个算法为什么要设计成这样说的很清楚。这种写法是很推荐的。因为论文除了要讲清楚怎么做的，还要讲清楚为什么要这么做，也就是motivation。

但是在related work部分，虽然提到了很多相关工作，但是对于iGPT和Beit这两篇相关性最大的论文，并没有详细介绍他们的方法，这是不推荐的。


### 8. [MIST: Multiple Instance Spatial Transformer](https://openaccess.thecvf.com/content/CVPR2021/html/Angles_MIST_Multiple_Instance_Spatial_Transformer_CVPR_2021_paper.html)

*CVPR 2021*



## Contrastive Learning Models

### 1. [MoCo: Momentum Contrast for Unsupervised Visual Representation Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)

*Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick*

*CVPR 2020 Best Paper Candidate*

>视觉领域里使用对比学习（contrastive learning）里程碑式的工作。对比学习从2019年开始就是一个很热的方向，其给从2017年开始就很卷的CV领域带来了一剂强心针。出现了一大批优秀的工作，MoCo就是其中之一。MoCo作为一个无监督的representation learning的工作，不仅在分类任务上逼近了有监督的baseline模型的效果，而且在很多主流的视觉任务上，比如说检测、分割、人体关键点检测等，都超越了有监督预训练的模型（也就是在ImageNet上预训练的模型）。所以说MoCo的出现，证明无监督学习真的可以，我们可能并不需要大规模的有标记的数据来做预训练。

>现在在NLP领域，很多大模型，比如说GPT，BERT等，都是通过自监督的方式来预训练得到的，而在CV领域也快了。

[CODE](https://github.com/facebookresearch/moco)

MoCo这篇论文是假设你已经对对比学习有一定的了解了，所以我们需要先对对比学习以及前人的工作做一些简单的介绍。如果对之前的工作不是很了解的话，就不能理解为什么MoCo里要用动量对比学习（momentum contrast），就无法体会到MoCo的精妙之处。

假设我们有三张图片，第一张图片是一个有着高兴表情的人，第二张图片是一个有着悲伤表情的人，而第三张图片是一只狗。我们希望模型能学习到前两张图片是一个类别，而第三张图片和前两张图片不是一个类别。所以说对比学习，顾名思义就是对比着学习。模型并不需要知道图片的真实标签，也就是不需要知道前两张图片是人，第三张图片是狗，只需要知道前两张图片类似，而第三张图片和前两张图片不类似。我们的模型对于这三张图片得到了对应的features，$$f_1, f_2, f_3$$，我们希望在特征空间里，$$f_1$$和$$f_2$$尽可能地接近，而和$$f_3$$尽可能的远离。如果我们的模型可以做到所有类似的图片的特征都在这个特征空间相近的区域内，那我们的目的也就达到了，我们所学习到的特征就会是一个很好的特征。

但是虽然上述操作并不需要知道图片的具体标签，可还是需要判断哪些图片是类似的，哪些图片不是。在CV领域，我们一般设计一些巧妙地代理任务（pretext task），来人为定义一些规则，这些规则可以用来定义哪些图片是相似的，哪些图片是不相似的。从而可以提供这样一个“监督信号”去训练模型。这也就是所谓的自监督训练。一个最常见的代理任务就是instance discrimination。假设我们有一个没有标注的图片数据集$$X$$，里面有$$N$$张照片，记为$$X_1, X_2, \cdots, X_N$$，那我们如何定义哪些图片是相似的，哪些图片是不相似的呢？instance discrimination是这样处理的。对于任意一张图片$$X_i$$，使用某种数据增强操作，比如说随即裁剪，可以得到新的图片，假设通过变换1（$$T_1$$）得到了图片$$X_i^1$$，通过变换2（$$T_2$$）得到了图片$$X_i^2$$，那么将这两张图片叫做正样本，而这个数据集里任意的其他照片$$X_j$$，$$j \neq i$$都被认为是负样本。也就是说instance discrimination这个任务认为每张图片自成一类，剩下的所有图片都和他不是一个类的。以ImageNet数据集举例的话，那现在就不是有1000个类了，而是有100多万个类，因为每张图片都是一个类。在定义好了正样本和负样本之后，我们利用模型得到输入图片的特征，然后利用一些对比学习的目标函数来训练这个模型，使得正样本之间的特征尽可能的接近，正样本和负样本之间的特征尽可能的远离，这样的目标函数可以是NCE loss等。

对比学习看似简单，但是具有很强的灵活性。我们只需要找到一种方式来定义什么是正样本，什么是负样本，接下来的操作都是类似的。这样的话，就可以采用各种方式来定义正样本和负样本了。比如说在视频领域，可以认为一个视频里的任意两帧都是正样本，而其它视频里的任意一帧都是负样本。比如说在NLP领域，[SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/pdf/2104.08821.pdf)这篇论文，将同样的句子喂给模型，做两次forward，每次forward的模型的dropout都不一样，从而得到的是不一样的特征，就认为这两个特征是正样本，而其它句子得到的特征都是负样本。比如说在CV领域，[Contrastive multiview coding](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560749.pdf)这篇论文你里认为一个物体的不同的view是正样本，比如说一个物体的RGB图像和深度图像也可以被认为是正样本。

从上面的叙述可以看出来，对比学习十分的灵活，只要你脑洞够大，合理的定义正样本和负样本，那就可以使用对比学习。所以后来就拓展到了多模态领域，也就出现了OpenAI提出的CLIP模型：[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020.pdf)。

已经有了对比学习的简要介绍和了解了instance discrimination之后，就可以来介绍MoCo这篇论文了。

**title**

momentum contrast说明了论文使用的方法是对比学习，而且使用了动量这个新颖的做法，后面的unsupervised visual representation learning说明其解决的是视觉表征学习的任务，而且是用一种无监督的方式进行的。对于动量来说，可以理解为$$y_t = m y_{t-1} + (1-m) x_t$$，$$m$$是超参数，$$y_{t-1}$$是上一个时刻的输出，$$y_t$$是这个时刻的输出，$$x_t$$是当前时刻的输入。也就是说不想让当前时刻的输出完全依赖当前时刻的输入，也依赖于之前时刻的输出，也就是有一种惯性。


**authors**

Facebook的大佬，五个人在Google scholar上的引用量加起来有50万了。


**Abstract**

我们为视觉表征学习提出了momentum contrast（MoCo）。通过将contrastive learning看作一个字典查询的任务（dictionary look-up），我们建立了一个动态字典（dynamic dictionary）。由两部分组成：一个队列（queue）和一个移动平均的编码器（moving-averaged encoder）。队列里的样本不需要做梯度回传，所以就可以往队列里放很多负样本，从而使得这个字典变得很大，使用移动平均编码器是为了使得字典里的特征尽量保持一致。在训练过程中，如果有一个很大的而且特征一致的字典，会对无监督的对比学习非常有好处。（这篇论文主要的亮点在于结果，所以摘要剩下很大的篇幅留给了结果）。对于在ImageNet上的classification任务，采用linear protocol的方法进行测试的话，MoCo能取得和之前的无监督方法差不多或者更好的效果。（这个linear protocol就是，如果我们先预训练好的一个backbone网络，现在要把它用到不同的数据集上的时候，我们将它的骨干网络冻住，只去学习最后的那个全连接层，也就是那个分类头，这样就相当于我们将一个预训练好的模型当作了一个特征提取器，我们只需要这个模型来抽取特征，这样就可以证明之前预训练好的模型所学习到的特征到底好不好。）更重要的是，MoCo学习到的特征能够很好的迁移到下游任务（这是这篇文章的精髓，也是这篇文章最大的卖点，因为我们之所以想做大规模的预训练，就是希望模型能够学习到一个很好的特征，而且这个特征具有很好的迁移性，我们就可以在一些没有那么多标注的下游任务上获得很好的结果。）MoCo能够在七个下游任务上：在数据集PASCAL，VOC，COCO和其它数据集上进行detection/segmentation，都比有监督的预训练模型效果要好，有的还要好很多。（无监督和有监督的预训练模型的框架都是一样的，比如说ResNet50，而只是训练的方式不一样。）这说明无监督和有监督representation learning之间的鸿沟对于很多视觉任务来说已经不存在了。

>在此之前也有使用无监督比有监督效果好的模型，但那都是针对某种特殊的下游任务，而MoCo是第一个在这么多的下游任务上都能让无监督的预训练比有监督的预训练效果好的模型。


**1. Introduction**

在NLP领域，非监督的表征学习十分的成功，比如说GPT，BERT等。但是在CV领域，有监督的预训练还是占据了主导地位。CV和NLP领域有这样的差异的原因可能是因为它们signal spaces的差异。对于NLP的任务来说，它们的signal spaces是离散的（words，sub-word units等等），从而可以很容易的建立一些tokenized的字典（tokenize的意思就是将某个word对应成某个特征），从而无监督学习就可以基于字典很容易的展开（因为可以简单的将字典里的所有的key都看成类别，这就是有监督学习的范式了，就有了类似于标签的东西去帮助学习）。对于CV来说就不一样了，因为CV任务的signal spaces是一个连续的，高维的空间，而且不像words那样有很强的语义信息而且浓缩的那么好，并不简洁，从而导致其并不适合去建立这样一个字典。没有这个字典，无监督学习就很难去建模了，从而在CV领域，无监督学习的效果还远不如有监督学习。

最近有一些无监督学习的方法是基于对比学习的，而且取得了不错的效果。虽然这些方法的出发点或者说做法都不一样，但是它们都可以被归纳为一种做法，也就是说构造一个动态字典（dynamic dictionary）。回到我们之前所说的instance discrimination。我们随机选一张图片，$$X_1$$，这张图片经过了不同的变换之后我们得到了$$X_1^1$$和$$X_1^2$$，这两张图片就组成了一个正样本对。一般我们将$$X_1^1$$称为锚点、基准点（anchor），将$$X_1^2$$称为positive，也就是相对于$$X_1^1$$来说，$$X_1^2$$是正样本，剩下的所有的图片$$X_2, \cdots, X_n$$都是负样本，叫做negative。接下来就是将这些样本喂给编码器，得到特征输出。将$$X_1^1$$喂给编码器$$E_1$$得到特征$$f_{11}$$，图片$$X_1^2$$喂给编码器$$E_2$$就得到了特征$$f_{12}$$，编码器$$E_1$$和编码器$$E_2$$可以是同一个模型，也可以是不同的模型，在MoCo这篇论文里用的是不同的模型。对于那些负样本，我们应该用$$E_2$$这个编码器来得到特征，因为所有的负样本，和正样本$$X_1^2$$都是相对于锚点$$X_1^1$$来说的，所以这些负样本也应该和正样本使用同样的编码器，从而它们的特征保持一致性。这些负样本得到的特征记为$$f_2, \cdots, f_n$$。对比学习就是让$$f_{11}$$和$$f_{12}$$在特征空间里尽可能的相近，而使得负样本的特征尽可能的远离$$f_{11}$$。如果将$$f_{12}, f_2, \cdots, f_n$$当成一个字典，也就是说其是利用编码器对原图像学习得来的。将$$f_{11}$$当成是一个query，将字典里的特征当成key，那么对比学习就变成了一个字典查询的问题。也就是说对比学习就是训练一个编码器来进行字典查找：让query尽可能的和其正样本的key相似，而和负样本的keys都远离。从而整个学习过程就是一个对比学习的框架，从而只需要去最小化一个对比学习的目标函数就可以了。在MoCo这篇论文里，他将对比学习的方法归纳为一个动态字典的问题，所以他不用anchor，positive，negative这些词，而使用query，key这些词，所以上述的$$X_1^1$$用$$X^q$$表示，而$$X_1^2$$用$$X^k$$表示。而特征$$f_{11}$$则用$$q$$表示，剩下的字典里的特征则是$$k_0, k_1, \cdots, k_n$$。整个过程如下图所示：

![1]({{ '/assets/images/MOCO-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}

从将对比学习当作动态字典的角度来看，我们猜想如果要有好的结果，那么这个字典需要有两个特性：（1）字典需要够大；（2）在训练的时候保持尽可能的一致性。因为只有当这个字典够大，我们就能更好的从这个连续的、高维的视觉空间里抽样。因为这个字典里的key越多，其所能表达的视觉信息就越丰富，当拿一个query去和这些key做对比的时候，就更有可能能学到那些将物体区分开的特征，那些更加本质的特征。如果这个字典很小，那模型可能就会学到一个捷径（shortcut solution），从而导致预训练好的模型不能够很好的泛化。对于第二点，字典里的这些key都应该利用相同或者相似的encoder来得到，这样在和query做对比的时候才能保证这个对比尽可能的一致，否则如果这些key是由不同的encoder得到，那么query可能就会找到那个和它使用相同或者相似的encoder的那个key，而不是具有相同语义信息的那个key，这样就是变相的引入了一个shortcut solution，从而模型学不好。现有的方法都至少被以上两个方面的某一方面所限制。

我们提出了MoCo（momentum contrast），是为了给无监督的对比学习构造一个又大又一致的字典，模型总览图（model overview）如fig1所示。

![2]({{ '/assets/images/MOCO-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*fig 1. MoCo通过利用contrastive loss将一个encoded的query和一个字典的encoded keys匹配的任务来训练一个visual representation encoder。字典的keys，$$\lbrace k_0, k_1, \cdots \rbrace$$是在训练过程中定义的一系列数据。字典被构造为了一个队列（queue），目前的batch会被加入队列，而最先进入的batch会被移出队列。keys是被一个缓慢更新的encoder来encode的，更新是由momentum方式完成的。上述方法使得使用一个大的且consistent的字典来学习visual representation成为可能。*

整个流程和之前所说的对比学习的一般框架是很像的，最大的区别就是queue和momentum encoder，这也是这篇文章最大的创新点。

为什么要用一个队列来表示这个字典呢？主要还是受限于显卡的内存。所以需要想一个办法，使得字典的大小和每次forward所需要的batch size的大小没有关系。作者想到了利用队列来巧妙解决这个问题。也就是说这个队列可以很大，但每次更新这个队列是一点一点进行的，每次加入一个新的batch的keys，将最先放入的一个batch的keys移除，这样就不需要在每个iteration的时候都使用到所有的字典里的keys，也就是说不需要将整个字典都放到内存里了，从而可以有一个很大的字典而不会内存溢出。但是之前说了，最好字典里的所有的keys都是一致的，也就是从相似的或者相同的encoder里得到的，而如果像上述描述的那样，每次加入一个新的batch的keys，每次这些keys都是由当前的encoder得出来的，而encoder也在更新，那这些keys不就不一致了嘛。所以作者提出了第二个改进，也就是momentum encoder。如果用$$\theta_q$$表示产生query的编码器的参数，用$$\theta_k$$表示产生keys的编码器的参数，那么$$\theta_k = m\theta_{k-1} + (1-m) \theta_q$$。也就是说，$$\theta_k$$刚开始是由$$\theta_q$$初始化而来的，但是如果我们选取了一个很大的动量$$m$$，那么$$\theta_k$$其实更新的是很缓慢的，从而保证了这个字典里的keys都是由相似的encoder得到的，尽最大可能保证了它们的一致性。基于上述两点贡献，MoCo这个方法可以构造一个又大又一致的字典，从而无监督的去学习visual representations。

接下来需要来看该选择什么代理任务作为自监督信号来进行模型训练了。MoCo只是提供了一种为对比学习提供动态字典的方式，而代理任务是可以灵活选取的。在这篇文章里，作者选用了instance discrimination任务，也就是之前所介绍的那个，也就是一个query和一个key如果是一张图片的不同views（比如说不同裁剪），那么称它们是一对正样本。使用这个代理任务，MoCo在ImageNet数据集上的linear classification任务能和其它的方法效果差不多。

无监督学习的一个主要的目的就是在大规模的数据集上进行预训练之后，能够迁移到一系列下游任务上。MoCo就做到了这一点。在7个下游任务上，MoCo这种无监督预训练的模型都能超越在ImageNet上进行有监督预训练的方式。而我们对于无监督预训练还有另外的期待，也就是希望如果模型更大，数据更多，那么效果会一直提升，也就是没有饱和的状态。为了验证这点，作者在又10亿张图片的数据集上也用MoCo来做了预训练，最后的结果还能够提升。上述结果证明MoCo可以对于很多视觉任务，减小甚至消除无监督学习方法和有监督学习方法之间的差异，在一些任务上还可以取代广泛使用的在ImageNet上利用有监督方法进行的预训练模型。


**2. Related work**

无监督学习/自监督学习一般有两个方向可以做：在代理任务上做文章，或者在目标函数上做文章。代理任务一般指的是那些大家不太感兴趣的任务，并不是分类、分割、检测等这些有实际应用场景的任务，这些代理任务的提出，主要是为了学习一个好的特征。而目标函数的研究是可以和代理任务的研究分开的。MoCo主要就是研究的目标函数的改进。MoCo提出的这个又大又一致的字典实际上影响的是后面InforNCE这个目标函数的计算。

*Loss functions*

最常见的构造目标函数的方式就是衡量模型的输出和它应该得到的固定的目标之间的差距，比如说对于auto-encoder，输入是一张原图或者一张被污染的图，希望通过encoder-decoder的方式将这张原图重建出来，这里既可以用$$L_1$$ loss也可以用$$L_2$$ loss。总之，衡量的是原输入和新建数据之间的差异，这种就属于生成式网络的做法，因为模型生成了一些数据（比如一张图片）。如果想使用判别式网络，比如说将一张图片分为9块，然后让模型能够预测每两块随机挑选的块之间的相对位置（有8种可能）。除了生成式和判别式两种目标函数以外，还有我们这里要介绍的其它的形式。

最著名的就是对比学习的目标函数，另外一个就是对抗型的目标函数。对比学习的目标函数就是在一个特征空间里，衡量各个样本对之间的相似性。它想达到的目标就是对于相似物体之间的特征尽量的近，对于不同物体之间的特征尽量的远。对比学习的目标函数和之前提到的生成式或者判别式的目标函数的最大的区别是，之前所提到的两种目标函数都是一个固定的目标，但是对比学习的目标函数是在训练的过程中不断地改变的，目标函数实际上是由encoder抽取的特征而决定的，也就是这篇文章指的字典。最近一些不错的非监督学习的方法使用的都是对比学习的方式，将在3.1里详谈。

对抗型的目标函数主要衡量的是两个概率分布之间的差异。这个目标函数主要是用来做无监督的数据生成的。但后来也有一些对抗性的方法被用来做representation Learning了。因为如果模型能够生成很好很真实的图片的话，那模型应该已经学习到了数据的分布，那模型的特征也应该是不错的。


*Pretext tasks*

过去几年有很多论文提出了各种形式的代理任务，比如说denoising auto-encoder，context auto-encoder（重建某个patch），colorization等等。


*Contrastive learning vs. pretext tasks*

不同的代理任务适合不同类型的对比学习的loss。


>为什么related work要从代理任务和目标函数来写相关工作呢？因为对于有监督学习来说，输入$$x$$，模型输出$$y$$，然后通过某种目标函数和ground truth相比较，从而学习模型。而对于非监督学习，就没有ground truth，从而就需要设计代理任务来自己找到某种监督信号，而有了监督信号之后，我们还需要设计相应的目标函数来更好地训练模型。所以说这两个方面是非监督学习最重要的方面。


**3. Method**

**3.1 Contrastive learning as distionary look-up**

对比学习，以及它的一些变体，都可以被认为是训练一个编码器来进行一个字典查找的任务。

假设我们已经有了一个编码好的query $$q$$，以及一系列已经编码好的样本$$\lbrace k_0, k_1, \cdots \rbrace$$（它们就是一个字典的keys）。假设在这个字典里，只有一个key（$$k_{+}$$）是和这个query配对的，也就是说$$q$$和$$k_{+}$$互为正样本对。然后我们就需要设计一个目标函数，这个目标函数应该要满足，当$$q$$和$$k_{+}$$相似的时候，loss值比较低，以及$$q$$和其它的keys的值不相近的时候，loss的值也比较低。我们利用向量之间的内积来表示它们之间的相似性（越大越相似），这篇论文采用了InfoNCE这样一个loss函数：

$$\mathcal L_q = -log \frac{exp(q \cdot k_{+} / \tau)}{\Sigma_{i=0}^K exp(q \cdot k_i / \tau)} \tag{1}$$

其中$$\tau$$是个超参数，用来控制分布形状的（更加peak或者更加平滑）。实际上上述公式和计算softmax的cross entropy loss一模一样，区别只是在于对于cross entropy loss，$$K$$指的是分类的类别数，而对于InfoNCE，$$K$$指的是负样本的个数，也就是字典的大小减1。上述公式1的分母实际上是一个正样本和$$K$$个负样本的和。实际上InfoNCE也就是做一个$$(K+1)$$类的分类任务，将$$q$$分到$$k_{+}$$这个类。

既然我们已经有了代理任务，也有了目标函数，现在就该讨论一下输入和模型是什么样的了。总体来说，query的representation是一个输入$$x^q$$通过一个编码器$$f_q$$所得到的，类似的，key的representation是一个输入$$x^k$$通过一个编码器$$f_k$$得到的。而$$x^q$$，$$x^k$$以及$$f_q$$，$$f_k$$具体是什么，是由代理任务决定的。输入$$x^q$$和$$x^k$$可以是图片、patches或者一系列带有上下文信息的patches。$$f_q$$和$$f_k$$可以相同，也可以部分参数共享，或者完全不同。


**3.2 Momentum Contrast**

从以上的这些角度来说，对比学习是一种在连续的高维信号上（比如图片）去构建离散的字典的一种方式。这个字典是动态的，因为字典里的key都是随机采样的，而且用来encode这些keys的编码器也是在训练过程中不断地改变的。作者认为，如果想通过非监督的对比学习学习到一个好的特征，那么这个动态字典需要大，而且具有一致性，因为大的字典能包含很多不同的负样本，从而能学习到更具有判别性的那些特征，而一致性是为了避免模型学习到一个shortcut solution。

基于这些研究动机，作者提出了MoCo。

**3.2.1 Dictionary as a queue**

我们方法的核心就是将一个字典用队列的方式表示出来。用队列的好处是我们可以重复利用那些已经encode好的keys，而这些keys是那些之前放进队列的mini-batch里的keys。这样就可以使得字典的大小和mini-batch的大小完全没有关系了，从而在训练的时候就可以只设置mini-batch的大小，而字典的大小很灵活，可以作为一个超参数被设置。

字典里的keys都一直在不断的被更新，新的mini-batch里的keys被放进来，而最初放入队列的mini-batch的keys被移除去。这个字典一直都是所有数据的一个子集，因为我们计算对比学习的目标函数InfoNCE的时候，只是取一个近似，而不是在整个数据集上计算一个loss。使用队列这种数据结构，可以使得维护字典的计算开销非常小。之所以使用队列来表示这个字典，是因为其具有先进先出的特性，从而每次移出的都是最先进入的mini-batch，而这部分keys和现在的keys差别是最大的了（因为encoder一直在更新），所以这样操作对于保持一致性是最好的了。

**3.2.2 Momentum update**

使用队列来表示字典可以使得字典很大，但是这样的话我们就没法使用对于队列里所有的keys进行梯度回传了，也就是说key的encoder没办法使用反向传播的方式来更新它的参数。一个简单的方案就是每个iteration结束后，直接将$$f_q$$的结果赋值给$$f_k$$。这样操作确实很简单，但是效果并不好。作者猜想效果不好的原因是因为$$f_q$$是一个变化很快的编码器，这样会造成字典里的keys的不一致性。我们提出一个momentum update来解决这个问题。

我们将key的encoder和query的encoder分别记为$$f_k$$和$$f_q$$，其参数分别记为$$\theta_k$$和$$\theta_q$$，那么$$\theta_k$$的更新方式如下：

$$\theta_{k+1} \leftarrow m \theta_k + (1 - m) \theta_q$$

这里$$0 \leq m < 1$$是momentum系数。

从而虽然字典里的keys是由不同的encoder产生的，但是因为这些encoders之间的差异很小，所以整个字典的keys的一致性还是很强的。作者实验中使用的是$$m=0.999$$而且比$$m=0.9$$效果要好得多，这个动量系数使得encoder的变化其实是非常缓慢的。

**3.2.3 Relations to previous mechanisms**

这里是填上之前introduction里第三段没说完的内容。之前的对比学习方法都可以看作是字典查找，但他们都或多或少受到字典大小或者字典一致性问题的影响，从而效果不好。这里就详细解释一下之前的方法是如何受限的，以及MoCo又是如何解决这些局限性的。

之前的对比学习方法可以分为两种架构，由fig2所示。

![3]({{ '/assets/images/MOCO-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*fig 2.*

第一种就是end-to-end的方式，由fig2 (a)所示。顾名思义，这种方式下的query和key的编码器都是可以通过梯度回传来更新参数的，但是这种方式的局限性在于字典的大小和mini-batch的大小是一样的，所以说我们就不能有很大的字典。SimCLR就是这里所说的end-to-end的方式，但是SimCLR是Google的工作，他们有钱而且有TPU，他们的batch size直接设置为8192，这已经足够用来做对比学习了。end-to-end的方式，他们每次计算字典里的key都是使用同一个encoder，所以一致性很高，但缺点就是字典大小就是mini batch的大小，不能设置的很大。

另一种是memery-bank的方式，由fig2 (b)所示。其就克服了end-to-end方法的字典小的问题，但牺牲掉了一部分的一致性。memory-bank的方法直接不需要key的encoder更新了，在一开始直接使用encoder来对于一个数据集里所有的图片都encode它的feature，也就是说如果是ImageNet，那就是100万多个feature，假设每个feature是128维，那就将这些features全部存在硬盘里。然后每次随机抽取一部分当作字典。因为并不需要反向传播这些被抽取的keys，所以字典可以选取的很大。在利用抽取的keys和query来更新query的encoder参数之后，利用query的encoder来encode被抽取的这些keys的features。这样操作会使得字典的一致性受到影响。

所以实际上MoCo这篇论文和memory bank这篇论文更加的接近，它们都只有一个encoder，也就是query的encoder，而都使用了某种数据存储方式来存储字典，从而降低内存的使用。其实memory bank那篇论文也意识到了keys特征不一致的问题，他们也使用了动量更新的方式，只不过他们动量更新的是特征，而MoCo动量更新的是编码器的参数。而且MoCo具有较好的拓展性，在上亿张图片的数据集上也能够较好的训练，而memory bank因为需要存下所有keys的features，对于上亿个这样的features，就需要大量空间来存储了。所以说，MoCo能同时提供一个又大又一致的字典，从而很好的进行对比学习。


**3.3 Pretext task**

代理任务用的就是instance discrimination，之前已经详细说过了，这里就不说了。

Algorithm1提供了MoCo的伪代码。

```shell
# f_q，f_k：query和key的编码器
# queue：K个keys构成的字典，用一个队列表示，大小是CxK，这里C就是特征的维度
# m：动量
# t：InfoNCE里的温度超参数

f_k.params = f_q.params     # 编码器的初始化，f_q的参数是随机初始化的
for x in loader:            # 从dataloader里取一个mini batch的数据，有N个，记为x，默认N=256，是标准的batch size，是可以在正常的GPU上进行训练的
    x_q = aug(x)            # x_q和x_k就构成了一个正样本对
    x_k = aug(x)
    
    q = f_q.forward(x_q)   # q的维度是NxC，也就是256x128，之所以将feature大小变为128，是为了和memory bank那篇文章保持一致，从而便于比较效果
    k = f_k.forward(x_k)   # k的维度也是NxC，也是256x128
    k = k.detech()         # 从而k就没有梯度回传了，不出现在pytorch的计算图里
    
    # 计算InfoNCE里的positive logits：维度为Nx1，也就是256x1
    l_pos = bmm(q.view(N,1,C), k.view(N,C,1))
    
    # 计算InfoNCE里的negative logits：维度为NxK，也就是256x65536，因为MoCo这篇论文默认的字典大小为65536
    l_neg = mm(q.view(N,C), queue.view(C,K))
    
    # 所有的logits拼接起来：Nx(1+K)，也就是256x65537
    logits = cat([l_pos, l_neg], dim=1)
    
    # contrastive loss，公式1，因为上面将l_pos放在了第1个位置，也就是index为0的位置，所以说正样本的分类编号就是0，从而用以下这种方式巧妙地设计了正确的label进行K+1分类
    labels = zeros(N)
    loss = CrossEntropyLoss(logits/t, labels)
    
    # SGD update：query的encoder的参数更新
    loss.backward()
    update(f_q.paras)
    
    # momentum update：key的encoder参数更新
    f_k.params = m * f_k.params + (1-m) * f_q.params
    
    # update字典
    enqueue(queue, k)     # enqueue当前的mini batch
    dequeue(queue)        # dequeue一个mini batch的keys
```

*Shuffling BN*

使用了BN之后，可能会导致当前batch的样本的信息泄露，因为需要计算当前batch的mean和variance，从而模型就很容易的找到那个正样本，而不需要真正去学一个好的模型，也就是说模型会找到一个shortcut solution。作者使用shuffling BN来解决这个问题。具体做法是，在取了一个batch的图片之后，生成了一个batch的query和一个batch的positive keys，因为使用的是多GPU训练，而BN的statistics计算是在每个GPU上单独计算的，所以在计算BN statistics之前先将这个batch里的positive keys的顺序打乱，计算BN statistics，然后再将顺序恢复。这样操作，就可以使得每个query和其对应的positive keys是用不同的BN statistics来归一化的，就避免了之前提到的问题。

>BN是很难用的技术，很容易就会导致效果不好，而且很难debug。所以现在Transformer架构的提出，直接使用Layer norm，让事情变得简单很多。

**4. Experiments**

作者使用了两个数据集：ImageNet数据集和Instagram 10亿数据集。因为文中使用的是instance discrimination这个代理任务，这个代理任务认为每个图片都是一类，所以文中将ImageNet-1k叫做ImageNet-1M。Instagram 1B数据集要复杂很多，这是真实世界的场景的图片， 可能含有一个或者多个物体，而且物体不一定位于图片中央，而ImageNet-1k数据集里都是一个位于图片中间的物体。

>MoCo使用的训练机器以及训练时间已经非常的亲民了，而且模型对于下游任务的泛化效果非常好。所以MoCo是一个既轻便又有效的模型，所以获得了很广泛的应用。所以尽管SimCLR的实验效果更好一些，但大家广泛使用的还是MoCo这套框架。

**4.1 Linear Classification Protocol**

当我们已经预训练好了模型之后，将预训练好的模型冻住，因为使用的是ResNet50的架构，所以作者在global average pooling层将feature map展平之后，直接在后面接上一个全连接层作为分类头，再监督训练这个分类头100个epoch。之后在ImageNet-1k测试集上测试分类的效果。

对于后面接的那个现行的分类头的训练，作者做了grid search（也就是找到最好的训练参数），发现最佳的learning rate是30。这是十分离谱的，因为一般来说除了在神经网络搜索，比如说NAS这种工作里之外，大部分的模型的learning rate都不会比1还大。而且这已经是一个预训练好的网络了，现在要做的只是微调，所以learning rate不应该太大。作者认为出现这种现象的原因是因为在ImageNet上通过无监督学习学到的特征分布和有监督学习学到的特征分布是非常不一样的。

**4.1.1 Ablation: contrastive loss mechanisms（使用queue构造大字典的好处）**

我们对比了fig2里提到的三种对比学习的机制的效果：也就是end-to-end，memory bank和MoCo。我们使用了相同的代理任务，并且使用了相同的InfoNCE作为loss function。结果由fig3表示。

![4]({{ '/assets/images/MOCO-4.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*fig 3. 横轴是使用的负样本的个数，可以粗略的认为就是字典的大小。纵轴是在ImageNet上linear classification protocol的准确率。*

受限于显卡内存，end-to-end的方法并不能使用非常大的字典，所以文章里最大只用到了1024。但是memory bank的方法可以使用很大的字典，所以它的曲线延伸了很远，但是效果要比end-to-end以及MoCo的方法都要差，作者认为就是因为字典里特征不一致导致的。可以看到，MoCo和end-to-end的方法在一开始重合度是很高的，但是因为没有实验支持，所以也不知道end-to-end的效果在字典很大的时候是否还能继续保持下去。但至少说明，MoCo是一个既能有很大字典又能有很好效果的方法，它是性能最好、对硬件要求最低、可扩展性也是最好的方法。

**4.1.2 Ablation: momentum（动量带来的好处）**

![5]({{ '/assets/images/MOCO-5.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}

可以看到，使用较大的动量，能达到比较好的效果。但是将动量变小到0.99或者0.9的时候，性能下降就比较明显了。当去掉动量之后，模型不能收敛，loss一直在震荡，从而训练失败。上述结果表明，建立一个特征一致性的字典是非常重要的。

>文章提出了几点贡献，就需要对这些贡献做相对的消融实验，这样才能证明所提出的贡献是有效的。

**4.1.3 Comparison with previous results**

![6]({{ '/assets/images/MOCO-6.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}

这里所有的方法都是在linear classification protocol的框架下进行的，也都是在ImageNet这个数据集上进行的。这些方法都是将预训练好的网络当作一个特征提取器，抽出来特征，再去训练一个全连接层当作分类头，最后得到这些表里的结果。所有的这些结果都没有去更新预训练好的那个模型backbone。而且因为对于无监督学习来说，模型的大小是很重要的，一般来说，模型越大，效果就越好，所以为了公平比较，作者还列出了各个方法所使用的模型的参数数量（也就是模型大小）。


**4.2 Transferring features**

作者想验证一下，MoCo非监督预训练所学习到的特征，到底能不能在下游任务上取得很好的迁移学习效果。这是这篇文章的点睛之笔。

无监督学习的主要目标，就是要去学习一个可以迁移的特征。在ImageNet上做有监督的预训练的最大的作用，就是这个预训练好的模型可以作为下游任务的初始化，从而下游任务在这个预训练好的模型上微调就可以，这样即使下游任务只有很少的标注数据的时候，也能有很好的效果。作者使用CV领域最常见的一个任务，detection，来进行MoCo无监督预训练模型和有监督预训练的比较（都是在ImageNet上）。

但是还需要先说两个问题。

*Normalization*

首先就是归一化。在4.1里一开始也提到了，如果拿MoCo预训练的模型直接作为特征提取器，那么在下游linear classification protocol任务微调的时候，最佳学习率是30，因为MoCo预训练所学习到的特征和有监督学习所学习到的特征分布是非常不一样的。但我们现在要使用MoCo预训练模型来做不同的下游任务，如果对于每个下游任务都做grid search就太复杂了，都去找一下它的最佳学习率是不现实的，也失去了无监督预训练的意义。

而当特征分布不一样的时候，最常见的解决方法就是归一化。作者在微调的时候使用了feature normalization的方法。现在之前预训练好的模型参数不再固定了，整个模型的参数都在微调了，而且使用了Sync BN的方式（也就是说在多卡训练的时候，在每个GPU上计算出来的BN mean和variance都统一起来计算一个统一的mean和variance，然后再应用到每个GPU上的BN的计算）。这样就会让特征的归一化做的更加彻底，也会让模型训练更加稳定。而且对于下游任务新增加的那些层，我们也为他们加上BN，这样可以控制值域的范围，从而更好的使得特征归一化。

我们对使用MoCo非监督和有监督预训练的框架在微调的时候都使用了上述的normalization操作，这样MoCo预训练模型在微调的时候就可以使用和监督预训练模型相同的超参数作为训练参数（比如learning rate）来进行微调了。

*Schedules*

对于一些下游任务，如果它的训练集很大的话，比如说COCO上的detection，那其实即使模型是随机初始化的，效果也会很好，那这样的话，预训练就没有什么意义了，那不管是无监督预训练还是有监督预训练就没有意义了，我们直接随机初始化模型，然后在下游任务数据集上训练就行了。但作者提出，即使是这样的情况，也需要在下游任务上训练很久才能有很好的效果。所以说对这样的下游任务，我们缩短它的训练时间，从而我们还可以比较我们的MoCo无监督预训练模型和有监督预训练模型的效果。

将上述两点结合起来，对于MoCo预训练来说，我们在下游任务上的微调使用的超参数就和监督方式预训练模型使用的超参数一样了。这样就省了我们为不同下游任务寻找不同超参数的烦恼。

**4.2.1 PASCAL VOC object detection**

![7]({{ '/assets/images/MOCO-7.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}

两个表格用了不同的网络结构，每个表格都是四行，第一行是随机初始化的网络，是baseline，第二行是在ImageNet上使用监督学习的方式进行预训练，第三四行是MoCo预训练方式，使用的分别是ImageNet和Instagram数据集。

之前对end-to-end、memeory bank和MoCo这三种对比学习的方式进行比较是在ImageNet上进行linear classification protocol对比的，这次再在下游任务上比较这三种方式的效果，进一步增强说服力。而且end-to-end和memory bank的非监督预训练模型在下游任务上的效果并没有超过监督学习预训练模型的效果，只有MoCo是真的超越了。

![8]({{ '/assets/images/MOCO-8.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}

**4.2.2 COCO object detection and segmentation**

![9]({{ '/assets/images/MOCO-9.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}

进一步在COCO数据集上也进行了detection任务效果的测试。因为COCO数据集很大，不需要预训练也能有很好的效果，所以作者对于不同的模型框架也比较了不同的下游任务训练时长下的效果。MoCo仍然是效果最好的。

而且作者还不辞辛苦的在COCO数据集上的很多下游任务上都进行了测试，MoCo的效果都要比有监督的预训练模型效果好（或者差不多），可见MoCo确实是很有效果的。

![10]({{ '/assets/images/MOCO-10.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}

作者还提供了一个总结。MoCo在很多下游任务上都要比有监督的预训练模型效果要好，但是在某些任务上MoCo的效果差一点，这些任务是实例分割（instance segmentation）和语义分割（semantic segmentation）的任务上。

>从而有些人怀疑，对比学习是不是不太适合做这种dense prediction的任务（每个像素点都要去预测的任务）的预训练。后续也有了很多基于这个出发点的工作，比如说[Dense Contrastive Learning for Self-Supervised Visual Pre-Training](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Dense_Contrastive_Learning_for_Self-Supervised_Visual_Pre-Training_CVPR_2021_paper.pdf)，[Exploring Cross-Image Pixel Contrast for Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Exploring_Cross-Image_Pixel_Contrast_for_Semantic_Segmentation_ICCV_2021_paper.pdf)，[Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning](https://arxiv.org/pdf/2011.10043.pdf)等。

总体来说，MoCo在很多下游视觉任务上，已经将无监督预训练模型和有监督预训练模型之间的差别抹平了。

而且，和NLP里得出的结论一样，如果有更大的数据集，有更大的模型，那么无监督预训练的效果就会更好。这个也符合真实世界里无监督学习的目标。


>上述的实验里，4.1部分是linear classification protocol，也就是先在ImageNet上使用文中提到的MoCo方式进行无监督与训练，然后再有监督的训练一个分类头，从而在ImageNet测试集上测试classification任务的效果，而和文中方法对比的方法也就是end-to-end和memory bank的无监督预训练方式。注意到，MoCo的预训练方法最后达到的linear classification准确率其实也就只有60%多，这和利用有监督方法在整个ImageNet上训练，然后测试的效果不能比（现在最好已经到80%多了）。但是这篇文章的目的本身也就不是为了在这个任务上和有监督的方法比，这一节的目的就是为了让MoCo和end-to-end以及memory bank的方法比，从而证明都是无监督对比学习，MoCo的方式是最好的（但肯定比不过有监督学习。但是在4.2里，情况就不一样了。因为对于一些下游任务来说，其数据集可能不够大，比如说在PASCAL VOC数据集上进行的试验，如果是一般的方法，那就是随机初始化模型，然后在PASCAL VOC数据集上进行监督学习，作者证明这样的效果不如先将模型初始化为MoCo在ImageNet上预训练好的模型，然后再在PASCAL VOC数据集上监督学习来的效果好，从而证明了MoCo无监督预训练的有效性。而对于COCO数据集，这个数据集很大，所以如果直接随机初始化，然后再在这个数据集上进行监督学习，和利用MoCo预训练初始化，然后再在COCO数据集上进行监督学习，最后的效果差不多。这就体现不出来MoCo预训练的作用了。所以作者采用了减少在COCO数据集上监督学习时间的操作，作者说，如果使用了MoCo预训练模型，那么即使在COCO数据集上进行很少的几个epoch，就可以达到不错的效果，这样也证明了MoCo预训练模型的作用。


**5. Discussion and Conclusion**

我们提出的无监督表征学习方法在一系列视觉任务上都取得了很好的结果。但有一些开放性的问题值得讨论。首先，当我们使用10亿的数据集而不是ImageNet那个100万的数据集的时候，性能提升是有的，但是比较小。这说明大规模的数据集并没有被很好的利用起来。一个更好的代理任务可能可以解决这个问题。除了这个简单的instance discrimination的代理任务，我们有没有可能将MoCo和其它的代理任务，比如说masked auto-encoding结合起来用呢？比如说NLP里的BERT。MoCo设计的初衷就是去构造一个大的字典，从而让正负样本能够更有效地去对比，提供一个稳定的自监督信号，最后去训练这个模型。


### 2. [对比学习串烧：对比学习在CV领域的发展历程总结]（从2018年到2021年12月31日）

将具有代表性的工作进行总结，对比学习在CV领域的发展大致可以分为四个阶段：（1）百花齐放：InstDisc，InvaSpread，CPC，CMC。在这个阶段，方法、模型都还没有统一，目标函数也没有统一，代理任务也没有统一，所以说是一个百花齐放的时代。（2）CV双雄：MoCo和SimCLR之间的较量，介绍MoCov1，SimCLRv1，MoCov2，SimCLRv2，以及CPC和CMC的延伸工作，还有SwAV。这个阶段发展非常迅速，这些工作一般都是间隔一到两个月就进行了更新。ImageNet上最好的成绩基本上每个月都在被刷新。（3）不用负样本也可以做对比学习：BYOL以及后续的一些改进。最后，SimSiam将上述所有的方法进行了总结，都融入了一个框架之中。SimSiam是使用CNN做对比学习的一个总结性的工作。（4）之后就到了Transformer时代：MoCov3和DINO。

>Vision Transformer十分火爆，所以很多人都使用ViT来设计模型。对于无监督学习来说，不管是对比学习，还是掩码学习，现在大家主要都是用ViT来做了。

#### 第一阶段

第一阶段将会介绍四篇论文。它们使用的代理任务是不一样的，有个体判别，有生成模型来预测未来的数据，还有多视角多模态的判别。它们使用的目标函数也不是很相同，有NCE，有InfoNCE，还有NCE的其他变体。使用的模型也不一样，比如说InstDisc就使用一个编码器和一个memory bank，而InvaSpread就只使用了一个编码器，CPC使用了一个编码器和一个自回归模型（RNN或者LSTM），CMC就可能会有多个编码器。它们所针对的任务也从音频到图片、文字、强化学习等，非常丰富多彩。所以第一阶段是一个百花齐放的阶段。大概是从2018年到2019年中期。

#### 1. [InstDisc: Unsupervised Feature Learning via Non-Parametric Instance Discrimination](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Unsupervised_Feature_Learning_CVPR_2018_paper.pdf)

*CVPR 2018*

>这篇文章就是MoCo那篇文章里提到的使用memory bank方法的论文，也是MoCo那篇论文里的文献61，被多次提及。这篇工作是mmlab做的，也是大佬作品。

是这篇文章提出了个体判别（instance discrimination）这个代理任务，后续很多论文（包括MoCo）都是使用的这个代理任务。如果说MoCo是个里程碑式的工作，那么InstDisc这篇工作就是巨人的肩膀。除了MoCo还有很多工作，很多实验的细节都是直接根据这篇论文里的实验设置来的。

![instdisc1]({{ '/assets/images/INSTDISC-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 1. 作者提出的方法收到了有监督学习的启发。如果将一张leopard的图片，喂给一个已经用有监督学习方式训练好了的分类器，我们可以发现分类器给出的分类结果，概率大小排名前几的都是和leopard很相关的类别，比如说leopard，jaguar，cheetach，snow leopard等，从图片语义角度来看，这些类别的图片和leopard的图片长的都是很相似的。而概率大小排名靠后的那些类别，比如说lifeboat，shopping cart，bookcase等，都是和leopard没有一点关系的类别。造成这样的结果并不是因为标签语义上相似的原因（因为并没有参杂NLP的内容，所以所有的标签在模型看来应该是同等关系和地位的），只是因为这些类别的图片的appearance就是有很多相似性。作者根据这个观察，提出了个体判别（instance discrimination）这个任务，将按照类别的有监督学习推到了极致：将每一个instance都看成一个类别，也就是每一张图片都是一个类别，而我们的目标就是让模型学习一个特征，将每一张图片都区分开来。*

>这个想法十分自然但又十分合理，因为模型本身就对于每张图片就应该会给出不同的feature embedding，只不过同一个类别的图片的embedding比较接近而已。但作者通过上述实验结果发现，即使是不同的类别的图片，只要其appearance比较接近，那它们的embedding也会比较类似，这样的想法推到极致，就变成了那可以让模型认为每一张图片都是一个类别，这并不影响同一个真实类别的图片的embedding相似这样一个事实。但这样设置就巧妙了不再使用标记的监督信号，从而造就了一个无监督学习的任务：instance discrimination。

>这个fig1画的很好，不仅介绍了研究动机，也一句话介绍了instance discrimination这个代理任务，十分清晰明了。

![instdisc2]({{ '/assets/images/INSTDISC-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 2. 介绍了文章使用的方法，也就是pipeline。作者使用一个CNN backbone将每一张输入图片都编码为一个特征，而这些特征在最后的特征空间内，都能尽可能的分开（因为对于instance discrimination这个任务来说，每个图片自成一类，所以每张图片的特征都应该和其它图片的特征尽量的分开）。而训练这个神经网络的方法就是对比学习，从而需要有正样本和负样本。对于instance discrimination这个任务来说，正样本就是经过一些数据增强的这张图片本身，而负样本就是数据集里所有的其他图片。而问题是，如果做对比学习，如此庞大的负样本特征应该如何存储呢？作者提出了使用memory bank的形式，将所有的图片的特征都存到这个memory bank里，也就是一个2D的矩阵，每一行是一个图片的特征。ImageNet一共有128万张图片，从而memory bank就有128万行，从而特征的维度就不能太高，要不然存储的空间就太大了。作者使用的是128维。*

整个的前向过程（forward）如下：比如说batch size使用的是256，而模型是ResNet50。从而最后分类头之前的输出特征是2048维，使用MLP将其降维到128维，再对每个图片的特征做$$L_2$$ normalization，从而得到了一个$$256 \times 128$$的矩阵，这就代表了正样本。而负样本就是从memory bank里随机抽取而来，在这篇论文里作者每次抽取4096个负样本。这样有了正样本和负样本，就可以使用NCEloss来计算loss从而更新网络参数了。一旦网络更新完了之后，就可以将这个mini batch，也就是这256张图片计算出来的特征，在memory bank里进行更新，从而更新memory bank里所存储的图片的特征。然后就是重复这个过程，不断更新模型，更新特征，更新memory bank，从而使得最终学习到的特征具有区分性。

这篇论文里还有很多细节都设计得很巧妙，比如说proximal regularization，因为和正常的classification任务不一样，我们这里每个图片就是一个类，从而每一类就一张图片，会因为采样的问题导致训练有很大的波动，从而作者设计了一个方法，这个方法其实很类似于MoCo里的动量更新的思想，但是MoCo里是使用动量更新模型参数，而这里是使用动量更新feature embedding。假设输入图片是$$x_i$$，网络是$$f_{\theta}$$，特征是$$v_i^{(t)}$$，那么$$v_i^{(t)} = f_{\theta}(x_i)$$。在$$t-1$$时刻的memory bank就是$$V = \lbrace v^{(t-1)} \rbrace$$。然后，对于这个正样本的loss就定义如下：

$$-log h(i, v_i^{(t-1)}) + \lambda \lVert v_i^{(t)} - v_{i-1}^{(t-1)} \rVert_2^2 $$

其中$$h$$表示的是某种NCEloss函数。所以说，就是用$$t-1$$时刻的feature embedding来计算NCEloss，同时又让$$t$$时刻的feature embedding和$$t-1$$时刻的feature embedding不要想差的太远，也就是控制feature embedding的变化程度，从而达到使得训练更加平滑的目的。

效果如fig3所示。

![instdisc3]({{ '/assets/images/INSTDISC-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 3. Proximal regularization的效果。原始的没有使用这个方式的训练曲线非常抖动，加了proximal regularization的训练曲线平滑多了。*

除了proximal regularization，文章第四节里的实验设置里的超参数设置，比如说NCEloss里的温度系数，feature embedding的大小，batch size的大小，负样本的个数，learning rate的大小，训练的epoch数量等，都被后来的论文尤其是MoCo沿用了下来，所以说INSTDISC这篇论文是对比学习的里程碑式的工作。

这篇文章提出了instance discrimination这个代理任务，而且将这个代理任务和NCEloss相结合，取得了不错的无监督表征学习的结果。同时他还提出了使用别的数据结构（memory bank来存储负样本，还提出了proximal regularization来用动量的方式对feature embedding进行更新，对后来的对比学习的工作起到了至关重要的推进作用。


#### 2. [InvaSpread: Unsupervised embedding learning via invariant and spreading instance feature](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ye_Unsupervised_Embedding_Learning_via_Invariant_and_Spreading_Instance_Feature_CVPR_2019_paper.pdf)

*CVPR 2019*

这篇论文可以被理解为SimCLR的前身。作者提出的方法没有使用额外的数据结构来存储负样本，其负样本就是来自于一个mini batch，而且只使用一个encoder来进行end-to-end的学习。这篇文章提出的方法就是基本的对比学习。

![inva1]({{ '/assets/images/INVA-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 1. 同一张图片经过不同的数据增强之后，经过编码器得到的特征应该很类似，而不同的图片经过编码器之后得到的特征应该差别较大。*

这篇文章的basic idea很简单，对于相似的图片或者物体，其特征应该invariant，对于不同的图片或者物体，其特征应该spreading，就是标题里的两个单词。

具体做法如fig2所示。

![inva2]({{ '/assets/images/INVA-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 2. 代理任务上，这篇文章也选取了个体判别这个任务，我们主要来看作者是如何选取正负样本，以及正负样本的个数。来看前向过程。batch size是256，也就是fig2里最左侧上面有256张图片（$$x_1, x_2, \cdots, x_{256}$$）。而经过了数据增强，fig2左侧下面也是256张图片。对于一张图片，比如$$x_1$$来说，其正样本就是经过数据增强后的图片$$\hat x_1$$，负样本就是剩下的所有的图片，包括$$x_2,x_3, \cdots, x_{256}$$，也包括经过了数据增强后的图片$$\hat x_2, \hat x_3, \cdots, \hat x_{256}$$。也就是对于这个mini batch来说，正样本个数是256个，负样本的个数是$$(256-1) \times 2$$。这里就和之前InstDisc那篇论文里的方法不太一样了，这里不从memory bank里取负样本，而是直接从一个mini batch里取负样本，这样做的好处就是可以做end-to-end的训练了，这也就是MoCo那篇文章里所对比的end-to-end的方法。剩下的前向过程和InstDisc里的方法就差不多了，也是使用了NCEloss的一个变体。*

之所以说这篇论文，是因为它属于另一个流派，也就是端到端（end-to-end）的方式，只需要一个编码器，不需要使用外部的数据结构来存储大量的负样本，它的正负样本都来自于同一个mini-batch。然而既然其和SimCLR那么相似，为什么没有取得SimCLR那么好的效果呢？这个原因在MoCo那篇论文里已经提及，就是因为这样的方法需要字典足够的大，也就是做对比学习的时候，负样本最好足够多。而这篇文章所使用的batch size就是256，远比SimCLR里使用TPU的batch size小，而且也没有SimCLR里使用的那么多的数据增强操作。所以这篇文章的结果并没有那么好，所以并没有引起多少关注，但实际上和SimCLR是一样的。


#### 3. [CPC: Representation Learning with Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf)

[CODE](https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch)

*Arxiv 2019*

前两篇论文都是使用个体判别这个任务做的对比学习，而这篇论文使用了不一样方法。一般机器学习就分为判别式模型（discriminative model）和生成式模型（generative model），个体判别这个任务显然是属于判别式模型，而相对应的肯定就会有一些生成式模型的代理任务，就比如这篇文章CPC里所用的predictive coding这个任务。

CPC这篇论文实际上很厉害，它是一个很通用的结构，其可以处理音频、图片、文本以及在强化学习中使用。为了简单起见，fig1里以音频输入作为例子来介绍CPC的结构。

![cpc1]({{ '/assets/images/CPC-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 1. CPC的一个overview。*

CPC这个模型的想法是，我们有一个时序的输入，比如说fig1里的$$\cdots, x_{t-3}, x_{t-2}, x_{t-1}, x_t, x_{t+1}, x_{t+2}, x_{t+3}, x_{t+4}, \cdots$$，$$x_t$$是当前时刻，往前的是之前的时刻的输入，往后的是将来的时刻的输入。我们现在将之前时刻的输入，也就是$$x_t$$（包括$$x_t$$）之前时刻的输入都喂给一个相同的编码器，这个编码器对于不同时刻的输入就会给出对应的特征，再将这些特征序列喂给一个自回归（auto-aggressive）的模型，常见的就是RNN或者LSTM，从而得到每个时刻最终的输出，而时刻$$t$$的输入$$x_t$$对应的最后的输出就是$$C_t$$，context representation，这是代表上下文的一个特征表示。作者认为，如果这个$$C_t$$学习的足够好，也就是其真的有当前以及之前的输入的信息，那它应该可以做出一些合理的预测。所以就应该可以用$$C_t$$来预测未来时刻的编码器的输出，也就是$$x_{t+1}$$往后的输入经过编码器之后得到的输出$$z_{t+1}$$等等。

那对比学习体现在哪里呢？这里的正样本就是未来时刻的输入$$x_{t+1}, x_{t+2}, x_{t+3}, \cdots$$通过编码器得到的未来时刻的输出，$$z_{t+1}, z_{t+2}, z_{t+3}, \cdots$$，而query则是根据$$C_t$$经过某些模型得到的未来时刻的输出的预测。负样本的定义就很广泛，可以任意选取输出，根据这个编码器得到的输出，都可以作为负样本来使用。

CPC里的这套思想是很普适的。将输入换成文本，就相当于每个时刻的输入是一个个的单词，而将输入换成图片，就可以在图片中截取图片patches，就可以利用一部分的图片patches，来预测其它部分的patches。方法非常的灵活。


#### 4. [CMC: Contrastive Multiview Coding](https://arxiv.org/pdf/1906.05849.pdf)

*Arxiv 2019*

这篇论文定义正样本的方式就更加广泛了：一个物体的多个模态可以被当作正样本。

人类观察这个世界是通过很多个传感器的，比如说眼睛、耳朵等。这些都充当着不同的传感器来给大脑提供不同的信号，将每个传感器所提供的信息看作一个view。那每个view都是带有噪声的，而且可能是不完整的。但是最重要的那些信息，是在这些views之间共享的，比如说基础的物理定律、几何形状或者说它们的语义信息等。比如说，一只狗，既可以被看到，也可以被听到，也可以被感受到。基于上述想法，作者提出他们想要去学习一个强大的特征，这个特征具有view-invariant的特性。也就是说CMC这篇工作的目的就是要增大不同views之间的mutual information。如果能学习到一个特征，能抓住所有的views里的最关键的信息，那这个特征就非常的好了。

那么CMC是如何选取正样本、负样本来做对比学习的呢？fig1解释了这个过程。

![cmc1]({{ '/assets/images/CMC-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 1. 文中选取的是NYU-RGBD这个数据集。这个数据集对于每个图片有四个view，分别是原始的图片，图片的深度信息（也就是每个点距离相机的距离），surface normal信息和经过物体分割后的图片。CMC认为尽管上述输入是不同的模态，但是其对应的都是同一个场景，它们应该互为正样本，也就是说，当我们有了一个特征空间的时候，同一张图片的四个views得出的四个特征，在这个特征空间里就应该很接近。如果我们在随机挑一张图片的任意一个view，其生成的特征，就应该和这些特征远离。这就是CMC定义正负样本的方式。*

定义好了正负样本之后，剩下的工作就和之前的对比学习工作的操作差不多了。但CMC还有两点值得注意。

首先，CMC是第一个做多视角的对比学习的工作，其不仅证明了对比学习的灵活性，还证明了这种多视角、多模态的可行性。所以接下来，OpenAI很快做出了CLIP（[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020.pdf)），也就是说如果有一张图片，还有一个描述这张图片内容的文本，那么这张图片和这个文本就可以当作一个正样本对，就可以来做多模态的对比学习。同时，CMC这篇文章的作者，利用对比学习的思想，做了一篇蒸馏的工作，它们的主要想法是，对于不同的网络，不管网络是大是小，是什么形式的，只要输入的是同一张图片，它们的输出特征就应该尽可能的类似，也就是teacher模型的输出和student模型的输出对于同一张图片来说尽可能的类似。通过这种方式，将teacher模型和student模型对于同一张图片的输出特征作为一个正样本对，从而可以去做对比学习（ICLR2020的文章，[Contrastive representation distillation](https://arxiv.org/pdf/1910.10699.pdf)）。CMC这篇论文让人们意识到了对比学习的灵活性。

其次就是，CMC的最大的一个局限性就是，对于不同的模态，不同的视角，我们可能需要不一样的编码器，因为输入的形式可能非常的不一样。这就可能导致，你使用几个视角，就得使用几个编码器。在训练的时候，计算代价就有点高。比如说在CLIP这篇论文里，它的文本端就使用了一个类似于BERT的Transformer，而图片端就使用了一个ViT，也就是需要两个编码器。但是Transformer有个很大的好处，其是有可能能够处理多模态的数据的。现在已经有人这么做了，ICLR 2022的一篇文章MA CLIP（[MA-CLIP: Towards Modality-Agnostic Contrastive Language-Image Pre-training](https://openreview.net/forum?id=ROteIE-4A6W)）就使用一个Transformer同时处理两个输入模态。这可能才是Transformer最吸引人的地方，就是一个网络可以处理很多模态的数据，而不用对网络和数据做多少操作。


#### 第二阶段

从2019年中到2020年中。而这一阶段所要讲的就是MoCo和SimCLR。因为MoCo和SimCLR的效果很好，所以这段时间Arxiv每个月都有关于对比学习的论文。这股潮流直到2020年底Vision Transformer的提出才消褪。

到了第二阶段，不同的论文里很多的细节都趋于统一了，比如说目标函数基本就是infoNCE了，而模型也都是一个编码器（ResNet50或者更大的）后面加上一个projection head，大家也都采用了更强的数据增强，也都想用动量编码器，也都尝试去训练更久。最后在ImageNet上的准确率也逐渐逼近有监督学习的baseline模型。

#### 1. [MoCov1: Momentum Contrast for Unsupervised Visual Representation Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)

*CVPR 2020 Best Paper Candidate*

我们已经精读过了这篇论文，所以这里就主要介绍一下和其他论文的区别与联系。MoCo或者叫做MoCov1主要就是将所有的对比学习的问题都归纳为了一个字典查询的问题。它提出了两个主要的内容，一个是使用队列表示字典，另一个是使用动量的方式更新编码器参数，从而去构造一个又大又一致的字典，从而帮助更好的对比学习。

MoCo和InstDisc是非常类似的，其就是使用队列来取代memory bank，作为一个额外的数据结构来存储这些负样本。用动量编码器（momentum encoder）去取代了InstDisc论文里loss里的proximal regularization，从而能达到动量的更新编码器的目的，而不是动量的去更新编码器所学习到的特征，从而能得到更好的结果。但是整体的出发点，包括实现的细节都是十分类似的。MoCo这篇论文使用的也是ResNet50作为模型框架，也是使用了0.07的温度参数，图片的特征维度也是设定为128维，也使用了$$L_2$$ normalization，这都是和InstDisc一样的。只不过MoCo使用了InfoNCE，而InstDisc使用的是NCE作为目标函数。而且MoCo里的数据增强方式也和InstDisc一样，训练200个epoch，初始learning rate是0.03都和InstDisc保持一致。所以说也可以认为MoCo是InstDisc的改进工作。

但是MoCo真正出色的地方有两点。首先，它的改进真的是简单有效，而且影响了后续一系列的工作。比如说动量编码器，在后续的SimCLR，BYOL，一直到最新的对比学习的工作都还在使用。说明它的改进是深刻而又有效的，不仅帮助MoCo在无监督预训练模型上的下游任务里超过了有监督预训练模型，对于后续的工作也是影响巨大。另一个就是MoCo的写作方式。一种普通的写作方式就是在introduction里介绍对比学习是什么，然后已经有了哪些工作，比如说有end-to-end的方法，有InstDisc这种使用memory bank的方法，然后介绍它们各自的缺点和局限性。从而这篇文章提出MoCo，用队列去解决字典大小的问题，用动量编码器去解决字典里特征不一致的问题。最后通过实验证明效果很好，第一次用一个无监督训练的模型在下游任务上的效果要比有监督预训练模型的效果好。上述这种写法也是一种直接明了的写作方式，大部分论文的写作都是按照这个套路来的。但是MoCo的作者的写作水平明显就高了一个层次。上来先介绍CV和NLP之间的区别，然后说明为什么无监督学习在CV这边做的并不好。第二段再说对比学习，但也并不是像上面那样详细的介绍之前的对比学习的方法，而是将对比学习解释为了一个字典查找的问题，将问题给归纳升华了。然后在CV和NLP大一统的框架下，在所有对比学习方法都大一统的框架下，作者提出了MoCo这样一个方法。希望能用一个又大又一致的字典，去整体提高对比学习的性能。从而一下提高了论文的写作的格局，这远不是之前那种写作方式可以比的。而且在introduction里的写作风格还延续到了第三节的method里。作者并没有一上来就给出一个模型总览图，也没有具体说MoCo的模型是什么，代理任务是什么，而是先从最后的目标函数入手，定义了正负样本，然后再去解释网络结构，再利用伪代码来介绍算法。而且在3.1里，为了让模型更加普适，作者并没定义输入是什么，也没有定义网络结构是什么样的，也就是说，什么样的输入都可以，它可以是图片，可以是图片patches，也可以是具有上下文信息的图片patches，而图片patches则是对应了CPC这篇论文。而query和key的编码器既可以是相同的，可以是部分共享的，也可以是完全不同的，相同的就对应了InvaSpread那篇论文，而完全不同的就对应了CMC那篇论文。所以说MoCo这种自顶向下的写作方式也是非常值得借鉴的。但这需要有极强的功力，写得不好的话，就很难让读者看懂了。


#### 2. [SimCLRv1: A Simple Framework for Contrastive Learning of Visual Representations](http://proceedings.mlr.press/v119/chen20j/chen20j.pdf)

*ICML 2020*

*Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton*

这是Hinton指导的，在Google做的论文。这个方法如标题所示，真的非常简单，概念简单，方法简单，只不过需要很大的batch size，从而不是很好上手。

![simclr1]({{ '/assets/images/SIMCLR-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 1. 输入是一个mini batch的图片，$$\pmb x$$，对这个mini batch里所有的图片做数据增强，从而得到$$\pmb x_i$$和$$\pmb x_j$$。同一张图片经过不同的数据增强得到的图片就是正样本，而不同的图片就是负样本。从而如果batch size是$$N$$，那么正样本的个数是$$N$$，负样本就是经过这两次数据增强后的除了这张图片以外的所有的图片，也就是$$2(N-1)$$张。这和InvaSpread里的一样，只不过InvaSpread里是原图片和经过一次数据增强后的图片，这里没有再用原图片。再有了正负样本之后，就使用编码器$$f$$进行编码，这里的$$f$$是共用的。如果这个$$f$$是ResNet50的话，那这里的$$\pmb h$$就是2048维的。SimCLR的一个重大的创新点是在$$\pmb h$$后又加了一个projector，也就是一层MLP，也就是一个全连接层加上一个ReLU。但是就是这么一个简单的MLP，使得其在ImageNet的分类任务上直接提升了10%，这样的效果在别的无监督学习方法或者有监督学习方法里很难观察到的。很少有加上一个MLP就提升了10%的效果的。所以说这是一个非常有趣也非常惊讶的结果。一般来说，$$z$$的维度会比$$h$$要小一点，为了和之前的工作保持一致性，这篇文章里也将$$z$$设置为128维。最后在$$z$$上进行对比学习，它们使用的是normalized temperature scaled的交叉熵函数作为目标函数。normalized的意思就是将$$z$$进行了$$l_2$$归一化，temperature scaled就是加了温度系数$$\tau$$，所以说这里的目标函数和NCE loss也是很相似的。这个projection head，$$g$$只有在训练的时候才会被使用，在做下游任务的时候，就不需要$$g$$了，而是使用$$h$$来做下游任务。这样就能够和之前的工作进行公平的对比，因为我们并没有新增层数，还是用的ResNet50，而新增的$$g$$只是为了让预训练效果更好。*

>注意，在MoCo还有InstDisc，InvaSpread等工作里，是将ResNet的global average pooling层的输出取出，这层的输出是2048维，然后加了一个$$nn.Linear$$层，将其降维为$$128$$维，然后做的后续操作。注意$$nn.Linear$$层只是对于输入$$x$$做一个$$xA +b$$的操作，并没有ReLU等非线性操作。而对于这篇文章的projection head $$g$$，同样也是对于ResNet的global average pooling输出，加上了一个带有一个隐层的MLP，也就是$$g$$，如果$$g$$的输入是$$x$$的话，那输出就是$$W_2(\sigma(W_1 x))$$，这里$$\sigma$$表示的是非线性函数，比如说ReLU。注意这里是有很大区别的，这也是SimCLR这篇文章说的一个很大的创新点。

SimCLR确实是很简单，和MoCo比起来，这里只有一个编码器，既不需要memory bank，也不需要queue，也不需要动量编码器，正负样本也都是从同一个mini batch里来的。整个前向过程非常直接，就是图片进入编码器编码，然后projector降维，最后算一个对比学习的loss。非常符合大家对深度学习这个工作的期待。

InvaSpread可以看作是SimCLR的前身。整体的思路和结构是很相似的。而SimCLR和InvaSpread的区别也写在了文章的贡献列表里了。首先，SimCLR用了更多的数据增强，它们说对比学习需要很强的数据增强技术。第二个，就是他们加了一个projection head，$$g$$函数，一个可以学习的非线性变换（带有一个隐层的MLP）。最后，就是他们用了更大的batch size，而且训练的时间更久。

作者也说了，文中提到的技术或多或少都在以前的工作里出现过，作者说本文的方法就是将这些方法结合，但是确实达到了很好的效果。而且在附录里做了大量的消融实验，从而证明每个模块都是不可或缺的。但实际上因为SimCLR确实好用，其方法对后续的工作产生了长远的影响，比如说MoCov2，BYOL里也在编码器之后加上了这样一个projection head。并且本文里提到的数据增强的操作，在后续文章里也被广泛的使用。以及文章里使用的lars这个优化器去做大batch size的训练，之后的BYOL也采用了同样的策略。总之，SimCLR十分简单，但却也为后续的很多研究铺平了道路。

我们来看一下SimLCR里的数据增强。

![simclr2]({{ '/assets/images/SIMCLR-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 2. 各种图像增强方法。*

我们再来看一下projection head的影响。

![simclr3]({{ '/assets/images/SIMCLR-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 3. 这里，蓝色表示的是和InstDisc，MoCo，InvaSpread等论文里一样的方法，也就是对于ResNet的global average pooling层的2048维的输出，进行一个nn.Linear操作，也就是一个linear projection。红色表示的是本文里用的方法，也就是对于ResNet的global average pooling层的2048维的输出，进行一个带有一个隐层的MLP的计算，也就是$$W_2(\sigma(W_1 x))$$。而绿色表示的是直接将ResNet的global average pooling层的2048维的输出拿来做计算。*

从fig3可以得出两个结论。第一，加上了projection之后效果会增强很多，而nonlinear projection（本文里的做法）要比InstDisc等文章里用的linear projection的做法效果还要好。第二，projection的维度对结果的影响并不明显。


#### 3. [MoCov2: Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/pdf/2003.04297.pdf))

MoCov2实际上只有两页，只能算作一片技术报告，而不是一篇论文了。但是即使只有两页，仍然传达了相当大的信息量。

作者在看到SimCLRv1的效果很好，并且发现SimCLR提出的方法都可以很简便的拿来用在MoCo上之后，就在MoCov1的基础上加上了projection head和更多的数据增强操作，从而大幅超过了MoCov1的效果，也超过了SimCLRv1的效果。

>SimCLRv1是2月13日放在Arxiv上，而MoCov2是3月9号放到Arxiv上的。效率十分惊人。

MoCov2相对于MoCov1的改进由table1进行了总结：

![mocov21]({{ '/assets/images/MOCOV2-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}

一共改动了四块内容：加了projection head，也就是nonlinear projection；做了更多的数据增强；使用cosine learning rate schedule调整learning rate；使用了更多的epoch。

最后这个epoch很有意思。不仅是MoCov2，SimCLR以及最近的MAE也都是，训练的时间越久，效果就越好。这似乎是很多无监督学习模型的一个特性。

作者最后还是说，相对于SimCLR来说，MoCo的最大优势就是硬件要求低，只需要在一台八张V100的机器上训练1-2天就能得到表中的结果，而SimCLR要获得类似的结果需要付出更大的硬件代价以及时间更久。


#### 4. [SimCLRv2: Big Self-supervised Models are Strong Semi-Supervised Learners]()

这篇文章的重点并不在于SimCLR的改进，而是提出一种新的学习模式，也就是先使用自监督学习的方法来学习一个模型，再利用很小一部分有标签的数据对模型进行微调，再微调之后再利用这个学习好的模型对很多无标签的数据生成伪标签，从而就可以在另一个模型上进行自监督学习了。

作者只是在第3页提到了如何改进之前的SimCLRv1模型。做了如下三点改进：

* 有一个公认的事实就是无监督学习利用更大的模型效果就会更好，于是作者将之前的ResNet50换成了ResNet-152，从而这个backbone网络变得很强。
* 加深了之前的nonlinear projection，也就是projection head，将之前的linear + relu + linear加深成了linear + relu + lineaer + relu + linear，也就是有两个隐层。作者也做了更多的实验加深更多的层数，发现两层就足够了，再加深也不会有什么改变。
* 也使用了MoCo里的动量编码器。但实际上只提升了一个点。作者解释道因为他们已经使用了非常大的mini batch了，比如说8192，已经有非常多的负样本了。

>但是这里很奇怪，MoCo之所以使用动量编码器，是因为不想使用很大的mini-batch又想尽量使得更新不同样本的编码器尽量相同的无奈之举。但实际上在SimCLR里，它们直接使用一个编码器每次提取所有的负样本的特征，这些负样本的特征是100%一致的，因为压根就是同一个编码器提取出来的，所以说还有什么必要使用动量编码器呢？即使每次更新很慢，造成的结果只是每次计算loss的时候，负样本的特征都差不多，但MoCo实际上并不关心次与次之间的负样本是不是一致，它只关心每次更新所有的负样本是否一致。

但是SimCLRv1和SimCLRv2都只做了分类的任务，但是MoCo在下游很多任务上都做了测试，而且还尝试了很多的数据集。所以MoCo系列的工作就更加CV friendly，所以他们投的都是CV的会议。而SimCLRv1投的是ICML，SimCLRv2投的是Neurips。所以说选中会议很很重要。


#### 5. [SwAV: Unsupervised Learning of Visual Features by Contrastive Cluster Assignments](https://proceedings.neurips.cc/paper/2020/file/70feb62b69f16e0238f741fab228fec2-Paper.pdf)

*Neurips 2020*

SwAV的全称呼是swaped assignment views，意思就是给定一张图片，如果能生成不同视角（views），那希望能通过一个视角得到的特征，来预测另外一个视角得到的特征，因为所有这些视角得到的特征，按道理来说都应该是很接近的。

这篇文章就是将对比学习和聚类结合到了一起。聚类本身也是一种无监督的特征学习方式，而且其也是希望相似的物体在聚类中心附近，而不相似的物体尽量推开，推到别的聚类中心。所以聚类本身和对比学习的目标以及做法都比较接近。

fig1解释了SwAV是如何将对比学习和聚类方法结合起来的。

![swav1]({{ '/assets/images/SWAV-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*fig 1. 左侧是对比学习的方法，右侧是SwAV的方法。左侧的方法已经很了解了，同一张图片$$x$$，做两次数据增强得到了$$x_1,x_2$$，然后通过一个编码器得到了$$z_1,z_2$$，再利用某种对比学习loss来计算相似性。而SwAV的想法是，直接将样本通过编码器得到的特征进行比较，这个方法太原始而且浪费资源了。即使是MoCo，他也只是使用了6万多个负样本，而并没有使用全部的128万个负样本，所以这还是个近似。SwAV想法是能否不近似呢，而且能否不直接与所有的负样本进行比较呢？SwAV提出，与其和所有的负样本比，比如和一些具有代表性的负样本比，也就是和聚类的中心比。聚类中心就是右图里的$$C$$，它实际上是个矩阵，维度是$$D \times K$$，其中$$D$$是特征维度，和$$z$$一样，也就是128维，而$$K$$是聚类中心的个数，取的是3000，这是之前很多聚类的论文里用的超参数。来看一下SwAV的前向过程。前面部分还是一样的，输入是$$x$$，经过两种数据增强之后，再经过编码器，得到了特征$$z_1,z_2$$。但现在不直接比较$$z_1,z_2$$来计算contrastive loss。相同的，我们还是认为$$z_1,z_2$$应该是相似的。但这个时候，我们使用聚类的方法，用$$z_1$$和$$C$$生成一个目标$$Q_1$$，用$$z_2$$和$$C$$生成一个目标$$Q_2$$，$$Q_1, Q_2$$就相当于ground truth，它们的大小为$$B \times K$$。而它的代理任务是，如果$$z_1,z_2$$很相似，那它们应该是可以互相去做预测的。所以说利用$$z_1$$点乘$$C$$应该可以去预测$$Q_2$$，反之亦然。所以说点乘之后的结果就是预测，而ground truth就是之前按照clustering分类而得到的$$Q_1,Q_2$$。所以通过以上这种换位预测（swapped prediction）的方式，SwAV可以对这个模型进行训练。*

使用这种方法的好处有哪些呢？
* 首先，正如这篇论文里所说，如果要使用负样本的话，就需要使用成千上万个负样本，而且即使如此，也只是一个近似。但如果现在去和聚类中心进行对比，只需要几百或者几千个聚类中心就足以表示所有的负样本了。因为原本数据集就没有那么多个类，ImageNet才1000类，COCO就80个类，所以说文中所用的3000个聚类中心足够用了。但是3000相对于MoCo里用的好几万的负样本来说，数量还是小了很多的。
* 其次，这些聚类中心是可以有明确的语义的，而之前的方法都是抽样的负样本，这样找到的负样本可能含有正样本，或者说样本类比也不均衡。所以不如使用聚类中心有效。

>对聚类方法感兴趣，可以看这篇论文一作的另一篇聚类的文章deep cluster（[Deep Clustering for Unsupervised Learning of Visual Features](https://openaccess.thecvf.com/content_ECCV_2018/papers/Mathilde_Caron_Deep_Clustering_for_ECCV_2018_paper.pdf)）。

来看看SwAV的效果。SwAV的效果非常好，不仅比MoCov1，MoCov2，SimCLRv1，SimCLRv2，InstDisc，CMC，CPC的效果都要好，而且比之后要说的BYOL，SimSiam的效果也要好。它实际上是目前使用ResNet50做backbone的工作里分最高的，75.3%。而比较的任务还是ImageNet上的linear classification，也就是预训练好之后，模型就冻住，只训练全连接层来做classification。

![swav2]({{ '/assets/images/SWAV-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*fig 2.*

fig2显示了效果。在对比学习出来之前，无监督学习效果都不到60%。在MoCo提出之后，后续的对比学习方法逐渐分数开始提高。但SwAV还是要比他们都要好。而且和有监督的baseline的76.5%的准确率差不了多少了。而且这个有监督的baseline是从头到尾都在ImageNet上训练，最后在测试集上测试结果的，而我们的SwAV在测试结果的时候模型参数是冻住的，都没有再微调。

SwAV的效果这么好，另一个重点是它使用了一个非常有用的trick：增加正样本的数量。原先的方法都是每张图片只有一个正样本，也就是说，对于一张$$224 \times 224$$的图片，先将其resize到$$256 \times 256$$，然后再随机crop两个$$224 \times 224$$的图片，这样两个图片是很大的，所以有很多的重合部分，而且也都基本包含了图片的全部信息。作者认为，这样的操作只能让模型学习到图片的全局信息，如果想学习到局部信息，那就需要一些小一点的crop。而且作者还希望能增加模型的正样本的数量，但又不希望增加模型的复杂度，所以他们最后的做法是，将在resize之后的图片上，crop两个$$160 \times 160$$的图片，再crop四个$$96 \times 96$$的图片，这样就有了6张图片了。作者认为那两张大的图片应该掌握了图片的全局信息，而小的crops则应该掌握更多的局部信息。

作者之后做了实验，发现这种multi-crop的方法很有用，除了对于监督学习来说效果变差以外，对于其它的无监督对比学习方法，比如说SimCLR等，正确率都有较大的增长。而且对于聚类的无监督学习方法，比如说deepCluster，multi-crop带来的提升就更大了。

所以说，如果去掉multi-crop这个技术，那实际上SwAV的效果和MoCov2的效果就差不多了，所以说使用聚类加上对比学习和仅仅使用对比学习的效果是差不多的，而multi-crop这个技术带来了很大的提升。所以后续很多工作实际上借鉴了multi-crop这个技术，而并没有过多关注SwAV这篇论文用到的方法。


#### 6. [CPCv2: Data-Efficient Image Recognition with Contrastive Predictive Coding](https://arxiv.org/pdf/1905.09272v3.pdf)

也是融合了很多的技巧，用了更大的模型，用了更大的图像块，做了更多方向上的下游任务，将batch norm换成了layer norm，还使用了更多的数据增强。直接将CPCv1在ImageNet上的40%的结果提升到了70%多，就和之前所说的其他方法都差不多了。


#### 7. [CMCv2: What Makes for Good Views for Contrastive Learning?](https://arxiv.org/pdf/2005.10243.pdf)

[CODE](https://hobbitlong.github.io/InfoMin/)

也叫做InfoMin。做的是一种分析性的工作。文章主要提出一个InfoMin的原则，最小化互信息，minimal mutual information。这看似和之前的方法的想法相反，因为之前都是想尽量最大化两个views之间的mutual information。作者的实际意思是，mutual information不能太大也不能太小，一味的太大会导致模型泛化效果不好，所以文章的重点在于如何找到一个合适的mutual information。作者按照InfoMin的原则，使用合适的数据增强，和合适的views之后，作者发现对于很多对比学习的方法效果都有提升。他们最后的结果在ImageNet上的正确率也有73%，也是很不错的。


#### 第三阶段

讲一下不用负样本的对比学习。第二阶段里的SWaV就已经有这个趋势了，它可以算作一个承上启下的工作，它也没有用负样本，而是使用的聚类中心，但是SWaV毕竟还是有一个明确的对比的对象那个的。但接下来要说的BYOL和SimSiam，实际上只涉及到了正样本，已经没有负样本或者聚类中心这样一个明确的对比对象了。

#### 1. [BYOL: Bootstrap Your Own Latent A New Approach to Self-Supervised Learning](https://papers.nips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf)

标题里的bootstrap的意思就是已经有了什么东西了，然后在它之上进行改造。latent就是特征的意思，latent，hidden，feature，embedding都是特征的意思，不同的表述方式罢了。BYOL的意思就是自己跟自己学，根本不需要负样本，左脚踩右脚上天。不用任何形式的负样本确实是一种很新颖的对比学习方式，所以在标题里也表明了这是一种全新的方式。

>为什么在对比学习中不用负样本就如此引人注目呢？因为在对比学习中，负样本是一个约束。如果在计算目标函数的时候只有正样本，那么目标只有一个，那就是让相似的样本的特征也尽可能的相似。那么这样就会有一个明显的shortcut解：不管输入是什么，模型直接都给出同样的特征。这个时候的loss是0，用这个shortcut解就可以完美解决问题，模型就躺平了，但实际上什么也没学到。而加上了负样本这个约束，就会使得相似的物体有相似的特征，不相似的物体的特征是不相似的，这样模型才有动力继续学习，因为如果它还是给出完全一样的特征，那么对于负样本来说loss就无穷大了。所以说负样本可以约束模型学习到有意义的解，防止learning collapse或者model collapse的情况发生。BYOL之所以神奇就是因为其不用负样本也可以进行对比学习，而且还达到了74.3%的准确率，已经很不错了。

下面来看看BYOL为什么有这样的魔力。





### 预测dense representation的对比学习方法（也就是representation learning所得到的特征的分辨率和输入图片的分辨率相同，对于每个像素点都学习一个特征）

#### 1. [Dense Contrastive Learning for Self-Supervised Visual Pre-training](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Dense_Contrastive_Learning_for_Self-Supervised_Visual_Pre-Training_CVPR_2021_paper.pdf)

[CODE](https://git.io/DenseCL)

*Xinlong Wang, Rufeng Zhang, Chunhua Shen, Taokong, Lei Li*

*CVPR 2021*

**Abstract**

到现在为止，大多数的自监督学习方法都是为了image classification任务设计以及优化的。这些pre-trained模型对于需要dense prediction的任务来说就是sub-optimal的了，因为image-level和pixel-level的prediction之间存在很大的差距。为了填补这个空缺，我们的目标是设计一个高效的dense自监督学习方法，通过考虑local features之间的关系，来学习pixel-level（或者叫做local features）的feature。我们提出的框架名字叫做dense contrastive learning（DenseCL），它是通过在pixel-level上优化输入一对图片之间的features来实现自监督学习的。

相对于baseline，MoCo-v2来说，我们的方法计算速度略微变慢了一点，但可以忽略不计（小于1%），但是对于需要dense prediction的下游任务，比如说object detection，semantic segmentation，instance segmentation等，效果都要好很多。而且要比目前的sota效果都要好很多：和MoCo-v2 baseline相比，我们在PASCAL VOC object detection任务上有2.0% AP的提升，在COCO object detection任务上有1.1% AP的提升，在COCO instance segmentation任务上有0.9% AP的提升，在PASCAL VOC semantic segmentation任务上有3.0% mIoU的提升，在Cityscapes semantic segmentation上有1.8% mIoU的提升。


**1. Introduction**

预训练现在已经成了CV领域的一个范式了。一个经典的预训练框架下，模型首先在一个大规模的数据集上预训练，然后再在下游任务上利用较少的数据进行微调模型。用监督方式在ImageNet上进行预训练来解决classification任务的这个方式，在CV领域流行了很多年。然而，在ImageNet上用监督方式进行的预训练使用的是classification任务，它与下游那些需要dense prediction的任务，比如说object detection，semantic segmentation等，本质上有着较大的差距。classification任务关注的是对于一张输入图片给一个标签，而后者需要针对每个像素点都进行计算。比如说，semantic segmentation的目标是对于每个pixel都给一个标签，object detection则是对于图中每个物体都要预测一个bounding box和标签。对于上述这个问题，一个直接的解决办法就是直接使用dense prediction的任务来进行预训练。然而，这些dense prediction任务的标注是很难的，所以说并不太可能能获取一个很大的有标签的数据集来做dense prediction的预训练。

最近，无监督的视觉预训练火了起来，其想要从没有标签的数据集中来学习图片的representations。有一些工作已经取得了不错的效果（SimCLR，MoCo等），其对于有些下游任务，和使用ImageNet上进行classification的预训练效果差不多甚至更好。然而，之前提到的ImageNet使用classification任务进行预训练从而导致由于预训练任务和下游dense prediction任务之间差距过大而导致的下游任务效果不好的这个问题仍然存在。首先，绝大多数自监督方法在预训练时使用的代理任务是instance discrimination，也就是每个image属于一类，这还是image-level的classification任务。其次，现存的无监督预训练方法的下游任务都是classification。然而，正如[Rethinking imagenet pre-training](https://openaccess.thecvf.com/content_ICCV_2019/papers/He_Rethinking_ImageNet_Pre-Training_ICCV_2019_paper.pdf)里所说的，在下游classification任务上效果好，不代表这个预训练的特征在object detection等任务上效果会好。因此，适用于dense prediction下游任务的自监督学习方法需求性是很强的。而且因为是无监督预训练方式，标注也是不需要的。一个简单的想法就是使用某种dense prediction的任务作为预训练任务，这样预训练任务和下游任务之间的距离就被缩小了。

受到有监督的dense prediction任务的启发（比如说semantic segmentation），我们提出了dense contrastive learning（DenseCL）方法来进行无监督的预训练。DenseCL将预训练里的任务看作一个pairwise contrastive learning任务而不是全局的image classification任务。首先，文章提出了一个dense projection head，其将backbone获得的feature map映射为dense feature vectors（也就是每个pixel都有个feature vector）。文章里所用的projection head因为输出的是dense vectors，所以天然的保留了位置信息（MoCo等方法里用的projection head是将从backbone得到的feature map通过global average pooling直接映射到了单个vector，来表示整张图的信息）。其次，文章通过获取不同views之间的correspondence信息来为每个local feature定义positive sample。为了构建一个非监督的目标函数，文章进一步定义了一个dense contrastive loss，将CPC那篇文章里定义的InfoNCE loss拓展为dense形式的。通过使用上述方法，文章使用一个fully convolutional network来实现了dense contrastive learning，这和很多dense prediction的下游任务是相同的。

文章的主要贡献总结如下：

* 文章提出了一个新的contrastive learning的框架：dense contrastive learning，其在像素层面或者在local features层面进行contrastive learning从而来学习特征。
* 通过使用提出的dense contrastive learning，文章定义了一个简单高效的自监督学习方法，来为下游dense prediction任务预训练特征，这个自监督预训练方法叫做DenseCL，其填补了自监督预训练和dense prediction下游任务之间的差异。
* DenseCL在很多dense prediction下游任务上的效果要比MoCo v2这个baseline好很多，包括object detection，instance segmentation，semantic segmentation等，而且要比使用监督方式在ImageNet上使用classification进行预训练的效果要更加好。


**2. Related Work**

*Self-supervised pre-training*

自监督预训练的成功，归功于两个方面：contrastive learning和pretext tasks。用来训练网络的目标函数一般就是reconstruction loss或者是contrastive loss（衡量不同views之间的co-occurrence）。contrastive learning目前是sota方法所使用的目标函数：也就是之前串烧里所说的那些contrastive learning的方法。

研究者们还探究了很多的pretext tasks来为了更好的学习到好的representation。这些代理任务包括：colorization，context autoencoders，inpainting，spatial jigsaw puzzle，discriminate orientation等等（可以参见self-supervised learning的那个网页）。这些方法在CV领域只获取了有限的效果。突破性的方法是在SimCLR这篇论文里所提出来的instance discrimination这样一个代理任务。这个代理任务认为同一张图片augment出来的不同的图片所得到的feature是相似的，和其他图片都是不相似的，也就是说，每张图片自成一类。在之后的工作里，contrastive learning框架+instance discrimination代理任务，从而进行无监督的预训练，已经成为了最流行的方式。文章提出的DenseCL也率属于自监督预训练模型框架，但DenseCL所学习到的特征对于dense prediction的下游任务比如说semantic segmentation，object detection等有着更好的效果。

*Pre-training for dense prediction tasks*

预训练已经在很多dense prediction任务上取得了不错的效果，比如说对于object detection（[You Only Look Once: Unified, real-time object detection]()，[Faster R-CNN：Towards real-time object detection with region proposal networks]()），semantic segmentation（[Fully convolutional networks for semantic segmentation]()）。这些模型都是先用ImageNet上的classification监督任务预训练好的模型进行微调得到的。已经有一些工作开始研究在模型结构层面探究在ImageNet上使用classification任务进行预训练和dense prediction下游任务之间的差距：[HyperNet: Towards accurate region proposal generation and joint object detection]()，[DetNet: Design backbone for object detection]()，[Deep high-resolution representation learning for human pose estimation]()，[EfficientDet: Scalable and efficient object detection]()。YOLO9000这篇文章提出在classification和detection数据集上来训练detector。[Rethinking imagenet pre-training](https://openaccess.thecvf.com/content_ICCV_2019/papers/He_Rethinking_ImageNet_Pre-Training_ICCV_2019_paper.pdf)说即使使用非常大的数据集（比如说Instagram数据集，要比ImageNet大3000倍）进行使用classification任务的预训练，其对于后续的object detection任务效果的提升是非常小的。最近也有工作表明在有标注的object detection数据集（比如说COCO）上进行监督方式的预训练，对于下游的object detection和semantic segmentation任务可以取得和使用ImageNet进行classification预训练差不多的效果。尽管针对dense prediction下游任务的预训练这样一个问题在DenseCL提出之前就被研究过，却很少有研究如何使用非监督的方式对dense prediction下游任务进行预训练的。[Contrastive learning of global and local features for medical image segmentation with limited annotations]()和[Unsupervised learning of dense visual representations]()这两篇文章也认为在local features层面进行contrastive learning是很重要的。但和这篇文章相比，一个最大的不同是它们利用的是geometric transformation来构造的positive pairs。这会带来以下几个问题：1）不灵活的data augmentation。需要为每个类型的data augmentation精心设计来保证dense matching。2）有限的应用场景。如果两个views之间的geometric transformation不可用的时候，这个方法就不行了，比如说从一个视频里截取的两张图片可以被认为是positive pairs，但就没有geometric transformation信息了。而本文章提出的方法和data pre-processing的过程毫无关系，因此可以实现快速而灵活的操作，而不需要考虑使用了什么data augmentation或者说图片是怎么采样得到的。

*Visual correspondence*

visual correspondence问题考虑的是从同一个场景得到的两张图片pixel之间的对应关系，这对于很多应用，比如说optical flow，structure-from-motion，visual SLAM，3D reconstruction等问题来说都是很重要的一部分。visual correspondence可以被描述为学习匹配的patches或者points之间的feature相似性。最近，有一系列基于CNN的方法被提出用来衡量不同图片patches之间的相似性，包括监督（[Universal correspondence network]()，[FCSS: Fully convolutional self-similarity for dense semantic correspondence]()）和非监督（[Unsupervised learning of dense shape correspondence]()，[Unsupervised feature learning for dense correspondences across scenes]()）的方法。前期的这些工作往往是针对某个特性的应用场景来学习correspondence，而且大多都是有明确的监督信号的。而DenseCL学习到的特征却可以在很多dense prediction任务中都被使用。


**3. Method**

**3.1 Background**

对于非监督的representation learning预训练，最关键的突破就是MoCo和SimCLR（还有之前contrastive learning串烧里说的那些），这两篇论文都是从无标签的数据上使用contrastive learning学习到好的representations。下面简要介绍一下sota的非监督learning框架，从而浓缩为一个框架。

*Pipeline*

给定一个无标签的数据集，我们使用instance discrimination作为代理任务，其会将每张图片的features都和其它图片的features区分开。对于每张图片，使用随机的data augmentation得到随机的views，然后将每个view都喂给一个encoder从而得到features。encoder由两个部分组成：backbone和projection head，两者是按顺序连接起来的。backbone是我们在预训练结束之后，对于下游任务来说还需要微调的东西，而projection head在预训练使用完之后就不需要了。encoder是使用contrastive loss进行优化的，如fig1里的(a)所示。

*Loss Function*

如MoCo那篇论文里所说的，contrastive learning的过程可以被理解为一个字典查找的过程。对于每个被编码的query $$q$$，有一个集合的被编码的keys $$\lbrace k_0, k_1, \cdots \rbrace$$，其中有一个$$k_+$$是positive key，是匹配$$q$$的。被编码的query和keys是不同的views经过backbone和projection head所得到的，对于每个query，其positive是从同一张图片的不同的view编码得来的，而其余的keys都是不同图片的views。一个contrastive loss function，叫做InfoNCE，可以将query和positive变得很相似，而和其它的keys变得不相似：

$$\mathcal L_q = -log \frac{exp(q \cdot k_+ / \tau}{exp(q \cdot k_+) + \Sigma_{k_-} exp(q \cdot k_- / \tau)}$$

其中$$\tau$$是温度系数，是个超参数。

![dense1]({{ '/assets/images/DENSE-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*fig 1. *


**3.2 DenseCL Pipeline**

文章提出了一个新的自监督的学习框架，是为了dense prediction的下游任务而设计的，这个框架叫做DenseCL。DenseCL将现有的自监督representation学习框架扩展到了dense的范畴。和2.1 background里提到的框架比，核心区别在于encoder和loss function的设计不一样。给定一个输入的view，其先通过一个backbone（比如ResNet或者VGG等CNN框架）得到dense feature map，然后再通过一个如下描述的projection head计算输出。这个projection head由两个平行的sub-heads组成：global projection head和dense projection head。global projection head就和之前提到的那些方法（MoCo等）里的projection head一样，获取的是view的全局整体特征。而dense projection head的输入是从backbone获取的feature map，而输出仍然是dense feature vectors。

dense projection head的设计是。不再使用average global pooling和MLP，而是使用$$1 \times 1$$的convolutional layers。backbone和这两个平行的projection head是端到端进行训练的，通过优化一个联合的同时在global features和local features层面上的pairwise contrastive (dis)similarity loss来实现的。


**3.3 Dense Contrastive Learning**

我们通过将原先的contrastive loss function拓展到一个dense的情况来进行dense contrastive learning。对于每个被编码的query $$r$$，我们定义一个集合的被编码的keys $$\lbrace t_0, t_1, \cdots \rbrace$$。然而，现在每个query并不再表示整张view，而表示的是每个view的一个局部区域。具体来说，每个被编码的query表示的是由dense projection head得到的ddense feature vector的一个像素点，其中dense feature vector的大小为$$S_h$$和$$S_w$$，它两可以相等也可以不等，简单起见，设置$$S_h = S_w = S$$。接下来我们要为每个从dense feature vector找到的query，得到它的negative keys和positive keys。其中negative keys比较简单，就是其它输入图片的views通过backbone和global projection head所得到的features，记为$$t_-$$，而positive keys则是我们需要重点关注的，其是从同一张图片的另一个views通过backbone和dense projection head所得到的大小为$$S_h \times S_w$$的dense feature vectors里的$$S^2$$个像素点中的一个，然而如何找到这一个在接下来的2.4里说，现在假设我们已经找到了，记为$$t_+$$。从而这个dense contrastive loss定义如下：

$$\mathcal L_r = \frac{1}{S^2} \Sigma_{s} -log \frac{epx(r^s \cdot t^s_+ / \tau)}{epx(r^s \cdot t^s_+) + \Sigma_{t^s_- epx(r^s \cdot t^s_- / \tau)}$$

其中$$r^s$$表示的是dense feature vector的第$$s$$个被编码的query（一共有$$S^2$$个）。

总体来说，DenseCL的loss就是：

$$\mathcal L = (1 - \lambda) \mathcal L_q + \lambda \mathcal L_r$$

其中$$\lambda$$是控制loss比例的超参数。在下面的实验里$$\lambda=0.5$$。


**3.4 Dense Correspondence across views**

我们对于输入的一张图片的两个views来找到它们之间的dense correspondence。对于每个view，backbone获取了feature map，记为$$\pmb F \in \mathbb R^{H \times W \times K}$$，基于这个feature map，dense projection head获取了dense feature vectors $$\Theta \in \mathbb R^{S_h \times S_w \times E$$}$$。我们这里设置$$S_w = S_h = S$$，但其实它们也是可以不同的。dense correspondence则是在两个views的dense feature vectors，$$\Theta_1$$和$$\Theta_2$$，之间获取的。文章是使用backbone feature maps $$\pmb F_1$$和$$\pmb F_2$$来匹配$$\pmb \Theta_1$$和$$\pmb \Theta_2$$的。首先将$$\pmb F_1$$和$$\pmb F_2$$使用adaptive average pooling降采样到大小为$$S \times S$$，然后再计算它两之间的cosine similarity矩阵 $$\Delta \in \mathbb R^{S^2 \times S^2}$$。而$$\pmb \Theta_1$$和$$\pmb \Theta_2$$之间的匹配规则是：每个view的dense feature vector的每个像素点在另一个view里的匹配点就是和这个像素点similarity最高的那个像素点。具体来说，对于$$\pmb \Theta_1$$，我们想找到$$\pmb \Theta_1$$里每个像素点在$$\pmb \Theta_2$$里的对应点，就是沿着dimension=2来计算$$\Delta$$的argmax：

$$c_i = argmax_j sim(\pmb f_i, \pmb f_j^{'})$$

其中$$\pmb f_i$$是$$\pmb F_1$$的第$$i$$个feature vector，而$$\pmb f_j^{'}$$则是$$\pmb F_2$$的第$$j$$个feature vector。$$sim(u,v)$$表示$$u$$和$$v$$之间的cosine similarity。上述结果表明$$\pmb \Theta_1$$的第$$i$$个像素点对应的是$$\pmb \Theta_2$$里的第$$c_i$$个像素点。上述整个过程可以使用矩阵计算很快速的完成。


### 2. [Crafting Better Contrastive Views for Siamese Representation Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_Crafting_Better_Contrastive_Views_for_Siamese_Representation_Learning_CVPR_2022_paper.pdf)

*CVPR 2022*


### 3. [Instances as Queries](https://openaccess.thecvf.com/content/ICCV2021/papers/Fang_Instances_As_Queries_ICCV_2021_paper.pdf)

[CODE](https://github.com/hustvl/QueryInst)

*ICCV 2021*




## Uncategorized Papers

### 1. [Neural Ordinary Differential Equations](https://proceedings.neurips.cc/paper/2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf)

*NeurIPS 2018*


### 2. [Rethinking ImageNet Pre-training](https://openaccess.thecvf.com/content_ICCV_2019/papers/He_Rethinking_ImageNet_Pre-Training_ICCV_2019_paper.pdf)

*ICCV 2019*


### 3. [MA-CLIP: Towards Modality-Agnostic Contrastive Language-Image Pre-training](https://openreview.net/forum?id=ROteIE-4A6W)

*Under Review ICLR 2022*


### 4. [MIST: Multiple Instance Spatial Transformer](https://openaccess.thecvf.com/content/CVPR2021/html/Angles_MIST_Multiple_Instance_Spatial_Transformer_CVPR_2021_paper.html)

*CVPR 2021*


### 5. [Deep Clustering for Unsupervised Learning of Visual Features](https://openaccess.thecvf.com/content_ECCV_2018/papers/Mathilde_Caron_Deep_Clustering_for_ECCV_2018_paper.pdf)

*ECCV 2018*


---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*

