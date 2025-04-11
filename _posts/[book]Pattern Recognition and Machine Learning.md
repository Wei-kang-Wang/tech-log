---
layout: post
comments: false
title: "[book]Pattern Recognition and Machine Learning"
date: 2021-12-17 01:09:00
tags: book-reading
---

> 这是*PATTERN RECOGNITION AND MACHINE LEARNING*的读书笔记.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

## Mathmatical Notation

calculus，linear algebra和probability theory是阅读这本书的数学基础。

在这本书里，向量vectors使用小写加粗罗马字母表示，比如 $$\pmb{x}$$ ，所有的vectors都是列向量。$$\pmb{x}^T$$ 表示行向量，因为$$T$$表示转置。大写的加粗罗马字母表示矩阵，比如 $$\pmb{M}$$ 。$$(w_1, \cdots, w_M)$$表示有$$M$$个元素的行向量，从而对应的列向量为 $$\pmb{w} = (w_1, \cdots, w_M)^T$$。

记号$$\left[a, b \right]$$表示从$$a$$到$$b$$的闭区间，而$$(a,b)$$则表示从$$a$$到$$b$$的开区间。

$$M \times M$$的identity矩阵（也叫做unit矩阵）记为$$\pmb{I_M}$$。

一个functional被记为$$f\left[y\right]$$，其中$$y(x)$$也是一个function。

$$g(x) = O(f(x))$$表示$$\lVert f(x) / g(x) \rVert$$在$$x \longrightarrow \infty$$的时候是有界的。

函数$$f(x,y)$$关于随机变量$$x$$的期望被记为$$\mathbb{E_x} \left[f(x,y)\right]$$。如果不造成歧义的话，可以省去下标，比如$$\mathbb{E} \left[g(x)\right]$$。如果$$x$$的分布是在另一个随机变量$$z$$的条件下的，那么这个条件期望被记为$$\mathbb{E_x} \left[f(x) \vert z\right]$$。类似的，variance被记为$$var\left[f(x)\right]$$。而对于向量型的随机变量，covariance则被记为$$cov\left[\pmb{x}, \pmb{y}\right]$$。我们使用$$cov\left[\pmb{x}\right]$$来表示$$cov\left[\pmb{x}, \pmb{x}\right]$$。

如果我们对于某个$$D$$维的向量$$\pmb{x} = (x_1, \cdots, x_D)^T$$有$$N$$个值$$\pmb{x_1}, \cdots, \pmb{x_N}$$，我们可以将其放在一个矩阵$$\pmb{X}$$里，其第$$n$$行就是向量$$\pmb{x_n}^T$$。如果是一维的随机变量，我们上述的矩阵就变成了向量，记为**x**，其是一个列向量，第$$n$$个位置的元素是$$x_n$$。注意到**x** 和$$\pmb{x}$$是不一样的，前者是$$N$$个随机向量放在一起组成的列向量，后者自身就是一个随机向量。


## 1. Introduction

考虑一个识别手写数字图像的例子。每个数字图像都是一个$$28 \times 28$$的pixel矩阵，我们将其拉成一个长度为784的向量$$\pmf{x}$$。我们的目的是设计一个算法来自动学习不同的手写数字图片的特征，从而能自动判定输入的手写数字的值。hand-crafted特征是可以用来辨别手写数字图片的，但其对于复杂的情况效果不好，而且需要大量手动设计的features。

用来训练上述这个自动检测的模型的参数的数据的集合$$\lbrace \pmb{x_1}, \cdots, \pmb{x_N} \rbrace$$叫做训练集，training set。在我们这个例子里，每个手写数字图片都有个标签表示类别，我们可以用target vector $$\pmb{t}$$来表示。每个图片$$\pmb{x}$$都有一个target vector $$\pmb{t}$$。

运行一个机器学习模型可以被看成一个函数$$pmb{f(x)}$$，其输入是一个$$\pmb{x}$$，而输出一个向量$$\pmb{y}$$，这个$$\pmb{y}$$和$$\pmb{t}$$的encode方式是一样的。这个函数$$\pmb{f(x)}$$的具体形式是在训练过程中由训练数据来决定的，称为training phase或者learning phase。当训练完成后，模型就可以对在训练过程中没有见到过的新的数据进行结果预测，这些新的数据被称为测试集，test set。模型能够对于没有在训练集中见过的新的数据给出正确的输出结果的能力称为泛化性，generalization。

>泛化性是机器学习的核心问题

对于大多数的问题，原输入的数据需要经过某些预处理，preprocessing，操作来将其转换到新的变量空间里，从而机器学习模型能更好的操作它们。这个preprocessing操作有时候也叫做feature extraction。需要注意，测试数据也需要经过和训练数据一样的preprocessing才可以。

>我们需要对preprocessing很注意，因为其会舍去一些信息，如果被舍去的信息很重要的话，就会影响效果。

训练数据既有input vectors又有它们对应的target vectors的问题称为监督学习问题，supervised learning problem。这个对于手写数字图片预测离散类别的任务称为classification problem。如果输出含有一个或者多个连续的值，那就称为regression problem。

训练数据只有input vectors $$\pmb{x}$$本身，而并没有相对应的target vectors的问题称为非监督学习问题，unsupervised learning problem。非监督学习问题可以是在数据中找到由相似数据构成的clusters，这个任务叫做clustering；或者找到输入数据的分布，叫做density estimation；或者将输入数据从一个高维空间投射到2或者3维空间从而可以进行visualization。

reinforcement learning旨在在给定的条件下做出合适的actions从而使得某种reward最大化。这种问题超出了本书的讨论范围，可以参考Sutton和Barto的书：Reinforcement learning: An Introduction。


### 1.1 Example: Polynomial Curve Fitting

我们考虑一个简单的regression problem。假设我们观察到的数据是$$x \in \mathbb{R}$$，以及每个值对应的target值$$t \in \mathbb{R}$$，我们要做的是学习一个模型，输入是$$x$$，输出是$$t$$。

我们这个例子的背后生成这些数据的真实的函数是$$t = sin(2 \pi x)$$，然后再在$$t$$上加了一些噪声。

假设我们的训练集有$$N$$个$$x$$的观察值，写成**x** = $$(x_1, \cdots, x_N)^T$$，每个$$x$$还有对应的$$t$$，记为 **t** = $$(t_1, \cdots, t_N)^T$$。

我们的目的是对于新的$$\hat x$$做出正确的预测$$\hat t$$，这需要我们能够隐式的学到训练数据背后的函数$$t = sin(2 \pi x)$$。但这是很困难的，因为我们需要从有限的数据集上进行泛化。

现在，我们考虑用一个多项式函数来拟合数据：

$$y(x, \pmb(w)) = w_0 + w_1 x + w_2 x^2 + \cdots + w_M x^M = \Sigma_{j=0}^M w_j x^j$$

其中$$M$$是多项式的阶，多项式系数$$w_0, w_1, \cdots, w_M$$合起来表示为$$\pmb{w}$$。

注意到我们上述的多项式函数模型，对于输入$$x$$来说是非线性的，但对于参数$$\pmb{w}$$来说是线性的。这样的对于需要学习的参数是线性的模型，称为linear models，会在第三和第四章里说。

上述参数$$\pmb{w}$$可以通过将多项式函数拟合训练数据的方式来学习。也就是最小化$$y(x, \pmb{w})$$和$$t$$之间的不匹配程度。一个最为常见的error function来衡量这种不匹配程度就是对于每个数据点$$x_n$$以及对应的$$t_n$$，计算预测的值$$y(x_n, \pmb{w})$$与$$t_n$$之间的差的平方，在对于所有的数据点求和：

$$E(\pmb{w}) = \frac{1}{2} \Sigma_{n=1}^N \lbrace y(x_n, \pmb{w}) - t_n \rbrace ^2 $$

我们可以直接通过$$E(\pmb{w})$$对$$\pmb{w}$$求导来得到使得$$E$$最小的$$\pmb{w^{\ast}}$$。

从而剩下的问题就是如何选取多项式函数的阶$$M$$。这涉及到一个重要的内容：model selection或者叫model comparison。

我们发现，对于比较小的$$M$$，其不具备拟合我们这个复杂函数生成的数据的能力，比如说$$M=0,1$$。对于$$M=3$$，效果不错。而对于$$M=9$$，我们的多项式函数能对于训练集每个值都做出完全正确的预测，也就是$$E(\pmb{w})=0$$。但其对于测试数据效果并不好，而且并没有很接近真实的$$sin(2 \pi x)$$。这种现象称为过拟合，over-fitting。

对于我们这个例子来说，$$M$$大的多项式函数应该要包含$$M$$小的多项式函数，而且因为真实的函数是$$sin(2 \pi x)$$，而这个函数的Taylor expansion含有无穷的级数，也应该是$$M$$越大越好。而且我们选取的loss是有closed-form的解的，并不是对于neural networks的情况那样，因为复杂的neural networks会收敛到比较差的local minima，而不收敛到全局minima（也就是说虽然复杂的neural network包含简单的neural network，但他不好被训练），而对于我们这个例子，全局minima直接用closed-form的解给出来了。所以说这里出现over-fitting的原因是target vector有噪音，从而大的$$M$$就会对这些噪声过于感兴趣，从而学到了复杂而非真实的函数。

上述over-fitting的问题对于固定的$$M$$可以通过增加训练数据的个数而得到解决。但是，我们并不应该通过训练数据的个数来决定模型的复杂度，我们应该通过问题本身来确定模型的复杂度。我们在之后的章节1.2.5里可以看到，我们这种用least squares来找模型参数的方法实际上是maximum likelihood方法的一个例子。而出现的over-fitting的情况可以看作maximum likelihood一个普遍会出现的问题。通过使用Bayesian方法，over-fitting问题可以被缓解。而且在Bayesian模型的框架里，模型的参数的个数和训练数据的个数没关系，模型的有效参数可以自适应训练数据的个数。

而我们现在还没涉及到Bayesian模型，所以我们来看看还有什么简单的方法能使得我们对于较少的训练时据仍然可以使用复杂的模型。一个常见的用来控制over-fitting的技术是regularization，其给loss增加惩罚项来控制模型的参数的值。最简单的regularization是：

$$\tilde{E} (\pmb{w}) = \frac{1}{2} \Sigma_{n=1}^N \lbrace y(x_n, \pmb{w}) - t_n \rbrace ^2 + \frac{\lambda}{2} \lVert \pmb{w} \rVert ^2$$

其中$$\lVert \pmb{w} \rVert ^2$$ = w_0^2 + \cdots, w_N^2$$，$$\lambda$$来控制regularization的重要性。

regularization方法在统计学里叫做shrinkage methods，因为其控制减小模型参数的值。而使用上述平方和的方式的regularization叫做ridge regression，在neural networks里，叫做weight decay。

我们如果希望找到合适的模型复杂度或者regularization的系数，有一种方法就是，将我们的数据，分为训练数据，和validation数据，也叫做hold-out集合，用来找到合适的超参数，比如说这个例子里的模型复杂度$$M$$和正则项系数$$\lambda$$。但是数据是很可贵的，用这种方法是对数据的一种浪费，应该要寻找到更高效的方法。


### 1.2 Probability Theory

机器学习的核心概念就是如何取衡量不确定性。这种不确定性既出现在target vectors上的随机noise，也因为训练集合是有限的。probability theory为机器学习中衡量不确定性给了一个系统性的描述框架。和1.5里的decision theory结合起来，我们就可以对于给定的信息来做出最优的判断了（即使信息是不完整或者有不确定性）。

我们考虑一个简单的例子。假设我们有两个盒子，一个红一个蓝，在红盒子里有2个苹果6个橙子，蓝盒子里有3个苹果1个橙子。假设我们随机选一个盒子，然后再随机拿一个水果出来，记录下水果的种类之后再将水果放回。重复这个操作很多次。假设我们有40%的概率选中红盒子，60%的概率选中蓝盒子。

在上述例子里，我们将选中哪个盒子这样一个量设定为一个随机变量，$$B$$。这个随机变量有两个值，$$r$$和$$b$$。类似的，我们最后拿到的水果的种类也是一个随机变量，记为$$F$$，其也有两个值，$$a$$和$$o$$。

我们有$$p(B=r) = 4/10$$以及$$p(B=b) = 6/10$$。

两个重要的probability rule是sum rule和product rule。

考虑两个随机变量$$X$$和$$Y$$。假设$$X$$可以取的值为$$x_i$$，其中$$i=1,\cdots,M$$，而$$Y$$可以取的值为$$y_j$$，其中$$j=1,\cdots,L$$。考虑我们一共选取$$N$$个点，而$$X=x_i$$和$$Y=y_j$$的点的个数为$$n_{ij}$$，不管$$Y$$的值的情况下$$X=x_i$$的个数为$$c_i$$，不管$$X$$的值的情况下$$Y=y_j$$的个数为$$r_j$$。

$$X=x_i$$并且$$Y=y_j$$的概率$$p(X=x_i, Y=y_j)$$叫做$$X=x_i$$，$$Y=y_j$$的联合概率，joint probability。其的值为：

$$p(X=x_i, Y=y_j) = \frac{n_{ij}}{N}$$

$$X=x_i$$的概率$$p(X=x_i)$$的值为：

$$p(X=x_i) = \frac{c_i}{N}$$

我们还有$$c_i = \Sigma_{j} n_{ij}$$，所以：

$$p(X=x_i) = \frac{\Sigma_{j} n_{ij}}{N} = \Sigma_{j} p(X=x_i, Y=y_j)$$

这就是sum rule。$$p(X=x_i)$$叫做marginal probability。

如果我们只考虑$$X=x_i$$的那些点，那么在这些点中$$Y=y_j$$所占的比例被称为给定$$X=x_i$$的情况下$$Y=y_j$$的条件概率，conditional probability，记为$$p(Y=y_j \vert X=x_i)$$，其的值为：

$$p(Y=y_j \vert X=x_i) = \frac{n_{ij}}{c_i}$$

从而我们可以得到：

$$p(X=x_i, Y=y_j) = p(Y=y_j \vert X=x_i) p(X=x_i)$$

这叫做product rule。













---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*
