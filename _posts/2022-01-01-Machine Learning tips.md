---
layout: post
comments: false
title: "Machine Learning Tips"
date: 2021-12-18 01:09:00

---

> This post of various machine learning knowledge.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

## 1. Batch Normalization 批标准化

**1.1 Feature Scaling (or Feature Normalization)**

先从Feature scaling，或者叫做Feature normalization说起。

假设$$x_1$$和$$x_2$$的数据差距很大：

$$x_1$$的范围是个位数，而$x_2$的范围是1e+3左右，假设$$x_1$$所乘的weight是$$w_1$$，$$x_2$$所乘的weight是$$w_2$$，那么因为$$x_2$$会比$$x_1$$大很多，所以$$w_2$$和$$w_1$$同等变化，$w_2$的影响会更大。

将$$w_1$$和$$w_2$$对于loss的影响作图，如fig 1所示。在$$w_1$$方向上的梯度，在$$w_2$$方向上的梯度较大。这样会让training变得不容易，因为要在不同的方向上给不同的learning rate。

所以说我们通过feature scaling的方式把不同的feature做normalization，从而使得loss关于不同参数的图看起来像正圆形（2维），从而就可以设置一个training rate了。如fig 1右侧。

![feature scaling]({{ '/assets/images/batch_normalization_1.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Feature Scaling.*

**Feature Scaling的算法**

对于第i个维度的每一个data，都减去该维度的mean，再除以该维度的standard deviation。如fig 2所示。

![feature scaling algorithm]({{ '/assets/images/batch_normalization_2.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. Feature Scaling Algorithm.*

**1.2 Batch Normalization**

而在deep learning中，我们经常会对hidden layer输出的值做feature scaling：也就是input在做完feature scaling之后进入layer1，经过layer1运算后的output在进入layer2之前也做一次feature scaling，经过layer2运算后的output在进入layer3之前也做一次feature scaling，一直这样下去。如fig 3所示。不仅仅对input做scaling，还对neural network的中间层做scaling，粗糙的说，这就叫做batch normalization（实际上还有两个要学习的参数）。

![feature scaling nn]({{ '/assets/images/batch_normalization_3.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 3. Feature Scaling for neural network.*

按照上述的方式对于每个layer的输出都做一次feature scaling的方法对于neural network的训练是很有好处的。因为它使得internal covariante shift这个现象减弱了。

>internal covariate shift问题：
>将每个人想象成neural network的一层，想象三个人通过话筒传话，对于中间的人来说，如果左边的人让他把话筒降低，右边的人让他把话筒升高，他就会将左手话筒放低，右手话筒升高，这样的话，他所做的移动太多，两侧都要调整很多，反映到neural network上，就会导致训练的结果不好，如fig 3所示。实际上internal covariate shift的意思就是在深层网络训练的过程中，由于网络中参数变化而引起内部结点数据分布发生变化。

当一个neural network很深的时候，后面的layer会根据前面的layer输出的结果做出调整，但是前面的layer会根据更前面的layer的变动进行调整，这样每次训练，见到了新的数据，所有的layer都动了，这样很难训练到一个好的结果。

为了解决这个问题，过去传统的方式就是将learning rate设置的小一点，这样每次模型的weights不会进行很大的改变（即使因为internal covariate shift导致gradients很大，或者方向不正确）。但这样会导致模型的训练很慢。

而feature scaling则很好的减弱了这个问题，其对每个layer的output都做了标准化的normalization，从而每个layer的输入都有mean=0，covariance=1，这样一个固定的statistics。

normalization可以放在activation function之前，也可以放在其之后。将normalization放在activation function之前，对于某些activation functions，比如说tanh或者sigmoid，因为其会有saturation region，而我们最好避免input落在saturation region里，因为这样会使得训练不动。所以先做normalization可以确保其不在saturation region里。

![bn]({{ '/assets/images/batch_normalization_4.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 4. Batch Normalization.*

在batch normalization中，我们计算的mean和variance都是根据一个batch来算的，因此我们希望这个batch就代表了整个training size，所以说实践上batch size不能太小，太小的话效果不好。

而batch normalization和我们之前所说的feature scaling的区别就在于它仍然还有可训练的参数。因为有时候我们不希望经过normalization之后的mean都是0，variance都是1，所以会为它增加两个可学习的参数，将normalization之后的数据再进行一步线性操作。如fig 5所示。

![fbn]({{ '/assets/images/batch_normalization_5.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 5. Full batch normalization.*

上述是training过程中batch normalization的做法，而在testing的时候，testing的data是一个个进来的，所以没法计算mean和variance。方法是在training的时候，记录下来每一个batch对应的$$\mu$$和$$\sigma^2$$，然后利用某种加权和的方式（比如说exponential weighted average, $$E_{t+1} = \alpha a + (1-\alpha)E_{t}$$）计算出整个training集的$$\mu$$和$$\sigma^2$$，从而在test时候用。

如fig 6所示。

![fbnt]({{ '/assets/images/batch_normalization_6.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 6. Batch normalization at testing time.*

使用batch normalization带来的好处：
* 减少训练实际，使得训练很深的网络变得可能。因为internal covariate shift被减弱了，所以可以用大的learning rate了。而且因为exploding/vanishing gradient的现象被减轻了（特别是对于activation function有saturation part的情况，比如说tanh或者sigmoid）
* initialization的影响减弱了。

**1.3 [Andrew Ng对于BatchNormalization的解释](https://www.youtube.com/watch?v=nUUqwaxLnWs)**

* Intuition 1: 对input做feature scaling可以帮助训练，而同样的操作也可以针对neural networks的中间层输出来做。

* Intuition 2: 对于一个网络来说，如果输入的training data的mean和variance一直在变，这个网络很难训练好，或者说如果训练集和测试集的mean和variance差别很大，那么这个训练好的网络在测试集上效果也不会好。比如说，仅仅在黑猫图片上训练的检测猫的神经网络，对于彩色猫的测试集效果不好。那么对于一个神经网络来说，对于较深的层来说，它的输入就是较浅的层的输出，而因为较浅的层也有可学习的权重，所以这个输出一直在变化，也就是covariance shift，这样导致较深层的输入一直在变化，从而很难学习。而batch normalization的作用就是让深层的输入较少的收到浅层的输出的影响。

* Intuition 3: batch normalization还有一些regularization的作用。因为每一层的mean和variance都是根据一个batch算出来的，而不是所有的数据，这引入了一些noise。




## 2. Singular Value Decomposition (SVD) 奇异值分解

**2.1 Eigendecomposition**

SVD的原始想法起源于eigendecomposition，所以我们先来介绍eigendecomposition。

对于一个方矩阵$$A \in C^{n \times n}$$，如果存在向量$$v \in C^{n}$$和标量$$\lambda \in C$$使得$$Av = \lambda v$$，那么就称$$\lambda$$为$$A$$的eigenvalue，而$$v$$为这个eigenvalue所对应的一个eigenvector。可以很直接地看出，对eigenvector进行scaling并不影响结果。

求解eigenvalue的方式是，因为$$Av = \lambda v$$，所以$$(A - \lambda I)v = 0$$。而如果$$det(A-\lambda I) \neq 0$$，那么$$A-\lambda I$$就是invertible的，那么就会得到$$v=0$$，与eigenvector不能为0矛盾。从而$$det(A-\lambda I)=0$$。这个式子是关于$$\lambda$$的$$n$$阶多项式，从而在复数域内有$$n$$个解。而且可以看出来，所有的eigenvalues的乘积等于$$det(A)$$（重复的根也要重复乘上）。

一个方矩阵$$A$$的所有eigenvalues构成的集合叫做这个矩阵的spectrum，记为$$\sigma(A)$$。

下面我们只考虑实矩阵$$A \in R^{n \times n}$$。其有以下的性质：

* 如果$$\lambda$$是$$A$$的eigenvalue，那么$$\bar{\lambda}$$也是$$A$$的eigenvalue。
* 不同的eigenvalues对应的eigenvectors是线性无关的。

对于对称实矩阵$$S \in R^{n \times n}$$，$$S^{T} = S$$，我们有很多特殊的很好的性质。

如果一个对称实矩阵$$S$$满足对于任意$$x \in R^{n}$$都有$$x^{T}Sx \geq 0$$，那么称$$S$$是positive semi-definite的（如果是严格大于，那就是positive definite）。

* 所有的eigenvalues都是实数。
* 对应于不同的eigenvalues的eigenvectors是orthogonal的（不仅限于linear independent了）。
* 如果$$S$$是positive semi-definite（positive definite）的，那么其所有的eigenvalues都是非负（正）的。
* 如果$$\lambda_1$$和$$\lambda_n$$分别是最大的和最小的eigenvalues，那么

$$\lambda_1 = max_{||x||=1} \langle x, Sx \rangle$$

以及

$$\lambda_n = min_{||x||=1} \langle x, Sx \rangle$$

对于eigenvalues的计算公式，我们有$$det(S-\lambda I) = 0$$，也就是关于$$\lambda$$的方程。假设这个关于$$\lambda$$的方程只有$$k$$个不同的根，即$$det(S-\lambda I) = (\lambda - \lambda_1)^{n_1} (\lambda - \lambda_2)^{n_2}...(\lambda - \lambda_k)^{n_k}=0$$。称$$n_i$$为eigenvalue $$\lambda_i$$的代数重数。对于特定的一个eigenvalue $$\lambda_i$$，我们有$$Sv = \lambda_i v$$，即$$(S-\lambda_i I) v = 0$$，从而其对应的eigenvectors都在矩阵$$S-\lambda_i I$$的null space里，这个null space对应的维数$$m_i$$称为eigenvalue $$\lambda_i$$的几何重数。我们有代数重数不小于几何重数这样的限制。

对称实矩阵的好处就是，对于任何eigenvalue，都有其代数重数等于几何重数，也就是说假设我们将对称实矩阵$$S$$的$$n$$个eigenvalues排成一个对角矩阵$$\Sigma$$（带重复的）：

$$
\begin{pmatrix}
\lambda_1 & 0 & 0 & \cdots & 0 \\
0 & \lambda_2 & 0 & \cdots & 0 \\
 & & \vdots & & \\
0 & 0 & \cdots & 0 & \lambda_n \\
\end{pmatrix}
$$

>为什么对称的实数方矩阵满足每个eigenvalue的代数重数等于几何重数，可以通过归纳法来证明。

我们可以对于每个eigenvalue都找到其对应的eigenvector，而且因为每个eigenvalue的几何重数等于代数重数，所以说对于重复的eigenvalue，仍然可以找到足够个数的线性无关的eigenvectors。将上述这些eigenvectors按照列排成一个矩阵，也就是每一列都是一个eigenvector，再将每一列的norm都变成1，那么这个矩阵就是一个orthonormal矩阵，记为V，有性质$$VV^T = V^TV$$。我们有$$SV = V \Sigma$$。从而$$S = V \Sigma V^T$$。而这个形式就是对称实矩阵$$S$$的eigendecomposition，满足中间的对角矩阵$$\Sigma$$都是$$S$$的eigenvalues，而$$V$$的每一列都是对应的eigenvector，而且满足$$V$$是个orthonormal的矩阵。


>对于不是对称的一般的实矩阵$$A \in R^{n \times n}$$，也有上述类似的decomposition。会存在orthogonal matrix $$V \in R^{n \times n}$$，和$$\Sigma \in R^{n \times n}$$，满足$$A = V \Sigma V^T$$，而$$\Sigma$$是一个block对角矩阵，也就是说$$\Sigma = diag(A_1, A_2, \cdots, A_m, 0, \cdots, 0)$$，而每个$$A_i$$都是一个二维的skew-symmetric矩阵，满足

$$
\begin{pmatrix}
0 & a_i \\ 
-a_i & 0 \\ 
\end{pmatrix}
$$

>其中$$a_i$$是实数。

**2.2 Singular Value Decomposition**

现在我们再来看singular value decomposition。singular value decomposition是eigenvalue decomposition对于一般矩阵$$A \in C^{m \times n}$$的推广。

SVD的正式定义为：对于矩阵$$A \in R^{m \times n}$$，$$m \geq n$$，并且$$rank(A) = p$$，我们有三个矩阵$$U \in R^{m \times p}$$，$$V \in R^{n \times p}$$和$$\Sigma \in R^{p \times p}$$具有以下性质，$$U$$和$$V$$的列都是orthonormal的，且$$\Sigma = diag(\sigma_1, \sigma_2, \cdots, \sigma_p)$$，$$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_p$$，满足：

$$A = U \Sigma V^T$$

注意到，上述SVD的定义是对eigenvalue decomposition的推广，后者将一个对称的方阵分解为：$$A = V \Sigma V^T$$，而$$V$$是orthonormal的，$$\Sigma=diag(\lambda_1, \cdots, \lambda_n)$$。SVD使得分解任何非满秩的非方矩阵成为了可能。从下述证明我们也可以看出其实SVD也是从eigenvalue decomposition推导出来的。

**Proof:**

对于一个矩阵$$A \in R^{m \times n}$$，$$m \geq n$$而且$$rank(A)=p$$，那么我们有

$$A^{T}A \in R^{n \times n}$$

是对称矩阵而且是positive semi-definite的。所以说它可以被进行eigenvalue decomposition，并且它的eigenvalues是非负的，表示为：$$\sigma_1^2 \geq \cdots \geq \sigma_n^2$$，以及相对应的orthonormal eigenvectors $$v_1, \cdots, v_n$$。这些非负的$$\sigma_i$$就叫做singular values。

因为

$$ker(A^TA) = ker(A)$$

和

$$range(A^TA) = range(A^T)$$

我们有$$span(v_{i_1},\cdots, v_{i_p}) = range(A^T)$$以及$$span(v_{i_{p+1}},\cdots,v_{i_{n}}) = ker(A)$$，这里$$i_1,\cdots,i_n$$是$$1,\cdots,n$$的一个排列，而且假设$$\sigma_{i_1} \geq \sigma_{i_2} \geq \cdots \geq \sigma_{i_p}$$。因为$$v_1,\cdots,v_n$$构成$$R^{n}$$空间的一组基，所以上述结果一定成立。从而，我们定义

$$u_k = \frac{1}{\sigma_k}Av_{i_k}$$

其中$$k=1,\cdots,p$$。也就是说，$$Av_{i_k} = \sigma_k u_k$$。因为$$v_{i_1},\cdots,v_{i_p}$$并不在$$ker(A)$$内，所以上述$$u_k$$都不是0。而且这些$$u_k \in R^{m}$$还是orthonormal的：

$$\left\langle u_i,u_j \right\rangle = \frac{1}{\sigma_i \sigma_j} \left\langle Avi, Avj \right\rangle = \frac{1}{\sigma_i \sigma_j} \left\langle v_i, A^TAv_j \right\rangle = \delta_{ij}$$

将集合$$(u_1, \cdots, u_p)$$拓展到$$R^m$$的一个基$$(u_1, \cdots, u_m)$$。因为$$Av_i = \sigma_i u_i$$，所以我们有：

$$
A(v_1, \cdots, v_n) = (u_1, \cdots, u_m)
\begin{pmatrix}
\sigma_{i_1} & 0 & 0 & \cdots & 0 \\
0 & \sigma_{i_2} & 0 & \cdots & 0 \\
0 & \cdots & \sigma_{i_p} & \cdots & 0 \\
\cdots & \cdots & \cdots & \cdots & 0 \\
0 & \cdots & \cdots & \cdots & 0
\end{pmatrix}
$$

也就是$$A\widetilde V = \widetilde U \widetilde \Sigma$$，因此

$$A = \widetilde U \widetilde \Sigma \widetilde V^T$$

现在将$$\widetilde U$$对应乘以0 singular values的列和$$\widetilde V^T$$对应的乘以0 singular values的行去掉，我们就得到了最终的$$A = U \Sigma V^T$$，其中$$U \in R^{m \times p}$$并且$$V \in R^{n \times p}$$。


**Another Proof:**

矩阵$$A^TA$$和$$AA^T$$都是positive semidefinite的。因此，它们的eigenvalues都是非负的。

如果$$\lambda_i$$是矩阵$$A^TA$$的一个非零的eigenvalue，其对应的eigenvector是$$v_i$$，那么我们可以有$$A^TAv_i = \sigma_i^2 v_i$$，其中$$\sigma_i = \sqrt{\lambda_i}$$。

将上面的式子两边左乘$$v_i^T$$得到：

$$v_i^T A^T A v_i = \sigma_i^2 v_i^T v_i$$

因此

$$v_i^TA^TAv_i = (Av_i)^T(Av_i) = ||Av_i||^2 = \sigma_i^2v_i^Tv_i = \sigma_i^2$$

也就是说$$\lVert Av_i \rVert = \sigma_i$$。

再次回到$$A^TAv_i = \sigma_i^2v_i$$，左乘$$A$$我们得到：

$$AA^TAv_i = \sigma_i^2Av_i$$

这个式子表明$$Av_i$$是矩阵$$AA^T$$的eigenvector，对应的eigenvalue是$$\sigma_i^2$$。

结合上面两点，我们知道$$Av_i / \sigma_i$$是矩阵$$AA^T$$的一个单位长度的eigenvector，而且$$Av_i = \sigma_i u_i$$。

将上述所有的这些不为零的特征值都做上面的操作，并结合起来，就可以得到矩阵$$A$$的SVD分解了。


**2.3 SVD的几何意义**

**2.3.1 定义**

eigenvalue和singular values都被用来描述矩阵作用于某些向量的标量，都是描述向量模长变化幅度的数值。它们的差异在于：

* eigen vector描述的是矩阵方向不变作用（invariant action）的向量。
* singular vector描述的是矩阵最大作用（maximum action）的向量。

这里的作用（action）所指的是矩阵与向量的乘积得到的一个新的向量，几何上相当于对向量进行了旋转和拉伸，就像是对向量施加了一个作用（action），或者叫变换。

如果有向量$$v$$能使得矩阵$$A$$与之的积$$Av=\lambda v$$，$$\lambda$$是标量，那么$$\lambda$$和$$v$$就分别是$$A$$的特征值和特征向量。

如果存在单位正交矩阵$$U$$和$$V$$，使得$$A = U \Sigma V^T$$，其中$$\Sigma$$为对角矩阵，对角线上的值被称为奇异值，$$U$$和$$V$$的列分别被称为$$A$$的左奇异向量和右奇异向量。

**2.3.2 特征值和特征向量**

特征向量给出了方向不变作用的方向。当矩阵$$A$$作用于特征向量时，它只是将特征向量$$x$$乘了个标量$$\lambda$$，缩放了$$x$$，并没有改变它的方向。

矩阵

$$
\begin{pmatrix}
0 & 2 \\
2 & 0 \\
\end{pmatrix}
$$

其特征值为-2和2，对应的特征向量分别为$$\left[1, 1\right]$$和$$\left[-1. 1\right]$$。如下图所示。

![1]({{ '/assets/images/SVD-1.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}


对于矩阵

$$
\begin{pmatrix}
1 & \frac{1}{3} \\
\frac{4}{3} & 1 \\
\end{pmatrix}
$$

单位圆被变换为椭圆，特征向量是所有向量中经过变形变换后（乘以$$A$$后）方向不变的向量。如下图所示。

![2]({{ '/assets/images/SVD-2.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

**2.3.3 奇异值与奇异向量**

特征向量不变的方向并不保证是拉伸效果最大的方向，而这是奇异值的方向。奇异值是非负实数，通常从大到小排列。

$$\sigma_1$$是矩阵$$A$$的最大奇异值，对应右奇异向量$$v$$。$$v$$是

$$argmax_{||x||=1}(||Ax||)$$

的解，换句话说，在$$A$$与所有单位向量$$x$$的乘积中，$$Av = \sigma_1$$是最大的；但不能保证乘积之后方向不变。在下图的变化中，单位圆变成椭圆，奇异向量对应椭圆的长半轴，不同于上例中同意矩阵的特征向量的方向。

![3]({{ '/assets/images/SVD-3.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

对于最大的特征值来说，其是

$$argmax_{||x||=1} \left\langle x, Ax \right\rangle$$

的解，因为引入了内积，所以引入了方向。

方向不变和拉伸最大都是矩阵的内禀的性质，方向不变在马尔科夫随机场中很重要；而拉伸最大方向则是数据方差分布最大的方向，所以所含的信息量最大，是PCA等方法中的核心思想。


**参考文献**

* https://zhuanlan.zhihu.com/p/30658304
* https://zhuanlan.zhihu.com/p/353637184


## 3. Support Vector Machine（SVM）原理


https://www.stat.cmu.edu/~larry/=sml/SVM.pdf
https://zhuanlan.zhihu.com/p/77750026
https://zhuanlan.zhihu.com/p/31886934
https://blog.csdn.net/u012990623/article/details/40272619
https://zhuanlan.zhihu.com/p/31886934#:~:text=%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA%EF%BC%88support%20vector,%E7%9A%84%E9%9D%9E%E7%BA%BF%E6%80%A7%E5%88%86%E7%B1%BB%E5%99%A8%E3%80%82



## 4. t-student Stochastic Neighborhood Embedding（t-SNE）原理

t-SNE（t-distributed stochastic neighbor embedding）是用于降维的一种算法，是由Laurens van der Maaten和Geoffrey Hinton在2008年的时候提出的。此外，t-SNE还是一种非线性的降维方法，非常适合将高维数据降维到2维或者3维从而用于可视化。

t-SNE是由SNE（也是Hinton提出的）发展而来的。所以先介绍SNE，再拓展到t-SNE。


**4.1 SNE**

**4.1.1. SNE的基本原理**

SNE是通过仿射变换（affinite）将数据点映射到概率分布上，主要包括两个步骤：
* SNE构建一个高维数据之间的关系的的概率分布，使得相似的对象有更高的概率被选择，而不相似的对象被选择的概率较低。
* SNE在低维空间里构建这些数据点的概率分布，

SNE模型是非监督的降维，它和K-means等算法不一样，它不能通过在一个数据集上训练好了之后再用在其他数据上（K-means就可以在一个数据集上训练得到了K个中心点，应用在其他数据上），SNE算法只能单独对某些数据进行操作，而不能转移到其他数据上。


**4.1.2. SNE原理推导**

SNE将数据点之间的欧几里得距离转换为条件概率来表达点与点之间的相似度。具体来说，给定一个数据集，包含$$N$$个数据点$$\lbrace x_1, x_2, \cdots, x_N \rbrace$$，SNE首先是计算概率$$p_{ij}$$，其正比于$$x_i$$和$$x_j$$之间的相似度，即：

$$p_{j \mid i} = \frac{\exp (-\mid \mid x_i - x_j \mid \mid ^2 / (2 \sigma^2_i))}{\sum_{k \neq i} \exp(-\mid \mid x_i - x_k \mid \mid ^2 / (2 \sigma^2_i))}$$

这里有一个参数是$$\sigma_i$$，对于不同的数据点$$x_i$$的取值不一样，后续再讨论。

对于降维之后低维的数据，记为$$\lbrace y_1, y_2, \cdots, y_N \rbrace$$，其可以定义类似上述的条件概率，这个时候我们指定每个数据点$$y_i$$对应的$$\sigma=\frac{1}{\sqrt{2}}$$：

$$q_{j \mid i} = \frac{\exp (-\mid \mid y_i - y_j \mid \mid ^2)}{\sum_{k \neq i} \exp(-\mid \mid y_i - y_k \mid \mid ^2)}$$

可以看到，SNE通过仿射变化将数据点映射到概率分布上，将两个数据点之间的欧式距离转换为以一个点为中心一定范围（方差）内另一个点出现的条件概率。而如果降维的效果较好，也就是说数据点之间的局部特征保留完整，那么$$p_{j \mid i} = q_{j \mid i}$$，也就是低维数据和高维数据两个条件概率的差异很小。我们一般用KL散度来定义概率分布之间的差异，从而目标函数（cost function）为：

$$C = \sum_i KL(P_i \mid \mid Q_i) = \sum_i \sum_j p_{j \mid i} \log \frac{p_{j \mid i}}{q_{j \mid i}}$$

其中$$P_i$$表示的是对于数据点$$i$$，所有其他的点构成的概率分布，这里$$P_i$$是一个概率分布，对于每个其它的点$$j$$，其概率值为之前定义的$$p_{j \mid i}$$。而我们要做的就是使得高维数据的每个数据点的概率分布$$P_i$$和低维数据的每个数据点的概率分布$$Q_i$$尽可能的相似。需要注意的是KL散度具有不对称的特性，用低维里距离较远的两个点来表达高维里距离较近的两个点会产生较大的cost，而用低维里距离较近的两个点来表达高维里距离较远的两个点的cost会小一点，如果$$q_{j \mid i} = 0.8$$，$$p_{j \mid i} = 0.2$$，那么$$p \log \frac{p}{q} = 1.11$$，而如果$$q_{j \mid i} = 0.2$$，$$p_{j \mid i} = 0.8$$，那么$$p \log \frac{p}{q} = 0.27$$。所以说，SNE会倾向于保留数据里的局部特征，也就是说高维数据里相近的点会在低维数据里得到较好的保留。


>下面要解决的问题就是对于每个数据点，如何选择$$\sigma$$


不同的点具有不同的$$\sigma$$，$$P_i$$的熵（entropy）会随着$$\sigma_i$$的增加而增加。SNE使用困惑度（perplexity）的概念，使用二分搜索的方式来找到一个最佳的$$\sigma_i$$。困惑度的定义为：$$Perp(P_i) = 2^{H(P_i)}$$，这里$$H(P_i)$$是$$P_i$$的熵，即$$H(P_i) = -\sum_j p_{j \mid i} \log p_{j \mid i}$$。困惑度可以解释为一个点附近的有效邻近点个数。SNE对困惑度的调整比较具有鲁棒性，通常选择5-50之间，给定之后，使用二分搜索的方式找到合适的$$\sigma_i$$。

每个数据点有了$$\sigma_i$$之后，$$p_{j \mid i}$$就可以计算了，从而$$P_i$$也可以算出来，是一个固定的值。我们的目标函数里的变量为$$y_i$$，从而其就等价于优化$$\sum_i \sum_j -p_{j \mid i} \log q_{j \mid i}$$，这个式子和softmax非常的类似。softmax的目标函数是$$\sum_i y_i \log p_i$$，其中$$y_i$$为ground truth标签，$$p_i$$是预测的标签，都是一个向量。计算目标函数对于$$y_i$$的梯度，得到：

$$\frac{\partial C}{\partial y_i} = 2 \sum_j (p_{j \mid i} + p_{i \mid j} - q_{j \mid i} - p_{i \mid j} q_{i \mid j})(y_i - y_j)$$

但优化上述目标函数比较困难，一开始可以使用较小的$$\sigma$$来进行初始化。而且可以引入momentum来更新$$y_i$$，即每次更新不仅和这次的梯度有关，还和之前的值有关。还可以在优化的初始阶段引入高斯噪声，再类似于模拟退火的方式逐渐减小该噪声避免陷入局部最优解。

以上就是SNE的过程。


**4.2. t-SNE**

尽管SNE提供了很好的可视化方法，但是它很难优化，而且存在crowding problem。从而，Hinton又提出了t-SNE方法。其与SNE的主要区别如下：

* 解决了KL散度衡量时的不对称的问题，简化了梯度公式
* 在低维空间下，使用t分布，而不是SNE里的高斯分布来表达两点之间的相似度

t-SNE在低维空间下使用t分布，是因为t分布比高斯分布具有更长的尾部，其可以避免crowding问题，以及使得优化更加容易。下面先介绍SNE如何实现的对称化，再介绍crowding问题，最后再介绍t-SNE算法。


**4.2.1. Symmetric SNE**

替换掉SNE里$$p_{j \mid i}$$和$$q_{j \mid i}$$之间不对称的KL散度的一个方法是，使用联合概率分布来替换条件概率分布，也就是说$$P_i$$代表的是高维空间里每两个点（包括点$$i$$）出现的概率，$$Q_i$$同理。从而目标函数为：

$$C = KL(P \mid \mid Q) = \sum_i \sum_j p_{i,j} \log \frac{p_{i,j}}{q_{i,j}}$$

将这种SNE成为symmetric SNE，因为$$p_{i,j} = p_{j,i}$$，以及$$q_{i,j} = q_{j,i}$$。这种情况下，概率分布$$p$$和$$q$$就改写为：

$$p_{i,j} = \frac{\exp(- \mid \mid x_i - x_j \mid \mid ^2 / (2 \sigma^2))}{\sum_{k \neq l} \exp(- \mid \mid x_k - x_l \mid \mid ^2 / (2 \sigma^2))}$$

$$q_{i,j} = \frac{\exp(- \mid \mid y_i - y_j \mid \mid ^2 / (2 \sigma^2))}{\sum_{k \neq l} \exp(- \mid \mid y_k - y_l \mid \mid ^2 / (2 \sigma^2))}$$

这种表达方式，使得整体简洁了很多。但是会引入异常值的问题。比如说$$x_i$$是异常值，那么$$\mid \mid x_i - x_j \mid \mid^2$$就会很大，从而对应所有的$$p_{i,j}$$都会很小，从而导致$$y_i$$对目标函数的影响较小。

为了解决这个问题，将上述联合分布的定义修改为：

$$p_{i,j} = \frac{p_{i \mid j} + p_{j \mid i}}{2}$$

$$q_{i,j} = \frac{q_{i \mid j} + q_{j \mid i}}{2}$$

而$$p_{i \mid j}$$和$$q_{j \mid i}$$的定义如SNE里一样。这样每个点（即使是异常点）也会对目标函数做出合理的贡献。这样定义的SNE的好处是梯度计算变得简单了：

$$\frac{\partial C}{\partial y_i} = 4 \sum_{j} (p_{i,j} - q_{i,j})(y_i - y_j)$$

实验证明，对称SNE能够产生和SNE一样好的结果，甚至有时候更好一点。


**4.2.2. Crowding问题**

拥挤问题就是说各个簇聚集在一起，无法区别开。假设以一个数据点为$$x_i$$为中心，半径为$$r$$的$$m$$维球，其内部数据是均匀分布的，但是随着$$m$$的增长，各个点几乎都分布在这个球的边缘附近，所以尽管点是均匀分布的，其与圆心的距离并不是均匀的，而是集中在了一个值附近，这样的高维数据映射到低维上，会导致点都聚集在一起。

**4.2.3. t-SNE**

t-SNE解决crowding的方法是在低位数据上的概率分布，不再使用高斯分布，而是使用尾部更长的t-分布。SNE实际上是将数据之间的距离用概率分布来表示了，而SNE所作的优化就是将高维数据对应的概率分布和低维数据对应的概率分布足够的相近。在低维空间下，使用更加偏重长尾分布的概率分布，可以使得高维数据里距离远的数据点在映射后依然能有一个不那么远的距离。

而且t-分布受到异常值的影响更小。

使用了t-分布之后的$$q$$定义为：

$$q_{i,j} = \frac{(1 + \mid \mid y_i - y_j \mid \mid ^2)^{-1}}{\sum_{k \neq l}(1 + \mid \mid y_k - y_l \mid \mid^2)^{-1}}$$

由下图可以看到，对于较大相似度的点，t-分布在低维空间里的距离需要稍小一点，而对于低相似度的点，t-分布在低位空间里的距离要求更远一点。这满足我们的要求，也就是同一簇内的店（距离较近）聚合的更加近，不同簇之间的点距离更远。


![t1]({{ '/assets/images/TSNE-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}


下图是t-SNE优化好了的一个结果。


![t2]({{ '/assets/images/TSNE-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

**4.2.4**不足

* t-SNE主要用于可视化，很难用于其他目的。因为其计算都是针对某个数据集，并不是学习到了某些参数，所以不能训练一个模型再用在其他的数据上。
* t-SNE倾向于保存局部特征，对于本征维度就很高的数据来说，不可能完整的不损失信息的压缩到2维或者3维。
* t-SNE没有唯一的最优解，而且t-SNE里的距离本身没有意义，表示的仅仅是概率分布。
* 训练过程太慢。


**4.3. 参考文献**

* http://www.datakit.cn/blog/2017/02/05/t_sne_full.html


## 5. Principal Component Analysis（PCA）原理分析

PCA（principal component analysis）是一种常见的数据分析方式，常用于高维数据的降维，可以用于提取数据的主要特征。

PCA的数学推导可以从最大可分性和最近重构性两方面进行，前者的优化条件为划分后方差最大，后者的优化条件为点到划分平面的距离最小。下面从最大可分性角度进行分析。

**5.1. 线性代数的一些基础知识**

两个向量$$A$$和$$B$$的内积等于$$\lvert A \rvert \lvert B \rvert cos \alpha$$，如果$$B$$为单位向量，则内积可以理解为向量$$A$$向向量$$B$$所在直线投影的标量大小。

在向量空间描述向量，需要先定义一组基，然后给出在这些基所在的各个直线上的投影，当基的模长为1时，也就是这个向量在这组基下的坐标值（向量和某个模长为1的基的内积，就是向量在这个基所在直线上的投影，也就是在这个基之下的坐标）。我们需要基是线性无关的，正交不是必要条件，但是正交基操作起来比较方便。

假设$$p_i$$是第$$i$$个基，一共$$R$$个，按行排列，记为矩阵$$P$$，$$a_j$$是这个向量空间里的向量，一共有$$M$$个，按列排列，记为矩阵$$A$$，那么$$PA$$就表示这些向量在以$$P$$为基之下的坐标值：

$$
\begin{pmatrix}
p_1 \\
p_2 \\
\vdots \\
p_R
\end{pmatrix}

\begin{pmatrix}
a_1 & a_2 & \cdots & a_M
\end{pmatrix}

=
\begin{pmatrix}
p_1 a_1 & p_2 a_2 & \cdots & p_1 a_M \\
p_2 a_1 & p_2 a_2 & \cdots & p_2 a_M \\
\vdots & \vdots & \cdots & \vdots \\
p_R a_1 & p_R a_2 & \cdots & p_R a_M
\end{pmatrix}

$$

所以说，上述操作也给矩阵乘法找了一个物理解释：两个矩阵相乘的意义就是，将右侧矩阵的每一个列向量变换到以左侧矩阵每一行为基的坐标系里去（前提是左边向量每个行向量模长为1，而且之间线性无关，且左侧矩阵的行向量和右侧矩阵的列向量的值是由同一组模长为1的基计算出来的）。


**5.2. 最大可分性**

使用不同的基可以让相同的向量数据给出不同的坐标值，而如果基的数量少于向量所在空间的维度，则可以达到对原向量数据降维的目的。

但关键问题在于：如何选择最优的基？具体来说，如果有一组$$N$$维向量组成的数据，要将其降到$$K$$维（$$K < N$$），那么我们该如何选择这$$K$$个基才能最大程度的保留原有的信息？

一种直观的看法是，希望这组向量在这组基之下的投影尽可能的分散（符合信息论的观点，熵越大信息越多，而且直观来看如果有重叠，那有部分数据就被盖住了）。

**5.2.1. 方差、协方差和协方差矩阵**

数据的分散程度可以用方差来表示，一个一维变量的方差定义为：$$Var(a) = \frac{1}{m} \sum_{i=1}^m (a_i - \mu)^2$$，其中$$\mu$$是这个变量的均值。而$$n$$维向量变量的方差定义为：$$Var(a) = \frac{1}{m} \sum_{i=1}^m (a_i - \mu) (a_i - \mu)^T$$，结果就是一个矩阵了，这个矩阵叫做variance-covariance matrix。也就是下面所说的协方差矩阵，矩阵里每一项表示的是这个向量变量每两个element之间的协方差（针对一维变量）。

>注意，$$Var(a) = \frac{1}{m} \sum_{i=1}^m (a_i - \mu)^T (a_i - \mu)$$的结果是一个标量，但是这个一般不被使用。

对于两个一维变量$$a$$和$$b$$，我们可以用协方差表示这两个变量的相关性。两个变量之间的线性相关性越强，则越表明这两个变量所含的相同的信息越多，也就是存在重复的信息。对于两个一维变量，协方差定义为：$$Cov(a, b) = \frac{1}{m} \sum_{i=1}^m (a_i - \mu_1)(b_i - \mu_2)$$，其中$$\mu_1$$和$$\mu_2$$分别表示$$a$$和$$b$$的均值。而对于相同维度的高维变量$$a$$和$$b$$来说，协方差定义为：$$Cov(a, b) = \frac{1}{m} \sum_{i=1}^m (a_i - \mu_1) (b_i - \mu_2)^T$$，是一个矩阵，上面定义的高维变量的方差是这个的特例，这个矩阵里的每一项表示的是这两个向量里分别取一个element之间的协方差（针对一维变量）。这里用不到这个定义。

我们的数据集有$$m$$个$$N$$维数据，数据的每个维度都可以理解为一个独立的一维变量，也就是每个变量都是一维变量，每个变量的采样值为$$m$$，而一共有$$N$$个这样的变量。

我们将每个维度的变量都转换为均值为0（也就是进行平移），那么变量之间的协方差就变成了这两个变量的内积，而每个变量本身的方差就变成了这个变量模长的平方。我们要做的就是使得协方差尽可能的小，而方差尽可能的大。

从而我们的优化问题变为：**将一组$$m$$个$$N$$维向量数据降为$$K$$维，其目标是选择$$K$$个单位正交基，使得原始数据变换到这组基上后（也就是在这组基上的坐标值），各个变量之间的协方差为0，而变量的方差的和尽可能的大（也就是在正交的基的约束下，选择使得变换后的变量方差最大的K个基）。


>这两个链接给了协方差矩阵的一些解释和定义：
>https://math.stackexchange.com/questions/1742193/covariance-of-two-random-vectors
>https://math.stackexchange.com/questions/99560/generalization-of-variance-to-random-vectors


**5.2.2. 具体计算**

针对上述的优化目标，以下进行数学推导。

最终要达到的目标和变量内方差以及变量间协方差有密切关系，而这些变量是将原始$$m$$个$$N$$维向量数据每一维看作一个变量而得来的，也就是这些变量都是一维变量，而每个变量的采样值为$$m$$，而一共有$$N$$个变量。正如之前所说，如果每个变量的均值为0，那么方差和协方差均可以用内积来表示（注意此时做内积的向量，则是将某个维度的所有$$m$$个数据点构成的长度为$$m$$的向量拿来做内积）。

假设我们现在只有两个变量$$a$$和$$b$$（也就是说数据的维度$$N=2$$），将其按行组成矩阵$$X$$：

$$X = 
\begin{pmatrix}
a_1 & a_2 & \cdots & a_m \\
b_1 & b_2 & \cdots & b_m
\end{pmatrix}
$$

从而：

$$\frac{1}{m} X X^T = 
\begin{pmatrix}
\frac{1}{m} \sum_{i=1}^m a_i^2 & \frac{1}{m} \sum_{i=1}^m a_i b_i \\
\frac{1}{m} \sum_{i=1}^m a_i b_i & \frac{1}{m} \sum_{i=1}^m b_i^2
\end{pmatrix}

=

\begin{pmatrix}
Cov(a,a) & Cov(a,b) \\
Cov(a,b) & Cov(b,b)
\end{pmatrix}
$$

可以看到，所得到的矩阵，对角线上的是变量内的方差，其它地方是变量间的协方差。两者被统一到了一个矩阵里。

同理，假设有$$m$$个维度为$$n$$的向量数据，将其按行排列为矩阵$$X_{n,m}$$，设$$C = \frac{1}{m} X X^T$$，则$$C$$是一个对称矩阵，其对角线上的值分别对应各个变量的方差，而第$$i$$行$$j$$列对应的是第$$i$$个和第$$j$$个变量间的协方差。


根据我们的优化条件，我们需要将除了对角线以外的其他元素都化为0，而对角线上的元素需要按从大到小的顺序排列（变量内方差尽可能大），这样就达到了优化目的。

我们先来看一下原矩阵与基变换后矩阵的协方差矩阵之间的关系。

设原矩阵$$X \in R^{N \times m}$$对应的协方差矩阵为$$C$$，也就是$$C = \frac{1}{m} X X^T \in R^{N \times N}$$，而$$P \in R^{N \times N}$$是一组归一化的正交基按行组成的矩阵，设$$Y = PX$$，而$$Y$$对应的协方差矩阵为$$D$$，而$$D$$和$$C$$的关系为：

$$D = \frac{1}{m} Y Y^T = \frac{1}{m} PX X^T P^T = PCP^T$$

这样我们就知道，我们要找的$$P$$是能让原始协方差矩阵$$C$$对角化的$$P$$。也就是说，优化目标变为，寻找一个矩阵$$P$$，其中$$P$$的行向量是一组归一化了的正交基，满足PCP^T是一个对角矩阵，并且对角元素按从大到小依序排列，那么$$P$$的前$$K$$行就是要寻找的基，用$$P$$的前$$K$$行乘以$$X$$就将原数据$$X$$从$$N$$维降到了$$K$$维，并且满足之前所说的优化条件。

从而，我们现在要做的就是将矩阵$$C$$对角化。而协方差矩阵$$C$$是一个对称矩阵，对于实对称矩阵，其有一系列很好的性质：

* 实对称矩阵的不同特征值对应的特征向量正交
* 实对称矩阵的代数重数等于几何重数，也就是说特征向量的重根个数等于对应的特征向量空间的维数。从而大小为$$n \times n$$的矩阵，就有$$n$$个线性无关的特征向量

> 上述性质可以参考SVD分解里的内容。

从而，一个$$N \times N$$的实对称矩阵$$C$$，一定可以找到$$N$$个单位正交特征向量$$e_1, \cdots e_N$$，将其按列组成矩阵$$E = (e_1, \cdots, e_N)$$。这也就是eigendecomposition，从而：

$$E^T C E = \Lambda = 

\begin{pmatrix}
\lambda_1 & & & \\
 & \lambda_2 & & \\
 & & \cdots & \\
 & & & \lambda_n
\end{pmatrix}
$$

$$\Lambda$$为对角矩阵，对角线上的值就是矩阵特征值（可能有重复）。从而我们发现，$$P = E^T$$满足条件。也就是说$$P$$是原始数据$$X \in R^{N \times m}$$的协方差矩阵$$C = \frac{1}{m} X X^T \in R^{N \times N}$$的特征向量单位化之后按行排出的矩阵，其中每一行都是$$C$$的一个特征向量。如果设$$P$$按照$$\Lambda$$中特征值从大到小的顺序，将特征向量从上到下排列，则用$$P$$的前$$K$$行组成的矩阵乘以原始的数据矩阵$$X$$所得到的结果$$Y$$就是我们所需要的降维数据。


**5.3 补充**

* (1) 拉格朗日乘子法

在我们上述求解的过程中，我们希望变化后的变量具有的性质为：变量间协方差为零，变量内方差和尽可能的大。我们使用的是实对称矩阵的性质完成的。实际上还可以通过将其转化为最优化问题利用拉格朗日乘子法来予以推导。

对于归一化的正交基来说，向量在每个基下面的坐标就是这个向量和这个基的内积。所以说，对于这组正交基里的某一个基$$\omega$$来说，其和原始数据的每个向量做内积，对应的就是在这个正交基的框架下，每个原始数据对应的新的数据在这个基$$\omega$$下的坐标值（它们整体组成了新的一个变量，也就是$$Y$$里的某一行）。从而，我们计算这个基作用在原始数据集上找到的新的数据集的坐标值构成的新的这个变量的方差为：

$$D(x) = \frac{1}{m} \sum_{i=1}^m (\langle x_i, \omega \rangle)^2 = \frac{1}{m} \sum_{i=1}^m (x_i^T \omega)^2 = \frac{1}{m} \sum_{i=1}^m (x_i^T \omega)^T (x_i^T \omega) $$


$$= \frac{1}{m} \sum_{i=1}^m \omega^T x_i x_i^T \omega = \omega^T (\frac{1}{m} \sum_{i=1}^m x_i x_i^T) \omega $$。


可以看到，$$\frac{1}{m} x_i x_i^T$$就是原本数据的协方差，记这个为$$C$$，从而我们的目标为：

$$ max \lbrace \omega^T C \omega \rbrace$$

$$s.t., \omega^T \omega = 1$$

构造拉格朗日函数：$$L(\omega) = \omega^T C \omega + \lambda(1 - \omega^T \omega)$$

对$$\omega$$求导，得到$$C \omega = \lambda \omega$$，此时的方差$$D(x) = \omega^T C \omega = \lambda$$。

由上述式子看出，解$$\omega$$就是矩阵的协方差矩阵$$C$$对应的一个特征向量，而$$\lambda$$就是对应的特征值。我们要找最大方差，也就是要找协方差矩阵$$C$$的最大的特征值，而最佳的投影方向，也就是该特征值对应的特征向量的方向。

上述再继续下去，就可以找到$$K$$个满足要求的基，这些基因为都是实对称矩阵$$C$$的特征向量，所以可以让它们是归一化的正交基。

从而我们结束了基于最大可分性的PCA的推导。


* (2) 最近重构性原则

上述推导过程是基于最大可分性的，也就是样本点投影到这些基上的方差的和最大。除此方法外，还可以将其转换为线性回归问题，其目标是求解一个线性函数，使得对应直线能够更好的拟合样本点集合，从而将优化目标从方差最大转换为平方误差最小，因为映射距离越短，丢失的信息也就越小。这种方法也就是最近重构性原则。


**5.4. PCA求解步骤**

PCA的算法步骤如下：

假设有$$m$$个维度为$$n$$的数据。

* 将原始数据按列组成$$n$$行$$m$$列矩阵$$X$$
* 将$$X$$的每一行进行零均值化，也就是减去这一行的均值
* 求出协方差矩阵$$C = \frac{1}{m} X X^T$$
* 求出协方差矩阵的特征值和对应的特征向量
* 将特征向量按照特征值从大到小的顺序按行排列成矩阵，取前$$k$$行组成矩阵$$P$$
* $$Y=PX$$即为降维到$$k$$维后的数据


**5.5. 性质**

* （1）缓解维度灾难。PCA算法通过舍去一部分信息之后能够使得样本的采样密度增大（因为维数降低了），这是缓解维度灾难的重要手段
* （2）降噪。当数据受到噪声影响时，最小特征值对应的特征向量往往和噪声有关，将它们舍弃可以一定程度上减少噪声的影响
* （3）过拟合。PCA保留了主要信息，但这个信息只是针对训练集的，而且这个主要信息未必是重要信息，有可能舍弃了一些看似无用的信息但实际上却是这个数据集本质的信息（只是其在训练集上没有很好地表现出来），从而加重了过拟合
* （4）特征独立。PCA不仅将数据降维了，降维后的数据每个维度的特征也相互独立了（协方差为0）


**5.6. 后记**

* （1）零均值化

当对训练集进行PCA降维时，也需要对验证集、测试集执行相同的降维。而对验证集、测试集执行零均值化操作的时候，均值是由训练集计算而来的，不能使用验证集或者测试集的中间向量（也就是均值）。

其原因是，因为我们的训练集属于可以观察到的数据，而测试集是观察不到的，当然不知道它的均值，而验证集一般都是和训练集放在一起处理，处理完之后再抽取出来的，一般不会单独处理。如果单独处理了，其地位也和测试集一样，所以原因也是一样的。

另外，我们认为训练集、测试集、验证集是独立同分布的，所以这样做也是合理的。


* （2）与SVD进行对比

这是两个完全不同的概念，而且特征值和特征向量是方阵才有的，而SVD对于任意形状的矩阵都可以进行计算。

而实际上，PCA就是方阵（原数据矩阵对应的协方差矩阵）的特征值分解，也就是$$C = Q \Lambda Q^{T}$$。其中$$Q$$就是这个矩阵$$C$$的特征向量组成的矩阵，$$\Lambda$$是一个对角矩阵，对角线元素就是特征值，这些特征值所对应的特征向量就是描述这个矩阵变化方向用的。也就是说矩阵$$A$$的信息可以由其特征值和特征向量表示。

而SVD是对于矩阵$$A$$的协方差矩阵$$AA^T$$以及$$AA^T$$做特征值分解而推导出来的（这两种情况分别是将矩阵$$A$$每一列看成一个数据或者每一行看成一个数据）。矩阵$$A$$的SVD分解为：$$A_{m,n} = U_{m,m} \Lambda_{m,n} V_{n,n}^T \approx U_{m,k} \Lambda_{k,k} V_{n,k}^T$$，下标表示维度。其中$$U,V$$都是正交矩阵，且$$U^T U = I_m, V^T V = I_n$$。上述的约等于是将$$\Lambda$$里的$$n$$个奇异值的较小的那部分省略，只取$$k$$个最大的而得到的，这是人为可以控制的，比如说去掉那些和0非常接近的奇异值。

$$A^TA = (U \Lambda V^T)^T(U \Lambda V^T) = V \Lambda^2 V^T$$

$$AA^T = (U \Lambda V^T)(U \Lambda V^T)^T = U \Lambda^2 U^T$$

所以说，因为$$A^TA$$和$$AA^T$$都是实对称矩阵，根据eigendecomposition，$$V,U$$分别是$$A^T A$$以及$$A A^T$$的特征向量构成的矩阵，而且$$\Lambda^2$$的对角线元素就是这两个矩阵的特征值（$$A^TA$$和$$AA^T$$的特征值是一样的）。我们也能看出来，$$A$$的奇异值就是$$A^TA$$的特征值的非负平方根。

PCA需要对协方差矩阵$$C = \frac{1}{m} X X^T$$进行特征值分解，而SVD也是对矩阵$$A^TA$$进行特征值分解，如果取$$A = \frac{X^T}{\sqrt{m}}$$，那么两者等价。所以PCA问题可以换成SVD问题进行求解。

实际上Sklearn里的PCA就是用SVD进行求解的，因为：

* 当样本维度很高的时候，协方差矩阵计算起来太慢
* 方阵特征值计算的效率不高
* SVD除了特征值分解这组求解方式以外，还有更高效的迭代求解方式，避免了$$A^TA$$的计算
* 实际上PCA和SVD的右奇异向量的压缩效果相同。


**5.7. 参考文献**

* https://zhuanlan.zhihu.com/p/77151308
* https://zhuanlan.zhihu.com/p/32412043
* https://zhuanlan.zhihu.com/p/37777074
* https://www.zhihu.com/question/41120789



## 6. 目标检测任务里的mAP（mean average precision）的计算方法

目标检测由于其输出结果比图像分类更加复杂，所以评估起来也更加复杂。总体来说，最主要的一个评估指标是mAP。此外，运行效率也是目标检测算法一个重要的评估指标。不过介绍mAP之前，先通过一些例子，介绍一些相关的机器学习概念，最后介绍mAP。有关计算的函数在这个[github post](https://github.com/neoxu314/2D-Computer-Vision-Metrics)可以找到。

**6.1 PR曲线（Precision-Recall Curve）**

用两个例子去说明说明PR曲线以及机器学习中一些相关的基础概念。

**6.1.1 猫狗图像分类**

假设我们现在有猫和狗的图像分别50张（一共100张测试图片），我们用这个测试集对一个已经训练好的猫狗图像分类模型进行测试。如果我们要对这次分类测试进行评估的话，很简单，只需要去用下面的公式计算准确率。打个比方，如果对于这100张图片，我们有97张图片分类正确，那么我们的图像分类的模型在这个测试集上的准确率就是97/100 = 0.97。

准确率 = （正确分类的图片数量）/（总的图片数量）

**6.1.2 肺癌检测**

对于猫狗图像的分类任务，准确率可以说是最简单并且直接的评估指标。但是在分类问题上，准确率并不总是最合适的评估指标。现在假设我们有另外一个用于对肺部扫描图像进行肺癌检测的算法模型。我们有一个100张肺部扫描图像的测试集用于对我们的肺癌检测模型进行测试。因为患肺癌的人在现实生活中其实占比相对来说是很低的，所以我们的测试集中有97张图像对应没有肺癌（医疗中称为阴性）和3张图像对应有肺癌（医疗中称为阳性）。对于这样的测试集（97张阴性和3张阳性图像）来说，如果我们想取得相对较高的准确率很容易，最简单的方法是对于任何输入图片都判定为阴性就好。这样子，通过上面准确率的公式，我们的模型可以轻松获得97%的准确率。可是这样的模型真的有用吗？答案显而易见，没有任何用处。因为那三个应该被检测出肺癌的不幸的人，因为模型错误将其判定为阴性，最后因为没有被及时治疗而死亡。。对于这种数据集中不同类别数据的数量相差巨大的分类问题，我们一般叫做不平衡数据的分类问题。那么我们怎么样才能评估我们的肺癌检测模型呢？就要引入一些新的概念。对于测试数据来说，基于我们的预测分类值以及数据原本的真实分类值，我们可以画一个表格如下所示。

![map1]({{ '/assets/images/MAP-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

如果预测的是阳性并且真实值也是阳性，那么我们就管这次预测叫做TP（真阳性）；如果预测的是阳性但是真实值是阴性，那么我们就管这次预测叫做FP（假阳性）；如果预测值是阴性而且真实值也是阴性，那么我们管这次预测叫做TN（真阴性）；如果预测值是阴性但是真实值是阳性，那么我们管这次预测叫做FN（假阴性）。

**6.1.3 精确率（Precision）和召回率（Recall）**

有了上面TP、FP、FN、TN的概念，我们可以把我们模型在测试集上的所有预测结果划分到这四个类别中，从而得到每个类别对应的预测结果的数量。我们用TP、FP、FN、TN来代表对应类别的预测结果数量，就可以通过它们算出精确率和召回率。在这个例子中，Positive（正样本）代表患癌症（阳性），Negative（负样本）代表没患癌症（阴性）。

精确率 = TP / (TP + FP) = TP / 所有被模型预测为正样本的数据的数量

召回率 = TP / (TP + FN) = TP / 所有真实类别为正样本的数据的数量

如果我们的算法预测所有的数据样本都是阴性的话，那么这样的话TP就是0，所以精确率和召回率也都是0。这样来看，即便是准确率有97%，但是精确率和召回率都是0。所以仅靠准确率是完全不能评估我们的肺癌识别算法的。

**6.1.4 PR曲线**

在我们的算法模型进行预测的时候，算法会对每个数据样本产生一个在$$\left[0,1\right]$$区间的置信度。如果这个置信度高于某个阈值，那么我么就可以将这个数据样本判断为正样本（这个例子就是肺癌阳性），反之则是负样本（这个例子中是肺癌阴性）。 比如说，在我们这个例子中，这个阈值可以是0.5，那么高于0.5的就是阳性结果，低于0.5的就是阴性结果。很显然，这个阈值的设定会直接影响到我们的预测结果，也当然会影响到模型的精确率以及召回率。对于我们的这个肺癌检测算法模型来说，精确率和召回率都是越高越好。但是在现实情况中，一般如果精确率高的话，召回率就会相对较低，反之亦然。 举个例子，如果我们希望当且仅当算法模型十分确定时，才将一个肺部扫描图像预测为阳性。因为我们不希望我们告诉一个病人他患有肺癌，但是事实却是没有。那么我们可以设定一个比较高的阈值，这样我们模型的精确率会上升很多，但是召回率就会下降。这样的话，会导致我们的模型无法正确识别出某些真正患有肺癌的图像。换句话说，会有一些不幸的人得了肺癌但是却没有被检测出来。如果我们希望让模型能更多的发现肺癌，也就是说我们希望我们能将足够多的患有肺癌的患者的扫描图像预测为阳性，那么我们可以降低这个阈值。这会提高模型的召回率，但是降低精确率。导致的结果是，会有一些健康的人被错误的通知为患有癌症。当然我们也不想有这种情况。所以我们需要做的是找到一种精确率与召回率之间的平衡。其中一个方法就是画出PR曲线，然后用PR曲线下方的面积AUC（Area under Curve）去判断模型的好坏。我们只要通过设置不同的阈值，产生不同的精确率和召回率，然后将这些数据绘制成一个折线图，就能得到模型的PR曲线。下图中展示了一个PR曲线例子。

![map2]({{ '/assets/images/MAP-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

当然在这种数据不平衡的情况下，判断模型的好坏还有其他的一些指标比如说F1值和ROC。在目标检测算法的评估中，最重要的是了解什么是PR曲线。


**6.2 交并比（IoU）**

在了解什么是PR曲线之后，我们可以看下如何评估目标检测算法了。我们知道，模型检测算法既要输出用来定位目标物体位置的边界框，又要输出每个检测出来的目标物体的类别标签。那么我们评估的时候就要考虑两方面，第一个是物体的边界框是否准确，第二个是预测的类别标签是否正确。首先第一个概念是交并比，英文是IoU（Intersection over Union）。这个意思就是，假设模型对于某个物体预测了一个边界框，然后我们又有这个物体的真实的边界框，我们就通过这两个边界框相交的区域去判断我们预测的边界框是不是准确。相交区域越大越准确，当然如果完全重合肯定是最准确的情况。所以我们可以通过这两个边界框相交的区域除以并起来的区域得到交并比IoU。下面这个图片比较形象地把这个公式表达出来了。

![map3]({{ '/assets/images/MAP-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}


**6.3 平均精度均值（mAP）**

这是目标检测算法的主要评估指标。

有了上面的PR曲线和交并比的基础概念，我们就就可以计算平均精度均值（mAP）了。平均精度均值（mAP）是精度均值（AP）的平均值。所以想要算平均精度均值（mAP），我们需要先算精度均值（AP）。简单来说，假设在测试集中一共有$$K$$个类别的目标物体。我们先算出我们的模型对于每个类别的物体的AP，然后将这些AP相加在一起再除以所有类别的数量$$K$$，就可以得到最终模型的mAP。前面在介绍PR曲线时，我们知道，通过调整不同置信度阈值我们可以得到不同的预测结果，从而得到不同的精确率和召回率。这里我们引入一个类似的概念叫做IoU阈值。对于图像中某个类别的物体，我们的模型会预测对应的边界框，但是这个预测的边界框对应IoU可能很小也可能很大。所以假设边界框对应的IoU大于某个阈值（一般来说，比较常用的IoU阈值是0.5），我们就可以收这个预测的边界框是对的，或者说可以被划分为TP中。反之如果IoU小于阈值，那么这个预测的边界框就是错的，或者说是一个FP。如果对于图像中某个物体来说，我们的模型没有预测出对应的边界框，那么这种情况就可以被记为一次FN。有了TP、FP和FN我们就可以计算出精确率和召回率。所以通过调整这个IoU的阈值，我们可以为我们的模型画出其对于每个物体类别的PR曲线。所以，对于目标检测测试数据集中的每个物体类别来说，我们都可以为我们的模型画出其对应的PR曲线。通过PR曲线，我可以可以得到PR曲线对应的AP值。

在2010年以前，PASCAL VOC竞赛中AP是这么定义的：对于一条AP曲线，我们把recall的值从0到1划分为11份：0、0.1、0.2、0.3、0.4、0.5、0.6、0.7、0.8、0.9、1.0。在每个recall区间（0-0.1, 0.1-0.2，0.2-0.3，…，0.9-1.0）上我们计算精确率的最大值，然后再计算这些精确率最大值的总和并平均，就是AP值。这种方法叫做11-poinst-interpolation。

从2010年之后，PASCAL VOC竞赛把这11份recall点换成了PR曲线中的所有recall数据点。这种方法叫做all-points-interpolation。这个AP值也就是PR曲线下的面积值。当我们可以计算出每个物体类别对应的模型AP后，我们只需要对于所有AP求和再平均，就可以得到用于评估目标检测模型的mAP了。


## 7. 深度神经网络里GlobalAveragePooling层和Flatten层的区别与联系

CNN模型在从卷积层转变为Dense层时，需要使用GlobalAveragePooling或者Flatten层将卷积层所输出的feature map（大小为$$W \times H \times C$$）转换为vector（大小为$$L \times C$$）。而我们介绍一下这两种方法之间的差别。

GlobalAveragePooling层作用在大小为$$H \times W \times C$$的feature map上，对于每个通道里的$$H \times W$$个值，计算其平均值输出，从而输出大小为$$1 \times C$$，如下图所示：

![gap1]({{ '/assets/images/GAP-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

而Flatten层的操作，顾名思义，直接将大小为$$H \times W \times C$$的feature map展平成大小为$$H \ast W \ast C$$的一维向量，再继续后面的dense layer操作。

对于有些实验来说，GlobalAveragePooling要比Flatten效果更好。


## 8. weight decay

weight decay是在每次参数更新的梯度基础上再减去一个scaling的梯度（也就是下面的$$\lambda \theta_t$$）：

$$\theta_{t+1} = (1-\lambda) \theta_t - \alpha \nabla f_t (\theta_t)$$

其中$$\theta$$是模型参数，$$\alpha$$是学习率，$$\nabla f_t (\theta_t)$$是$$t$$时刻loss的梯度。

实际上，weight decay和给损失函数加上$$L_2$$ penalization等价。损失函数为$$f(\theta)$$，则新的损失函数为：

$$f(\theta)^{reg} = f(\theta) + \frac{\lambda^{'}}{2} \lVert \theta \rVert_2^2$$

但weight decay也仅仅是在SGD这种算法下和$$L_2$$ penalization是等价的，和别的算法（比如Adam）是没有直接关系的。


## 9. 训练神经网络的tips

机器学习博客TheOrangeDuck的作者，育碧蒙特利尔实验室的机器学习研究员Daniel Holden，根据自己工作中失败的教训，整理了一份神经网络出错原因清单，一共 11 条。

中文版：https://www.leiphone.com/category/yanxishe/ECKXN0Uuj6bHgk96.html

原版：https://theorangeduck.com/page/neural-network-not-working



---
