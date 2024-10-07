## Keypoint matching

### 1. [Multi-Image Semantic Matching by Mining Consistent Features](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Multi-Image_Semantic_Matching_CVPR_2018_paper.pdf)

*Qianqian Wang, Xiaowei Zhou, Kostas Daniilidis*

*CVPR 2018*

**Abstract**

这篇论文提出了一种多张图片匹配的方法来估计多张图片之间的语义对应关系。和之前需要优化所有的成对的对应关系的方法不一样，我们提出的方法只需要识别并匹配图片集合里一个稀疏的features集合就可以。利用这种方法，我们提出的方法可以去除那些不可重复的features，而且对于上千张图片也是可以操作的。我们还提出了一个low-rank的约束来确保在整个图片集合里的features对应关系具有geometric consistency。我们的方法除了在multi-graph matching和semantic flow任务上表现很好之外，我们还可以用这个方法来重构object-class模型，并且从没有任何标注的图片中找到object-class landmarks。


**1. Introduction**

计算图片之间的feature对应关系在CV领域是个很基础的问题。low-level的geometric features（比如说SIFT）对于相同场景下的图片匹配做得很好。最近，对于semantic matching的兴趣越来越高涨，比如说在不同的object instances或者scenes之间建立semantic对应关系。semantic matching这个领域大多数的研究工作集中在成对的情况下，也就是说每次只考虑一个图片对（两张）。而在多张图片中找到consistent correspondences在很多情况下很重要，比如说object-class model reconstruction，automatic landmark annotation等。这篇文章的关注点就是多张图片的semantic matching任务。

尽管semantic matching和multi-image matching问题已经有了很多的进展（related work），但是下述的问题仍然存在。首先，semantic matching的可重复的feature point的寻找仍然是个未解决的问题。之前的工作通过使用所有的pixels（dense flow）或者随机挑选points来避开这个问题，但是导致的结果就是大量的不可重复的features，它们导致和其它的图片之间建立不了真正的对应关系。其次，之前的multi-image matching方法主要优化的是cycle consistency，而并没有同时考虑geometric consistency。现在已经有了很多方法在pairwise的设定下加入了geometric约束（比如说RANSAC和graph matching），但对于multi-image的设定并没有解决方案。最后，现存的multi-image matching方法计算量很大，对于几百张图片就算不了了。所以分析更大的数据集需要更scalable的算法。

在大多数情况下，我们只需要一些具有cycle consistency和geometric consistency的可以高度重复的features能够对应就可以了，而它们只会占据features集合的一小部分。dense的对应关系可以通过插值的方法来实现。因此，和之前那些优化所有的pairwise对应关系的multi-image matching方法不同的是，我们将问题描述为一个feature选择和labeling的问题：从繁复的pairwise对应关系出发，我们在每张图片的初始candidates集合里选取一些feature points，然后通过给它们labels来构建它们在多张图片之间的对应关系。上述挑选和加上标签的过程是通过共同优化cycle consistency和geometric consistency来实现的。将问题描述成这样的形式可以让我们：1）显式的在初始的feature points集合里解决掉那些不可重复的feature points；2）大量的减少变量的数量，从而设计出一个scable的模型，可以解决上千张图片。受到factorization-based structure from motion的一些经典结果的启发，我们提出一个low-rank的约束来使得multi-image matching具有geometric consistency，而且其有很高效的优化算法。fig 1给了个例子来解释问题描述以及我们所提出的方法。

![eg]({{ '/assets/images/MATCHING-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. 给定多张图片的初始feature candidates和每两张图片之间的带有噪音的features之间的对应关系，我们提出的方法从这些feature points中识别出一部分可靠的，并且为它们在所有的图片间构建cycle consistent和geometric consistent的对应关系。上述figure给了一个通过我们提出的方法从1000张图片中挑选出的可靠的feature points的例子（彩色的十字表示feature points）。十字的颜色代表着其之间的对应关系。上述figure的最后一列显示了一张图片的初始的feature points集合（上面）以及手动标注的landmarks（下面）。有意思的是，用我们的方法（unsupervised的方法）来找到的可靠的feature points和手动标注的landmarks高度相似。*

我们这篇文章的主要贡献如下：

* 我们提出了一个新的方法来解决multi-image semantic matching的问题，我们将其转换为一个feature selection和labeling的问题。我们所提出的算法可以在图片集合中找到consistent的feature points，并且是scable的，可以处理上千张图片。
* 我们为multi-image matching介绍了一个新的low-rank的约束，从而我们的算法可以同时优化cycle-consistency和geometric consistency。
* 我们在标准的benchmarks上阐述了我们的方法的很好的效果。我们同时也展示了两个应用：1）从一系列同类但不同instance的图片出发重建3D object-class models，而且并不需要任何手动标注；2）匹配了1000张猫头的图片并且发现算法自动找到的feature points十分具有代表性，它们和表示eyes，ears，mouth等图片的关键点很重合，所以说我们这个方法在automatic landmark detection这个任务上也是很有潜力的。

**2. Related work**

**Image Matching**

在经典的image matching任务里，图片之间的稀疏的feature对应关系是通过low-level的geometric feature detectors（比如说corners和covariant区域）和descriptors（比如说SIFT，SURF，HOG等）来估计的。geometric consistency是通过使用RANSAC作为一个后续的处理步骤或者通过解决一个最小化图片之间的geometric distortion的graph matching问题来实现的。很多最近的工作尝试在不同的场景下找到semantic对应关系。hierarchical matching和region-based方法已经被用来找到图片里的high-level的semantics。

**Learning detectors and descriptors**

最近的结果表明利用CNN获取的deep features在matching领域很有效果，并且远远超过了手动设计的features，即便CNN不是针对matching任务来训练的也是一样。监督学习被用来显式的学习descriptors。监督学习的标签来自于手动标注的对应关系、图片transformations和一些额外的信息，比如说CAD models等。同时，也有一些研究尝试学习可重复并且对于transformations鲁棒的feature detectors。这些方法可以被看成一种非监督的学习方式，它们从图片集合中学到了可靠并且consistent的对应关系。


**Multi-image matching**

multi-image matching在技术上和joint maching方法相关。大多数现存的方法都是利用cycle-consistency来改进pairwise对应关系。很多方法被提了出来，比如说unclosed cycle elimination，constrained local optimization，spectral relaxation以及convex relaxation。我们所提出的方法和上述都不同，因为我们希望能找到在所有图片上最consistent的features而不是在所有的pairwise对应关系之间进行优化，从而使得我们的方法更加的scable。而且，我们介绍了一个low-rank的约束来对于所选择的feature points做geometric consistency的优化。在[这篇文章](https://openaccess.thecvf.com/content_iccv_2015/papers/Yan_A_Matrix_Decomposition_ICCV_2015_paper.pdf)里为了multi-graph matching提出的matrix decomposition的方法在graph edges上采用了同样的low-rank约束。在这篇文章里，我们的low-rank约束直接加在feature points locations上，算法更加高效。最近也有的方法提出一种高效的方法来为matching任务寻找具有辨别型的feature points clusters，但是他们并没有考虑geometric consistency的约束。



**3. Preliminaries and notation**

**3. 1 Pairwise matching**

给定n张需要去匹配的images，每张image有$$p_i$$个feature points，对于每个image pair $$(i,j)$$的pairwise feature对应关系可以用一个partial permutation矩阵$$P_{i,j} \in \{0, 1\}^{p_i, p_j}$$来表示，其还需要满足一个限制条件：doubly stochastic constraints，

$$0 \leq P_{i,j} 1 \leq 1$$

$$0 \leq P_{i,j}^T 1 \leq 1$$

$$P_{i,j}$$可以通过在满足上述条件下，最大化其自身和feature similarities之间的inner product来实现。这是一个linear assignment问题，这个问题已经被研究透彻了，并且可以用HUngarian algorithm来解决。寻找$$P_{i,j}$$还可以被构造成一个graph matching问题，之后转换为一个quadratic assignment问题（QAP）。而这个问题通过最大化一个既含有local compatibilities（feature similarity）信息又含有structural compatibilities（spatial rigidity）信息的目标函数来找到assignment。尽管QAP是NP-hard的，但很多高效的算法已经被提出来近似地解决这个问题。我们使用通过linear matching或者graph matching得到的结果，$$W_{i,j} \in R^{p_i \times p_j}$$，作为我们的输入。

**3.2 Cycle consistency**

近期的工作[1](https://arxiv.org/pdf/1402.1473.pdf), [2](https://pages.cs.wisc.edu/~pachauri/perm-sync/assignmentsync.pdf)和[3](https://link.springer.com/content/pdf/10.1007/978-3-319-10590-1_27.pdf)提出使用cycle consistency作为一个constraint来进行多张images的匹配。如果下面的条件对于任意的图片$$i,j,k$$都满足，那么所有的图片对之间的对应关系就是cyclically consistent的：

$$P_{i,j} = P_{i,z}P_{z,j}$$

cycle consistency可以通过介绍一个虚拟的universe来更加清晰的描述，这个universe定义为这些图片构成的集合的unique features [2](https://pages.cs.wisc.edu/~pachauri/perm-sync/assignmentsync.pdf)。universe里的每个feature point都必须至少被一张图片所记录到，并且还要建立与这张图片上某个feature point的对应关系。假设图片i和universe之间的对应关系可以用一个partial permutation矩阵$$X_i \in \{0,1\}^{p_i \times u}$$来表示，其中$$u$$是universe里feature points的个数，而且$$u \geq p_i$$对于所有的$$i$$都成立。因为对于universe来说，我们假设cycle-consistent是成立的，所以说pairwise对应关系就可以被表示为

$$P_{i,j} = X_i X_j^T$$

我们将所有的partial matrix按照下面的方式连起来：

![lql]({{ '/assets/images/MATCHING-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

有结果表明，集合$$\{P_{ij}| \forall i,j\}$$是cyclically consistent的当且仅当$$P$$可以被分解为$$XX^T$$。


**4. Proposed methods**

**4.1 Matching by labeling**

回忆一下，$$X \in \{0,1\}^{m \times u}$$是从image feature points到universe feature points的一个map，其中$$m$$和$$u$$分别表示image的local features的个数和universe的features的个数。另一种对$$X$$的解释是$$X$$的每一行是image里的feature的一个label。对于不同图片的features，如果它们的label相同，那么就是相互对应的。为了能够处理所有的图片里的features，之前的工作经常会用一个很大的$$u$$。

但是，并不是图片里的每一个feature都适合拿来做匹配。尤其是在semantic matching的任务里，很多的随机或者平均采样的features在多张图片之间是nonrepeatable的，它们应该在matching的时候被剔除。通过这个事实的启发，我们从大量的feature points里只挑选出$$k$$个，而$$k$$是我们预先设定好的超参数，也就是说每张图片只有$$k$$个feature points被选出来做matching。

>这个$$k$$需要针对图片里所包含的object的不同的种类做不同的调整，而不能通过某种方式学习到，这可以是将来的一个拓展的方向。

假设图片$$i$$的feature points和universe之间的对应关系的partially permutation矩阵是$$X_i \in \{0.1\}^{p_i \times k}$$。每个$$X_i$$是一个partial permutation矩阵，而且是一个瘦长型的矩阵，因为列数为$$k$$，是个较小的值。这个矩阵需要满足：

$$ 0 \leq X_i 1 \leq 1$$

$$ X_i^ T 1 = 1$$

也就是说每个图片里的feature points可能在universe里没有对应的点，有也最多只有一个。但universe里的每个点在每张图片里都有对应的feature points。

>这和我们考虑的情况就不一样了，因为我们考虑的是3D的object，每张图片可能不含有全部的关键点的信息，所以说每列的和也不是1。

>而且这种方法也不能有occlution，也就是说所有的图片里的object的所有的关键点都要能看得到。

集合$$\lbrace X_i, 1 \leq i \leq n \rbrace$$就是我们这篇工作里需要去估计的东西。$$X_iX_j^T$$就会给出图片$$i$$和图片$$j$$之间的feature points对应关系。根据3.2里说的，只要$$P$$能分解为$$X_iX_j^T$$，那么其就是cycle-consistent的，而此处就直接通过$$X_iX_j^T$$来表示$$P_{i,j}$$了，所以当然是cycle-consistent的。因为我们想要在initial pairwise matching的基础上找到具有cycle-consistent的并且具有辨识度的feature points，我们通过最小化initial pairwise matching的结果和利用$$X_iX_j^T$$算出来的图片$$i$$和图片$$j$$之间的cycle-consistent的匹配结果，来估计$$X$$的值：

$$ \min_{X} \frac{1}{4}||W - XX^T||^F_2$$

$$s.t. X_i \in P^{p_i \times k}, 1 \leq i \leq n$$

其中$$P$$代表所有的partial permutation矩阵的集合，$$W \in R^{m \times m}$$是$$W_{i,j}$$的集合，其中$$m = \Sigma^t_{i=1} p_i$$是所有图片的feature points的数量的总和。通过解决上述的优化问题，我们就能够找到这个图片集合里最具有代表性的$$k$$个互相匹配的feature points，而且它们满足cycle-consistent的性质。


**4.2 Geometric constraint**

假设我们对同一个场景的$$n$$张图片找到了$$k$$个匹配的feature points。我们使用$$M_i \in R^{2 \times k}$$来表示在图片$$i$$里的这$$k$$个feature points的二维坐标，而且是按照顺序排列的。将所有的这样$$n$$个矩阵$$M_i$$按照行连接起来，就得到了一个$$M \in R^{2n \times k}$$的矩阵，其中每列就是每个feature point。$$M$$在structure from motion里叫做measurement matrix [4](https://link.springer.com/article/10.1007/BF00129684)。有结果表明，在orthographic projection处理之后，$$M$$的秩是4。

在我们的问题里，我们将图片$$i$$里所有的feature points的二维坐标表示为$$C_i \in R^{2 \times p_i}$$。之后，通过$$X_i$$我们可以选出$$k$$个feature points，而这$$k$$个feature points在图片$$i$$里的二维坐标就是$$M_i^{'} = C_iX_i$$，其中$$M_i^{'}$$储存了所选中的$$k$$个feature points的在图片$$i$$里的二维坐标信息，而且是按顺序排列的。

同样的，我们也可以将$$M_i^{'}$$按行来排列，从而得到一个矩阵$$M^{'} \in R^{2n \times k}$$：

![ma]({{ '/assets/images/MATCHING-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

如果feature points都是正确的被挑选出来并且正确的被贴上label，那么矩阵$$M^{'}$$在经过orthographic projection之后就应该是秩为4的。即使object是non-rigid的，$$M^{'}$$仍然可以被一个low-rank矩阵近似 [5](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.230.5079&rep=rep1&type=pdf)。上述的结论可以很好的被利用起来，从而更好的来估计$$X$$的值。假设$$M^{'}$$的真实的秩不比$$r$$大。最小化下面的项可以让我们对于所选择的$$k$$个feature points加上geometric consistent的约束：

$$f_{geo} = \frac{1}{2}||M^{'} - Z||^F_2 = \frac{1}{2}\Sigma^n_{i=1} ||C_iX_i - Z_i||^F_2$$

其中$$Z \in R^{2n \times k}$$是一个辅助变量，其秩不大于$$r$$，并且$$Z_i \in R^{2 \times k}$$代表着矩阵$$Z$$的第$$(2i-1)$$行和第$$2i$$行。


>这里的操作很奇怪，为什么要引入一个新的$$Z$$，为什么不直接SVD分解矩阵$$M^{'}$$，之后挑选出最大的r个eigenvalue重新组合成一个矩阵，再来最小化$$M^{'}$$和这个新矩阵之间的Frobius norm。


**4.3 Formulation**

将cycle-consistency和geometric consistency联合起来，我们就获得了最后我们需要优化的目标函数：

$$ \min_{X,Z} \frac{1}{4}||W - XX^T||^F_2 + \frac{\lambda}{2}\Sigma^n_{i=1} ||C_iX_i - Z_i||^F_2$$

$$s.t. X_i \in P^{p_i \times k}, 1 \leq i \leq n, rank(Z) \leq r$$

其中$$\lambda$$控制geometric constraint所占的比重。


**4.4 Optimization**

对于上述的optimization问题，我们通过block coordinate descent方法来优化，也就是说交替的固定其它的变量优化其中一部分变量。

我们曾经尝试将上述optimization问题里对$$X$$矩阵要求点是0或者1的离散条件，改成$$X$$里的点是0到1之间的连续的值，这样的改动在quadratic assignment problems里很常见。然而，我们观察到如果我们做了这样的改动，那么$$C_iX_i = Z_i$$这个约束对于任意的$$Z_i$$来说就是个ill-posed问题，从而geometric constraint就不管用了。因此，我们要保留$$X$$的integer constraint。

>这里值得再深究

因此，为了让上述的optimization问题能够操作，我们将optimization式子里的两项解耦，也就是让它们没有关系，我们引入一个辅助变量$$Y \in R^{m \times k}$$，在第一项中用$$Y$$来替代$$X$$，并且再加入一项来使得$$X$$和$$Y$$尽量靠近，注意到$$Y$$里每个元素的值不是离散的0或者1，而是实数域。从而我们的optimization问题就变成了下面这样：

$$ \min_{X,Z} \frac{1}{4}||W - YY^T||^F_2 + \frac{\lambda}{2}\Sigma^n_{i=1} ||C_iX_i - Z_i||^F_2 + \frac{\rho}{2}||X - Y||^F_2$$

$$s.t. X_i \in P^{p_i \times k}, 1 \leq i \leq n, Y \in C, rank(Z) \leq r$$

其中$$C$$表示满足下列约束关系的矩阵集合：

$$0 \leq Y \leq 1, 0 \leq Y_i1 \leq 1, Y_i^T1 = 1, 1 \leq i \leq n$$

其中$$\lambda$$和$$\rho$$分别用来控制X和Y之间的差距以及geometric constraint所占的比重。当$$\rho$$趋近于无穷大的时候，上述optimization问题则等价于原始的那个。

将之前的optimization问题改造成上述的形式，是因为改造完之后，每个subproblem都变得很简单了。我们用以下的方式交替更新Y，X和Z。

Y是通过projected gradient descent方法来更新的：

$$ Y \leftarrow \Pi_C(Y-\eta(YY^TY - WY + \rho(Y-X))) $$

其中$$\Pi_C$$表示到$$C$$上的projection，$$\eta > 0$$是step size。我们先一直优化$$Y$$直到它收敛，再优化X和Z。

每个$$X_i$$是通过Hungarian algorithm来更新的，它的cost matrix构造如下：

$$H_i = \lambda D(C_i, Z_i) - 2 \rho Y_i$$$

其中$$D(C_i, Z_i) \in R^{p_i \times k}$$表示$$C_i$$和$$Z_i$$里每个二维点之间的Eucldean距离。

$$Z$$是通过SVD分解来更新的：

$$Z = U \Sigma V^T$$

其中U和V的columns都是来自于$$M^{'}$$的singular vectors，而$$\Sigma$$则是选取了$$M^{'}$$矩阵最大的$$r$$个singular values构造的对角矩阵。（也就是说构造一个rank为r的矩阵Z，其是用SVD分解$$M^{'}$$来实现的）

为了更好的收敛，我们利用一种逐步增大$$\rho$$的方式。对于每个$$\rho$$，我们按照上述方式先更新Y，再交替更新X和Z，直到目标函数不再减小。因为每次更新不会增加目标函数的值，所以说我们最终可以达到局部收敛。在我们的实验里，我们设置$$\rho$$为1，10，199，设置$$\lambda=1$$以及$$r=4$$。

因为这个优化问题是nonconvex的，而且还涉及连续和离散两种变量，所以说需要很好的初始化的技巧。我们首先按照解下述优化问题来初始化Y的值（忽略geometrical consistency）：

$$ \min_{Y} \frac{1}{4}||W - YY^T||^F_2 $$

$$s.t. Y \in C$$

这个优化问题的解是：

$$ Y \leftarrow \Pi_C(Y-\eta(YY^TY - WY)) $$

而X的初始化可以通过将Y离散化而得来，也就是将Y的值变成0或1。


**5. Experiments**

**5.1 Multi-graph matching**

我们先再multi-graph matching任务上验证我们方法的有效性，在这个任务里，feature point的位置是有标注的，但是它们的对应关系需要被正确预计。匹配的准确率是用recall来衡量的，定义为算法所找到的的真实的对应关系除以所有的真实的对应关系的总和。

我们使用CMU dataset和WILLOW Object Class dataset来做实验。CMU dataset包含hotel sequence和house sequence。每一帧图片都标注了30个feature points的位置并给了它们的SIFT特征。WILLOW Object Class dataset提供了五个object类型的图片（car，duck，motorbike，face和winebottle）以及为每张图片提供10个标注位置的特征点。因为物体的外观在每个类别的众多图片中差别很大，所以几何描述子SIFT很难作用。所以我们采用了利用预训练的CNN网络来提取deep features的方法来获取特征。详细的说，就是每张图片都喂给了一个AlexNet（在ImageNet上预训练），然后将Conv4和Conv5的关键点对应位置的输出连起来作为特征。对于这两个数据集，初始的pairwise对应关系都是通过linear matching solver Hungarian algorithm所得到的，之后再用我们本文中提出的方法来操作。三个有公开代码的方法被用作baselines：[spectral method](https://pages.cs.wisc.edu/~pachauri/perm-sync/assignmentsync.pdf)，[MatchLift](https://arxiv.org/pdf/1402.1473.pdf)和[MatchALS](https://openaccess.thecvf.com/content_iccv_2015/papers/Zhou_Multi-Image_Matching_via_ICCV_2015_paper.pdf)。对于所有的方法，universe的feature points的个数都被设置为每张图片里标注的points的数量。

recall的结果由Table 1显示。Table 1表明我们的方法要比其它的方法要好很多。Table 1还显示了两个结果：1）如果利用[graph matching solver RRWM](https://www.researchgate.net/profile/Minsu-Cho-5/publication/221304918_Reweighted_Random_Walks_for_Graph_Matching/links/54c50ae80cf256ed5a98633c/Reweighted-Random-Walks-for-Graph-Matching.pdf)来取代之前的初始的pairwise对应关系，匹配的准确率在除了duck的情况下都能到100%。2）如果忽略geometrical consistency，匹配的准确率会急剧降低。

![Ta]({{ '/assets/images/MATCHING-4.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Table 1. 在CMU dataset和WILLO Object Class dataset上的recall的结果。我们提出的方法与spectral method，MatchLift以及MatchALS方法进行了对比。$$Ours^-$$指的是我们的方法没有geometric consistency的情况。$$Input^+$$和$$Ours^+$$分别指的是使用RRWM graph matching方法来进行初始的pairwise对应关系的计算，在没用我们的方法和用了我们的方法之后的结果。

fig 2显示了利用geometrical consistency我们可以纠正geometrically distorted matches。

![exam]({{ '/assets/images/MATCHING-5.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*figure 2. 上面一行是没有使用geometrical consistency的情况，下面一行是使用了的情况。蓝色的十字表示正确的对应关系，而红色的十字表示错误的对应关系。*

我们所提出的方法还能够自动选择可靠的feature points用来做matching。fig 3给了个例子，我们给CMU dataset加了随机的feature points。而我们的方法可以去掉这些随机加入的feature points。

![add]({{ '/assets/images/MATCHING-6.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*figure 3. 正确的匹配是绿色的，错误的匹配是红色的。上面一行是只用了pairwise方法来找到的对应关系，而下面一行是再经过我们提出的方法找到的对应关系。我们还在原数据集的每一帧中加入了30个随机挑选位置的feature points作为outliers。我们设置$k=30$。*


**5.2 Dense semantic matching**

在这一节里，我们通过将我们提出的方法和region based semantic flow methods（比如说，[proposal flows](https://arxiv.org/pdf/1703.07144.pdf)）结合起来的方式来在dense matching这个任务上测试我们提出的方法的效果。在proposal flow method里，图片之间的region proposals的对应关系先被估计出来，然后再转换为一个dense flow field。对于一系列图片，我们将所提出的方法应用在proposal flow上，从而改进proposals的pairwise对应关系，因此改进dense flow。

我们通过在PF-WILLOW dataset上来衡量效果，这个数据集将WILLOW Object Class dataset分成了10个子类。它们分别是car(S), (G), (M), duck(S), motorbike(S), (G), (M), winebottle(w/oC), (w/C)和(M)，其中(S)和(G)表示side和general角度，而(C)表示背景很杂乱，(M)表示混合角度。每个子类都包含10张不同object的图片。

percentage of correct keypoints (PCK)被用来作为衡量标准。它衡量了当利用estimated flow将一张图片的标注的keypoints转移到另一张图片上的时候位置正确的keypoints的比例。一个预测的feature point被认为是在正确的位置上如果它到groundtruth point的距离在$$\alpha max(h,w)$$以内，其中$$\alpha$$在0到1之间，$$h$$和$$w$$分别是object bounding box的高和宽。对于proposal flow，selective search（SS）被用来作为proposal generator，HOG是feature descriptor，local offset matching（LOM）用作geometric matching strategy。每张图片都会被提取500个proposals，并用来做matching以及生成dense flow。在我们的算法里，每个proposal都被当做一个feature point，每个proposal的中心点就作为我们geometric consistency里所用的coordinate。对于每个类别我们都设定选取10个feature points，也就是$$k=10$$。

Table 2显示我们的方法改进了原论文里的效果。fig 4给了可视化的效果。

![add]({{ '/assets/images/MATCHING-7.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Table 2*

![add]({{ '/assets/images/MATCHING-8.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*figure 4. 我们将source images扭曲成target image，使用了proposal flow估计出来的dense correspondence，也使用了我们提出的方法进行了优化。*


**5.3 Object-class model reconstruction**

将不同的object instances匹配起来在从一系列图片来重建object-class model这个任务里一直都是一个主要的挑战。一些前期的工作依赖于图片里标注的keypoints [6](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Carreira_Virtual_View_Networks_2015_CVPR_paper.pdf), [7](https://openaccess.thecvf.com/content_cvpr_2015/papers/Kar_Category-Specific_Object_Reconstruction_2015_CVPR_paper.pdf), [8](https://openaccess.thecvf.com/content_cvpr_2014/papers/Vicente_Reconstructing_PASCAL_VOC_2014_CVPR_paper.pdf)。
最近也有工作不需要keypoints标注，但需要object mask来去除背景，[9](https://openaccess.thecvf.com/content_iccv_2015/papers/Zhou_Multi-Image_Matching_via_ICCV_2015_paper.pdf)。我们的方法可以为reconstruction提供consistent的对应关系，而且不需要额外的人工标注。我们利用FG3DCar dataset来验证，匹配一共37张左眼的sedan图片，之后再来重建一个3D模型。我们还额外收集了含有30张不同的motorbikes的dataset。

类似于[9](https://openaccess.thecvf.com/content_iccv_2015/papers/Zhou_Multi-Image_Matching_via_ICCV_2015_paper.pdf)的做法，我们在利用structured forests检测出的图片边缘上均匀采样。而不同于[9]的是，因为我们的方法能够去掉那些nonrepeatable的feature points，所以background里的feature points就自动没了，我们不需要object mask。大约每张图片获取了550个feature points candidates。在5.1里所说的deep features被用作每个feature points的feature，并且之前提到的graph matching solver RRWM被用来做初始pairwise matching。

我们使用precision作为衡量标准，定义为算法所找到的对应关系里真实的对应关系，除以算法找到的所有的对应关系。真实的对应关系的定义和前面在PCK里所用的方法差不多。我们使用不同的$$k$$来测试我们的方法的效果，fig 5显示了结果。

![add]({{ '/assets/images/MATCHING-9.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*figure 5. 早FG3DCar dataset上，对于不同的$$k$$值，precision的结果。*

fig 5的结果表明，使用我们的方法改进的效果要比使用pairwise matching要好得多，pairwise matching在背景很杂乱的时候效果很差。

>但是问题是，用precision来衡量，而且从fig 5也能看出来，在$$k$$很小的时候效果比较好，也就是说在feature points对应关系
越少，效果越好。但这并不能说明什么，因为对于对应关系多的情况，即使它的precision比较低，但也可能是因为多出来的那部分都检测的不对，但这并不能说明它的效果就弱于对应关系少但是效果好的情况。但文章里说的是，所用的对应关系越少，效果越好，说明它们的方法去除掉了那些nonrepeatable的feature points。但这是得不出这个结果的，除非可视化来看到底去除掉了哪些feature points。

fig 6说明了文中提出的方法确实能去除掉很多nonrepeatable的背景中的对应关系（看看就好，没有什么说服力）

![back]({{ '/assets/images/MATCHING-9.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*figure 6. 两个sedan图片之间的匹配结果。正确的和错误的匹配分别用蓝色和红色来表示。上面一行表示的是使用pairwise对应关系得到的结果，下面一行是使用文中提出的方法得到的结果。可以看到很多初始的背景里的feature points都被文中的方法给去掉了。*

对于reconstruction来说，我们简单的利用factorization method来运行了affine reconstruction。最后被选择的feature points，对应关系，以及reconstruction结果在fig 7中被可视化。可以看到，绝大多数的被选择的feature points落在object上并且都能正确匹配，尽管object的外观和视角都很不一样。尽管有一些噪音和没有追踪到的points，我们可以在reconstruction的结果中看到sedan和motorbikes的形状。如果用更成熟的reconstruction方法一定会得到更好的结果。

之前并没有工作在不同的instances之间利用unsupervised的方法进行relative pose estimation。

![rec]({{ '/assets/images/MATCHING-10.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*figure 7. *


**5.4 Automatic landmark annotation**

我们将所提出的方法应用在cat head dataset的前1000张图片上。和之前的实验一样，feature points candidates从图片边缘中采样得到，每张图片得到大约43个candidates。我们将$$k$$设置为10，也就是selected feature points的数量是10。结果在fig 1中显示。如同fig 1所示的那样，初始的candidates在整张图片上均匀分布（包括背景），而selected feature points都在object上，而且对应关系也能在具有不同外观和姿态的object上建立。更加有意思的是，这种自动检测的feature points和人工标注的landmarks重合度很高，都表示了cat的具有特征性的feature points，比如说ears，eyes，mouth等。这个结果说明了我们的方法对于automatic landmark detection这个方向也存在潜力，整个过程模拟了人标注的过程：我们比较了一系列的图片，并且找到了不随着外观、几何形状以及姿态而改变的那部分feature points。


**5.5 Computational complexity**

**6. Conclusion**

我们展示了一个新的方法，其将在多张图片上寻找semantic matching的问题转换为了feature points selection和labeling的问题来解决。我们所提出的方法可以在一系列图片之间建立可靠的feature points对应关系，而且这种对应关系还是cycle consistent和geometry consistent的。实验表明我们的方法要比之前的multi-image matching方法要好，并且对于上千张图片也是scable的。我们同时还阐述了几个可能的应用：改进dense flow estimation，不使用人工标注进行reconstruction object-class models，以及自动进行image landmark标注。


### 2. [A convex relaxation for multi-graph matching](https://openaccess.thecvf.com/content_CVPR_2019/html/Swoboda_A_Convex_Relaxation_for_Multi-Graph_Matching_CVPR_2019_paper.html)

*CVPR 2019*


### 3. [Probabilistic Permutation Synchronization Using the Riemannian Structure of the Birkhoff Polytope](https://openaccess.thecvf.com/content_CVPR_2019/html/Birdal_Probabilistic_Permutation_Synchronization_Using_the_Riemannian_Structure_of_the_Birkhoff_CVPR_2019_paper.html)

*CVPR 2019*


### 4. [HiPPI: Higher-Order Projected Power Iterations for Scalable Multi-Matching](https://openaccess.thecvf.com/content_ICCV_2019/html/Bernard_HiPPI_Higher-Order_Projected_Power_Iterations_for_Scalable_Multi-Matching_ICCV_2019_paper.html)

*ICCV 2019*


### 5. [MultiBodySync: Multi-Body Segmentation and Motion Estimation via 3D Scan Synchronization](https://openaccess.thecvf.com/content/CVPR2021/html/Huang_MultiBodySync_Multi-Body_Segmentation_and_Motion_Estimation_via_3D_Scan_Synchronization_CVPR_2021_paper.html)

*CVPR 2021*


### 6. [Quantum Permutation Synchronization](https://openaccess.thecvf.com/content/CVPR2021/html/Birdal_Quantum_Permutation_Synchronization_CVPR_2021_paper.html)

*CVPR 2021*


### 7. [Graduated Assignment for Joint Multi-Graph Matching and Clustering with Application to Unsupervised Graph Matching Network Learning](https://proceedings.neurips.cc/paper/2020/hash/e6384711491713d29bc63fc5eeb5ba4f-Abstract.html)

*NeurIPS 2020*


### 8. [Associative3D: Volumetric Reconstruction from Sparse Views](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600137.pdf)


*ECCV 2020*


### 9. [All Graphs Lead to Rome: Learning Geometric and Cycle-ConsistentRepresentations with Graph Convolutional Networks](https://arxiv.org/pdf/1901.02078.pdf)

*Arxiv 2019*


### 10. [Unsupervised 3D Reconstruction and Grouping of Rigid and Non-Rigid Categories](http://www.iri.upc.edu/files/scidoc/2344-Unsupervised-3D-Reconstruction-and-Grouping-of-Rigid-and-Non-Rigid-Categories.pdf)

*TPAMI 2022*


### 11. [Unifying Offline and Online Multi-Graph Matching via Finding Shortest Paths on Supergraph](https://ieeexplore.ieee.org/abstract/document/9076840)

*TPAMI 2021*


### 12. [Neural Graph Matching Network: Learning Lawler’s Quadratic Assignment Problem with Extension to Hypergraph and Multiple-graph Matching](https://ieeexplore.ieee.org/abstract/document/9426408)

*TPAMI 2021*


### 13. [Distributed and consistent multi-image feature matching via QuickMatch](https://journals.sagepub.com/doi/pdf/10.1177/0278364920917465)

*IJRR 2020*


### 14. [Robust Multi-Object Matching via Iterative Reweighting of the Graph Connection Laplacian](https://proceedings.neurips.cc/paper/2020/hash/ae06fbdc519bddaa88aa1b24bace4500-Abstract.html)

*NeurIPS 2020*


### 15. [Isometric Multi-Shape Matching](https://openaccess.thecvf.com/content/CVPR2021/html/Gao_Isometric_Multi-Shape_Matching_CVPR_2021_paper.html)

*CVPR 2021*


### 16. [Multiple Graph Matching and Clustering via Decayed Pairwise Matching Composition](https://ojs.aaai.org/index.php/AAAI/article/view/5528)

*AAAI 2020*


### 17. [Multi-View Multi-Human Association With Deep Assignment Network](https://ieeexplore.ieee.org/abstract/document/9693506)

*TIP 2022*


### 18. [Scalable Cluster-Consistency Statistics for Robust Multi-Object Matching](https://ieeexplore.ieee.org/abstract/document/9665954)

*3DV*


### 19. [Fast, Accurate and Memory-Efficient Partial Permutation Synchronization](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Fast_Accurate_and_Memory-Efficient_Partial_Permutation_Synchronization_CVPR_2022_paper.html)

*CVPR 2022*


### 20. [Graph Neural Networks For Multi-Image Matching](https://openreview.net/forum?id=Hkgpnn4YvH)

*ICLR 2020*


### 21. [Higher-order Projected Power Iterations for Scalable Multi-Matching](https://openaccess.thecvf.com/content_ICCV_2019/papers/Bernard_HiPPI_Higher-Order_Projected_Power_Iterations_for_Scalable_Multi-Matching_ICCV_2019_paper.pdf)

*ICCV 2019*


### 22. [Joint Deep Multi-Graph Matching and 3D Geometry Learning from Inhomogeneous 2D Image Collections](https://ojs.aaai.org/index.php/AAAI/article/view/20220)

*AAAI 2022*


### 23. [A-ACT: Action Anticipation through Cycle Transformations](https://arxiv.org/pdf/2204.00942.pdf)

*Arxiv 2022*


### 24. [Quantum Motion Segmentation](https://arxiv.org/pdf/2203.13185.pdf)

*Arxiv 2022*


### 25. [Image Matching from Handcrafted to Deep Features: A Survey](https://link.springer.com/content/pdf/10.1007/s11263-020-01359-2.pdf)

*IJCV 2021*


### 26.[PRNet: Self-Supervised Learning for Partial-to-Partial Registration](https://proceedings.neurips.cc/paper/2019/hash/ebad33b3c9fa1d10327bb55f9e79e2f3-Abstract.html)

*NeurIPS 2019*


### 27. [DeepBBS: Deep Best Buddies for Point Cloud Registration](https://arxiv.org/pdf/2110.03016.pdf)

*3DV 2021*


### 28. [Joint-task Self-supervised Learning for Temporal Correspondence]()

*NeurIPS 2019*


### 29. [Continuous Surface Embeddings](https://proceedings.neurips.cc/paper/2020/hash/c81e728d9d4c2f636f067f89cc14862c-Abstract.html)

*NeurIPS 2020*


### 30. [Learning Facial Representations From the Cycle-Consistency of Face](https://openaccess.thecvf.com/content/ICCV2021/html/Chang_Learning_Facial_Representations_From_the_Cycle-Consistency_of_Face_ICCV_2021_paper.html)

*ICCV 2021*


### 31. [Demystifying Unsupervised Semantic Correspondence Estimation](https://arxiv.org/pdf/2207.05054.pdf)

[POST](https://mehmetaygun.github.io/demistfy.html)

*ECCV 2022*



### 32. [Unsupervised Part Discovery via Feature Alignment](https://arxiv.org/pdf/2012.00313.pdf)

*Arxiv 2020*


### 33. [LAKe-Net: Topology-Aware Point Cloud Completion by Localizing Aligned Keypoints](https://openaccess.thecvf.com/content/CVPR2022/html/Tang_LAKe-Net_Topology-Aware_Point_Cloud_Completion_by_Localizing_Aligned_Keypoints_CVPR_2022_paper.html)

*CVPR 2022*
