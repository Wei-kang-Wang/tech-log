---
layout: post
comments: false
title: "[论文]Graph Neural Networks"
date: 2021-11-29 01:09:00
tags: paper-reading
---

> This post is a summary of graph neural network papers.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

## GNN Architecture

研究GNN架构/体系，基本涵盖了GNN最经典、必读的文章，如基于谱图卷积的GCNs三部曲， 序列图神经网络GGNN，基于注意力机制的GAT等。

### 1. Semi-Supervised Classification with Graph Convolutional Networks

*Thomas N. Kipf, Max Welling*

*NeuIPS 2017*

### 2. Graph Attention Networks

*Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio*

*ICLR 2018*

### 3. Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering

*NeuIPS 2016*

### 4. Predict then Propagate: Graph Neural Networks meet Personalized PageRank

*Johannes Klicpera, Aleksandar Bojchevski, Stephan Günnemann*

*ICLR 2019*

### 5. Gated Graph Sequence Neural Networks

*Li, Yujia N and Tarlow, Daniel and Brockschmidt, Marc and Zemel, Richard*

*ICLR 2016*

### 6. Inductive Representation Learning on Large Graphs

*William L. Hamilton, Rex Ying, Jure Leskovec*

*NeuIPS 2017*

### 7. Deep Graph Infomax

*Petar Veličković, William Fedus, William L. Hamilton, Pietro Liò, Yoshua Bengio, R Devon Hjelm*

*ICLR 2019*

### 8. Representation Learning on Graphs with Jumping Knowledge Networks

*Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, Stefanie Jegelka*

*ICML 2018*

### 9. DeepGCNs: Can GCNs Go as Deep as CNNs?

*Guohao Li, Matthias Müller, Ali Thabet, Bernard Ghanem*

*ICCV 2019*

### 10. DropEdge: Towards Deep Graph Convolutional Networks on Node Classification

*Yu Rong, Wenbing Huang, Tingyang Xu, Junzhou Huang*

*ICLR 2020*

### 11. [Breaking the Limits of Message Passing Graph Neural Networks](http://proceedings.mlr.press/v139/balcilar21a/balcilar21a.pdf)

[code](https://github.com/balcilar/gnn-matlang)

*Muhammet Balcilar, Pierre Heroux, Benoit Gauzere, Pascal Vasseur, Sebastien Adam, Paul Honeine*

*icml 2021*


## Large-scale Training

研究大规模图训练，如经典的GraphSAGE/PinSAGE中做大规模邻居结点采样的方法。补充一点，这类研究实际上在各类框架上也有，例如DGL，PyG，Euler等。一方面可以进行训练方式改进，如邻居结点采样/子图采样等；另一方面也可以进行训练环境的分布式改造，分布式环境下，原始大图切割为子图分布在不同的机器中，如何进行子图间的通信、跨图卷积等，也是很有挑战的难点。

### 1. Inductive Representation Learning on Large Graphs

*William L. Hamilton, Rex Ying, Jure Leskovec*

*NeuIPS 2017*

### 2. FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling

*Jie Chen, Tengfei Ma, Cao Xiao*

*ICLR 2018*

### 3. Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks

*Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, Cho-Jui Hsieh*

*KDD 2019*

### 4. GraphSAINT: Graph Sampling Based Inductive Learning Method

*Hanqing Zeng, Hongkuan Zhou, Ajitesh Srivastava, Rajgopal Kannan, Viktor Prasanna*

*ICLR 2020*

### 5. GNNAutoScale: Scalable and Expressive Graph Neural Networks via Historical Embeddings

*Matthias Fey, Jan E. Lenssen, Frank Weichert, Jure Leskovec*

*ICML 2021*

### 6. Scaling Graph Neural Networks with Approximate PageRank

*Aleksandar Bojchevski, Johannes Klicpera, Bryan Perozzi, Amol Kapoor, Martin Blais, Benedek Rózemberczki, Michal Lukasik, Stephan Günnemann*

*KDD 2020*

### 7. Stochastic training of graph convolutional networks with variance reduction

*Jianfei Chen, Jun Zhu, and Le Song*

*ICML 2018*

### 8. Adaptive sampling towards fast graph representation learning

*Wenbing Huang, Tong Zhang, Yu Rong, and Junzhou Huang*

*NeuIPS 2018*

### 9. SIGN: Scalable Inception Graph Neural Networks

*Fabrizio Frasca, Emanuele Rossi, Davide Eynard, Ben Chamberlain, Michael Bronstein, Federico Monti*

### 10. Simplifying Graph Convolutional Networks

*Felix Wu, Tianyi Zhang, Amauri Holanda de Souza Jr., Christopher Fifty, Tao Yu, Kilian Q. Weinberger*

*ICML 2019*


## Self-supervised Learning and pre-training

研究图的自监督学习和预训练。预训练在NLP任务中大放光采，研究者希望此类技术在图上也能够进行预训练，并期望更好地辅助下游的任务。经典的自监督学习方法如对比学习等。

### 1. Strategies for pre-training graph neural networks

*Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, Leskovec Jure*

*ICLR 2020*

### 2. Deep graph infomax

*Velikovi Petar, Fedus William, Hamilton William L, Li Pietro, Bengio Yoshua, Hjelm R Devon*

*ICLR 2019*

### 3. Inductive representation learning on large graphs

*Hamilton Will, Zhitao Ying, Leskovec Jure*

*NeurIPS 2017*

### 4. Infograph: Unsupervised and semi-supervised graph-level representation learning via mutual information maximization

*Sun Fan-Yun, Hoffmann Jordan, Verma Vikas, Tang Jian*

*ICLR 2020*

### 5. GCC: Graph contrastive coding for graph neural network pre-training

*Jiezhong Qiu, Qibin Chen, Yuxiao Dong, Jing Zhang, Hongxia Yang, Ming Ding, Kuansan Wang, Jie Tang*

*KDD 2020*

### 6. Contrastive multi-view representation learning on graphs

*Hassani Kaveh, Khasahmadi Amir Hosein*

*ICML 2020*

### 7. Graph contrastive learning with augmentations

*Yuning You, Tianlong Chen, Yongduo Sui, Ting Chen, Zhangyang Wang, Yang Shen*

*NeurIPS 2020*

### 8. GPT-GNN: Generative pre-training of graph neural networks

*Ziniu Hu, Yuxiao Dong, Kuansan Wang, Kai-Wei Chang, Yizhou Sun*

*KDD 2020*

### 9. When does self-supervision help graph convolutional networks?

*Yuning You, Tianlong Chen, Zhangyang Wang, Yang Shen*

*ICML 2020*

### 10. Deep graph contrastive representation learning

*Yanqiao Zhu, Yichen Xu, Feng Yu, Qiang Liu, Shu Wu, Liang Wang*

*ICML 2020*


## Oversmoothing and deep GNNs

研究GNN的过平滑问题和深层GNN。GNN的卷积操作在层数过多时会使得每个结点几乎都能间接融合其它所有结点的信息，导致所有结点的表征趋于相同，因此通常GNN的层数不宜设置过大。例如：推荐场景中的GNN通常设置为1层或2层即可达到最佳性能。如何突破这种瓶颈，使得GNN的层数更深，性能更好，也是近年来比较热门的研究方向。

### 1. Representation Learning on Graphs with Jumping Knowledge Networks

*Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, Stefanie Jegelka*

*ICML 2018*

### 2. Deeper insights into graph convolutional networks for semi-supervised learning

*Qimai Li, Zhichao Han, Xiao-ming Wu*

*AAAI 2018*

### 3. Predict then Propagate: Graph Neural Networks meet Personalized PageRank

*Johannes Klicpera, Aleksandar Bojchevski, Stephan Günnemann*

*ICLR 2019*

### 4. DeepGCNs: Can GCNs Go as Deep as CNNs?

*Guohao Li, Matthias Müller, Ali Thabet, Bernard Ghanem*

*ICCV 2019*

### 5. Layer-Dependent Importance Sampling for Training Deep and Large Graph Convolutional Networks

*Difan Zou, Ziniu Hu, Yewen Wang, Song Jiang, Yizhou Sun, Quanquan Gu*

*NeurIPS 2019*

### 6. DeeperGCN: All You Need to Train Deeper GCNs

*Guohao Li, Chenxin Xiong, Ali Thabet, Bernard Ghanem*

*arXiv 2020*

### 7. PairNorm: Tackling Oversmoothing in GNNs

*Lingxiao Zhao, Leman Akoglu*

*ICLR 2020*

### 8. DropEdge: Towards Deep Graph Convolutional Networks on Node Classification

*Yu Rong, Wenbing Huang, Tingyang Xu, Junzhou Huang*

*ICLR 2020*

### 9. Simple and Deep Graph Convolutional Networks

*Ming Chen, Zhewei Wei, Zengfeng Huang, Bolin Ding, Yaliang Li*

*ICML 2020*

### 10. Towards Deeper Graph Neural Networks

*Meng Liu, Hongyang Gao, and Shuiwang Ji*

*KDD 2020*


## Graph Robustness

研究图的鲁棒性，对噪声/恶意攻击的抗干扰和防御能力。比如去掉或者添加一些图结点或边，对下游任务的性能不应该造成太大的负面影响。这部分研究包括研究图攻击和图防御，挺多思想借鉴自对抗学习，如经典的GAN。

Adversarial attacks on neural networks for graph data. Zügner Daniel, Akbarnejad Amir, Günnemann Stephan. KDD 2018.
Adversarial attack on graph structured data. Dai Hanjun, Li Hui, Tian Tian, Huang Xin, Wang Lin, Zhu Jun, Song Le. ICML 2018.
Adversarial attacks on graph neural networks via meta learning. Zügner Daniel, Günnemann Stephan. ICLR 2019.
Robust graph convolutional networks against adversarial attacks. Zhu Dingyuan, Zhang Ziwei, Cui Peng, Zhu Wenwu. KDD 2019.
Adversarial attacks on node embeddings via graph poisoning. Bojchevski Aleksandar, Günnemann Stephan. ICML 2019.
Topology attack and defense for graph neural networks: An optimization perspective. Xu Kaidi, Chen Hongge, Liu Sijia, Chen Pin-Yu, Weng Tsui-Wei, Hong Mingyi, Lin Xue. IJCAI 2019.
Adversarial examples on graph data: Deep insights into attack and defense. Wu Huijun, Wang Chen, Tyshetskiy Yuriy, Docherty Andrew, Lu Kai, Zhu Liming. IJCAI 2019.
Certifiable robustness and robust training for graph convolutional networks. Zügner Daniel, Günnemann Stephan. KDD 2019
Graph adversarial training: Dynamically regularizing based on graph structure. Feng Fuli, He Xiangnan, Tang Jie, Chua Tat-Seng. TKDE 2019
Adversarial attack and defense on graph data: A survey. Sun Lichao, Dou Yingtong, Yang Carl, Wang Ji, Yu Philip S, He Lifang, Li Bo. arXiv preprint arXiv:1812.10528 2018.


## Explanability

研究图的可解释性。图结构数据本身能够将不同的Entity链接在一起，天然具备较强的解释潜力。早期的图模型主要基于meta-path来做一些可解释性，近年来也有关于诸多GNN做可解释性的文章，值得一读。

Explainability in graph neural networks: A taxonomic survey. Yuan Hao, Yu Haiyang, Gui Shurui, Ji Shuiwang. ARXIV 2020.
Gnnexplainer: Generating explanations for graph neural networks. Ying Rex, Bourgeois Dylan, You Jiaxuan, Zitnik Marinka, Leskovec Jure. NeurIPS 2019.
Explainability methods for graph convolutional neural networks. Pope Phillip E, Kolouri Soheil, Rostami Mohammad, Martin Charles E, Hoffmann Heiko. CVPR 2019.
Parameterized Explainer for Graph Neural Network. Luo Dongsheng, Cheng Wei, Xu Dongkuan, Yu Wenchao, Zong Bo, Chen Haifeng, Zhang Xiang. NeurIPS 2020.
Xgnn: Towards model-level explanations of graph neural networks. Yuan Hao, Tang Jiliang, Hu Xia, Ji Shuiwang. KDD 2020.
Attribution for Graph Neural Networks. Sanchez-Lengeling Benjamin, Wei Jennifer, Lee Brian, Reif Emily, Wang Peter, Qian Wesley, McCloskey Kevin, Colwell Lucy, Wiltschko Alexander. NeurIPS 2020.
PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks. Vu Minh, Thai My T.. NeurIPS 2020.
Explanation-based Weakly-supervised Learning of Visual Relations with Graph Networks. Federico Baldassarre and Kevin Smith and Josephine Sullivan and Hossein Azizpour. ECCV 2020.
GCAN: Graph-aware Co-Attention Networks for Explainable Fake News Detection on Social Media. Lu, Yi-Ju and Li, Cheng-Te. ACL 2020.
On Explainability of Graph Neural Networks via Subgraph Explorations. Yuan Hao, Yu Haiyang, Wang Jie, Li Kang, Ji Shuiwang. ICML 2021.


## Expressiveness and Generalisability

研究图的表达能力和泛化性，和经典的图同构测试关联，属于GNN比较基础的研究方向，如经典的GIN。

How Powerful are Graph Neural Networks?Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka. ICLR 2019.
Invariant and Equivariant Graph Networks.Haggai Maron, Heli Ben-Hamu, Nadav Shamir, Yaron Lipman. ICLR 2019.
Understanding Attention and Generalization in Graph Neural Networks.Boris Knyazev, Graham W. Taylor, Mohamed R. Amer. NeurIPS 2019.
Provably Powerful Graph Networks.Haggai Maron, Heli Ben-Hamu, Hadar Serviansky, Yaron Lipman. NeurIPS 2019.
Understanding the Representation Power of Graph Neural Networks in Learning Graph Topology.Nima Dehmamy, Albert-Laszlo Barabasi, Rose Yu. NeurIPS 2019.
On the equivalence between graph isomorphism testing and function approximation with GNNs.Zhengdao Chen, Soledad Villar, Lei Chen, Joan Bruna. NeurIPS 2019.
Universal Invariant and Equivariant Graph Neural Networks.Nicolas Keriven, Gabriel Peyré. NeurIPS 2019.
Stability and Generalization of Graph Convolutional Neural Networks.Saurabh Verma and Zhi-Li Zhang. KDD 2019.
Graph Neural Networks Exponentially Lose Expressive Power for Node Classification.Kenta Oono, Taiji Suzuki. ICLR 2020.
Generalization and Representational Limits of Graph Neural Networks.Vikas Garg, Stefanie Jegelka, Tommi Jaakkola. ICML 2020.


## Heterogeneous GNNs
研究异构图神经网络，也一直是热门方向。如HAN等。

Heterogeneous Graph Attention Network. Xiao Wang, Houye Ji, Chuan Shi, Bai Wang, Peng Cui, P. Yu, Yanfang Ye WWW 2019.
Representation Learning for Attributed Multiplex Heterogeneous Network. Yukuo Cen, Xu Zou, Jianwei Zhang, Hongxia Yang, Jingren Zhou, Jie Tang KDD 2019.
ActiveHNE: Active Heterogeneous Network EmbeddingXia Chen, Guoxian Yu, Jun Wang, Carlotta Domeniconi, Zhao Li, Xiangliang Zhang IJCAI 2019
Hypergraph Neural NetworksYifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, Yue Gao. AAAI 2019.
Dynamic Hypergraph Neural NetworksJianwen Jiang, Yuxuan Wei, Yifan Feng, Jingxuan Cao, Yue Gao IJCAI 2019.
HyperGCN: A New Method For Training Graph Convolutional Networks on Hypergraphs.Naganand Yadati, Madhav Nimishakavi, Prateek Yadav, Vikram Nitin, Anand Louis, Partha Talukdar
Type-aware Anchor Link Prediction across Heterogeneous Networks based on Graph Attention Network.Xiaoxue Li, Yanmin Shang, Yanan Cao, Yangxi Li, Jianlong Tan, Yanbing Liu. AAAI 2020
Composition-based Multi-Relational Graph Convolutional NetworksShikhar Vashishth, Soumya Sanyal, Vikram Nitin, Partha Talukdar. ICLR 2020
Hyper-SAGNN: a self-attention based graph neural network for hypergraphs.Ruochi Zhang, Yuesong Zou, Jian Ma. ICLR 2020
Heterogeneous graph transformerHu, Ziniu, Yuxiao Dong, Kuansan Wang, and Yizhou Sun WWW 2020


## GNNs for Recommendation
研究GNN在推荐场景中的应用，这类研究工作很实用，如GCMC, SRGNN, LightGCN等。关于更多在推荐领域的应用可以参见我的文章，推荐系统中二分图表示学习调研。

Graph Convolutional Matrix Completion.Rianne van den Berg, Thomas N. Kipf, Max Welling. KDD 2018.
Session-based recommendation with graph neural networks. Shu Wu, Yuyuan Tang, Yanqiao Zhu, Xing Xie, Liang Wang, Tieniu Tan. AAAI 2019.
Neural Graph Collaborative Filtering. Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, Tat-Seng Chua. SIGIR 2019.
Graph Neural Networks for Social Recommendation.Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, Dawei Yin. WWW 2019.
KGAT: Knowledge Graph Attention Network for Recommendation. Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu, Tat-Seng Chua. KDD 2019.
Fi-GNN: Modeling Feature Interactions via Graph Neural Networks for CTR Prediction. Zekun Li, Zeyu Cui, Shu Wu, Xiaoyu Zhang, Liang Wang. WWW 2019.
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang. SIGIR 2020.
Revisiting Graph based Collaborative Filtering: A Linear Residual Graph Convolutional Network Approach. Lei Chen, Richang Hong, Kun Zhang, Meng Wang. AAAI 2020.
TAGNN: Target Attentive Graph Neural Networks for Session-based Recommendation. Feng Yu, Yanqiao Zhu, Qiang Liu, Shu Wu, Liang Wang, Tieniu Tan. SIGIR 2020.
Multi-behavior Recommendation with Graph Convolutional Networks. Jin, Bowen and Gao, Chen and He, Xiangnan and Jin, Depeng and Li, Yong. SIGIR 2020.


## Chemistry and Biology
提出消息传递框架(Message Passing Neural Networks, MPNN) 最经典的文章：Neural message passing for quantum chemistry，最早就应用于化学领域。作者在这篇论文将现有神经网络模型抽象出其共性并提出一种信息传递神经网络框架(Message Passing Neural Networks, MPNN)，同时利用 MPNN 框架在分子分类预测中取得了一个不错的成绩。

Graph u-nets. Gao Hongyang, Ji Shuiwang. international conference on machine learning 2019.
MoleculeNet: a benchmark for molecular machine learning. Wu Zhenqin, Ramsundar Bharath, Feinberg Evan N, Gomes Joseph, Geniesse Caleb, Pappu Aneesh S, Leswing Karl, Pande Vijay. Chemical science 2018.
An end-to-end deep learning architecture for graph classification. Zhang Muhan, Cui Zhicheng, Neumann Marion, Chen Yixin. ThirtySecond AAAI Conference on Artificial Intelligence 2018.
Hierarchical graph representation learning with differentiable pooling. Ying Rex, You Jiaxuan, Morris Christopher, Ren Xiang, Hamilton William L, Leskovec Jure. arXiv preprint arXiv:1806.08804 2018.
How powerful are graph neural networks?. Xu Keyulu, Hu Weihua, Leskovec Jure, Jegelka Stefanie. arXiv preprint arXiv:1810.00826 2018.
Graph classification using structural attention. Lee John Boaz, Rossi Ryan, Kong Xiangnan. Proceedings of the th ACM SIGKDD International Conference on Knowledge Discovery Data Mining 2018.
Neural message passing for quantum chemistry. Gilmer Justin, Schoenholz Samuel S, Riley Patrick F, Vinyals Oriol, Dahl George E. International conference on machine learning 2017.
Learning convolutional neural networks for graphs. Niepert Mathias, Ahmed Mohamed, Kutzkov Konstantin. International conference on machine learning 2016.
Deep convolutional networks on graph-structured data. Henaff Mikael, Bruna Joan, LeCun Yann. arXiv preprint arXiv:1506.05163 2015.
Convolutional networks on graphs for learning molecular fingerprints. Duvenaud David, Maclaurin Dougal, Aguilera-Iparraguirre Jorge, Gómez-Bombarelli Rafael, Hirzel Timothy, Aspuru-Guzik Alán, Adams Ryan P. arXiv preprint arXiv:1509.09292 2015.









---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*

