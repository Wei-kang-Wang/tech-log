---
layout: post
comments: false
title: "[论文]Short Literature Review"
date: 2022-06-06 01:09:00
tags: paper-reading
---

> This post is a summary of short conclusions of papers.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

### [Unsupervised Learning of Visual 3D Keypoints for Control](http://proceedings.mlr.press/v139/chen21b/chen21b.pdf)
[Code](https://github.com/buoyancy99/unsup-3d-keypoints) [Post](https://buoyancy99.github.io/unsup-3d-keypoints/)

*Boyuan Chen, Pieter Abbeel, Deepak Pathak*

*ICML 2021*

This paper proposed a method to directly learn 3D keypoint coordinates with only 2D images of the same scene from different view points as input in an unsupervised learning manner. The camera parameters are known. 

The whole process is as below:

Important hyperparameter: Keypoint number $$K$$.

**Step1** For each input 2D image $$I_n$$, use an encoder $$\Phi_n$$ to learn $$k$$ heatmaps $$C_n^k \in R^{S\times S}$$ and depth maps $$D_n^k \in R^{S\times S}$$. For each $$C_n^k$$, they used softmax to turn it into a probabilistic heatmap $$H_n^k$$ to represent the probability that keypoint $$k$$ will be at each position. Each depth map $$D_n^k$$ is a dense prediction of distance from camera plane for 3D keypoint k being at each position, i.e., each element in $$D_n^k$$ is the distance of keypoint $$k$$ to the camera plane.
After getting $$H_n^k$$ and $$D_n^k$$, they used the expectation across the whole heatmap to calculate the coordinate of keypoint $$k$$ in camera $$n$$ plane, $$\left[u_n^k, v_n^k, d_n^k\right]$$ as:

$$u_n^k = \frac{1}{S}\Sigma_{u,v}u H_n^k(u,v)$$, $$v_n^k = \frac{1}{S}\Sigma_{u,v}v H_n^k(u,v)$$, $$d_n^k = \Sigma_{u=1}^{S}\Sigma_{v=1}^{S}D_n^k(u,v) H_n^k(u,v)$$

Note that for each camera, we know the projection function to project the world coordinates into camera coordinates, denote it as $$\Omega_n$$, and the opposite projection as $$\Omega_n^{-1}$$. 

**Step2** Each camera will give a prediction of keypoint $$k$$ coordinates, the authors used some weighted techniques to get the final keypoint coodinates from these $$n$$ predictions, i.e., $$x^i = \frac{1}{n}\Sigma_{i=1}^n A_n^i x_n^i$$, $$y^i = \frac{1}{n}\Sigma_{i=1}^n A_n^i y_n^i$$, $$z^i = \frac{1}{n}\Sigma_{i=1}^n A_n^i z_n^i$$, where $$\left[x^i, y^i, z^i\right]$$ is the final coodinate of keypoint $$i$$ in the world coordination, and $$\left[x_n^i, y_n^i, z_n^i\right]$$ is calculated from $$\Omega_n^{-1}(\left[u_n^i, v_n^i, d_n^i\right])$$.
The calculation of weights $$A_n^k$$ is as:

$$A_n^k = \frac{exp{(\frac{1}{S^2}\Sigma_{p=1}\Sigma_{q=1}C_n^k(p,q))}}{\Sigma_{i=1}^K exp{(\frac{1}{S^2}\Sigma_{p=1}\Sigma_{q=1}C_n^i(p,q))}}$$

**Step3** This step is a reconstruction step. The authors believed that if the decoder can reconstruct the input image from the keypoints projected from the learned 3D keypoints $$\left[x^i, y^i, z^i\right]$$, the model learns the geometry of the scene. The input of the decoder is the scaled stacked Gaussian distribution matrix with $$u_n^i, v_n^i$$ as the center and $$I_2/d_n^i$$ as the coviriance matrix. Note that each input image $$I_n$$ has a decoder $$\phi_n$$.

![Model Structure]({{ '/assets/images/Unsupervised_3D.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Overview of our Keypoint3D algorithm. (a) For each camera view, a fully convolutional neural network encodes the input image into K heat maps and depth maps. (b) We then treat these heat maps as probabilities to compute expectation of spatial $$u,v$$ coordinates in camera plane. These expected values and the saptial variances are used to resample final $$u,v$$ keypoint coordinates which adds noise that prevents the decoder from cheating to hide the input information in the relative locations $$u,v$$ keypoints. We also take expectation of depth coordinate, $$d$$, using the same probability distribution. These $$\left[u; v; d\right]$$ coordinates are then unprojected into the world coordinate. (c) We take attention-weighted average of keypoint estimations from different camera views to get a single prediction in the world coordinate. (d) For decoding, we project predicted keypoints in world coordinate to $$\left[u; v; d\right]$$ in each camera plane. (e) Each keypoint coordinate is mapped to a gaussian map, where a 2D gussian is created with mean at $$\left[u, v\right]$$ and std inversely proportional to $$d$$. For each camera, gaussian maps are stacked together and passed into decoder to reconstruct observed pixels from the camera. (f) Together with reconstruction, we also jointly train a task MLP policy on top of predicted world coordinates via reinforcement learning.*


### [Weakly-Supervised Discovery of Geometry-Aware Representation for 3D Human Pose Estimation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Weakly-Supervised_Discovery_of_Geometry-Aware_Representation_for_3D_Human_Pose_Estimation_CVPR_2019_paper.pdf)
*Xipeng Chen, Kwan-Yee Lin, Wentao Liu, Chen Qian, Liang Lin*

*CVPR 2019*

This work proposed a method to learn 3D coordinates of human body joints in order to do human pose estimation. This model is based on skeleton extracted from the raw RGB images, not an End-to-end framework.

Hyperparameter: Number of Keypoints $$K$$.

**Step1** Inputs are source image $$I_s$$ and target image $$I_t$$, and the rotation matrix are known due to the parameters of cameras. First, they use existing skeleton algorithm to extract skeleton maps of $$I_s$$ and $$I_t$$.

**Step2** Instead of a traditional encoder-decoder framework, they use a novel view synthesis method, i.e., source image $$I_s$$ are encoded and combined with rotation matrix $$R_{s \rightarrow t}$$, target image $$I_t$$ are reconstructed from the decoder. The 3D keypoint coordinates are the output of the encoder, as a geometry-aware representation, as explained in the paper. They also design the bidirectional encoder-decoder framework, which hinges on two encoder-decoder networks with same architecture to perform view synthesis in the two directions simultaneously, i.e., from $$I_s$$ to $$I_t$$ and from $$I_t$$ to $$I_s$$. These two reconstructions will involve two losses.

**Step3** They believe that the 3D keypoints of these two images should be the same. There are two encoders and the outputs should be the same, thus involve a new loss.

![Model Structure1]({{ '/assets/images/weak-supervised.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1 The framework of learning a geometry representation for 3D human pose in a weakly-supervised manner. There are three main components. (a)Image-skeleton mapping module is used to obtain 2D skeleton maps from raw images. (b)View synthesis module is in a position to learn the geometry representation in latent space by generating skeleton map under viewpoint $$j$$ from skeleton map under viewpoint $$i$$. (c) Since there is no explicit constrain to facilitate the representation to be semantic, a representation consistency constrain mechanism is proposed to further refine the representation.*



### [Hand Keypoint Detection in Single Images using Multiview Bootstrapping](https://openaccess.thecvf.com/content_cvpr_2017/papers/Simon_Hand_Keypoint_Detection_CVPR_2017_paper.pdf)
*Tomas Simon, Hanbyul Joo, Iain Matthews, Yaser Sheikh*

*CVPR 2017*

This paper proposed a framework to learn 3D keypoints of hand. 

The input is a keypoint detector $$d_0$$ trained on a small labelled dataset $$T_0$$, a sequence of images $$\lbrace I_v^f, v=1,2,...,V, f=1,2,..,F \rbrace$$, with $$v$$ denote the camera view and $$f$$ denote the time frame.

Hyperparameter: Number of Keypoints $$K$$.

**Step1** First use $$d_0$$ to calculate the image coordinates (no depth) and confidence of each keypoint $$k$$ of $$I_v^f$$, denoted as $$x_v^{f,k}$$ and $$C_v^{f,k}$$. Then use the random sample consensus to pick inliers out of each set $$\lbrace (x_v^{k}, C_v^{k}) \rbrace$$ for each time frame $$f$$. Then the 3D wolrd coodinates are computed as:

$$X_v^{k} = argmin_X \Sigma_{v \in I_v^{k}} ||P_v(X)-x_v^k||^2_F$$

where $$I_v^k$$ is the inlier set, with $$X_f^k \in R^3$$ the 3D triangulated keypoint $$k$$ in frame $$f$$, and $$P_v(X) \in R^2$$ denotes projection of 3D point $$X$$ into camera view $$v$$. They use calibrated cameras, thus $$P_v$$ are known.

**Step2** Then they use a window through the time frame, and pick the frame with highest score. The score is defined as the sum of $$C_v^k$$, thus the frame that has the biggest confidence of all keypoints detection from all camera views.

**Step3** After picking this frame, they add the labelled images into the orignal training dataset and train a new keypoint detector, $$d_1$$. And so on.


### [Learning deep network for detecting 3D object keypoints and 6D poses](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhao_Learning_Deep_Network_for_Detecting_3D_Object_Keypoints_and_6D_CVPR_2020_paper.pdf)
*Wanqing Zhao, Shaobo Zhang, Ziyu Guan, Wei Zhao*

*CVPR 2020*









---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*

