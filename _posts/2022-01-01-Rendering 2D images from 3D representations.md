---
layout: post
comments: false
title: "Rendering 2D images from 3D representations"
date: 2020-01-01 01:09:00
tags: paper-reading
---

> 这个post将会介绍各种从不同的3D representations来render 2D图片的方法。


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---


* 在NeRF里，以Lego场景为例，数据的生成是利用blender构建一个scene，然后rotate以及translate整个场景，这个过程中相机不动，从而获取一系列views。而每个数据点包括这个view的图片，rotation matrix和translation构成的transformation matrix，而且数据里还包括了相机的focal length，而且每个view的图片的长宽相等。而在具体操作的时候，认为scene是不动的，从而相机角度和位置就会由rotation matrix以及translation matrix来决定，也就是说，认为scene的中心位于world coordiate的$$(0,0,0)$$处，而camera的world coordinate由translation matrix决定，角度由rotation matrix决定，从而每条从光心出发的射线就都确定下来了。

* 在[unsuper3d](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wu_Unsupervised_Learning_of_Probably_Symmetric_Deformable_3D_Objects_From_Images_CVPR_2020_paper.pdf)里，depth的预测代码如下：
```python
self.canon_depth_raw = self.netD(self.input_im).squeeze(1)  # BxHxW
self.canon_depth = self.canon_depth_raw - self.canon_depth_raw.view(b,-1).mean(1).view(b,1,1)
self.canon_depth = self.canon_depth.tanh()
self.canon_depth = self.depth_rescaler(self.canon_depth)
```
其中self.netD是一个网络，输入为RGB图片，也就是input_im，输出是大小等同于输入图片尺寸的，单通道的depth map，无任何值域限制。也就是上述第一行。第二行将每张预测的depth map减去这个map的均值。第三行加上一个tanh()函数，将depth的值域限制在-1到1之间。最后，self.depth_rescaler()函数如下：$$lambda d: (1+d)/2 \times max-depth + (1-d)/2 \times min-depth$$。默认的max-depth=1.1，min-depth=0.9，
```python
## clamp border depth
depth_border = torch.zeros(1,h,w-4).to(self.input_im.device)
depth_border = nn.functional.pad(depth_border, (2,2), mode='constant', value=1)
self.canon_depth = self.canon_depth*(1-depth_border) + depth_border *self.border_depth
self.canon_depth = torch.cat([self.canon_depth, self.canon_depth.flip(2)], 0)  # flip
```
之后还会对图片边缘的depth进行处理，先构造一个depth_border，中心区域是0，周围宽度为2的为1，然后取负，加上1，乘以原self.canon_depth，也就是中心区域保留不动，边缘置零，边缘区域还要再加上self.border_depth = 0.7 x max-depth + 0.3 x min-depth。最终结果就是，最后的self.canon_depth的中心区域和之前预测的depth一样，边缘区域值都是self.border_depth。

albedo的预测代码如下：
```python
self.canon_albedo = self.netA(self.input_im)  # Bx3xHxW
self.canon_albedo = torch.cat([self.canon_albedo, self.canon_albedo.flip(3)], 0)  # flip
```
self.canon_albedo是由self.netA给出的，其输入是RGB图片，输出是大小等同于输入图片的albedo map，通道数为3，值域收到了tanh()函数的限制，在-1到1之间。

confidence map和percentage map的预测代码如下：
```python
conf_sigma_l1, conf_sigma_percl = self.netC(self.input_im)  # Bx2xHxW
self.conf_sigma_l1 = conf_sigma_l1[:,:1]
self.conf_sigma_l1_flip = conf_sigma_l1[:,1:]
self.conf_sigma_percl = conf_sigma_percl[:,:1]
self.conf_sigma_percl_flip = conf_sigma_percl[:,1:]
```
self.netC的输入是RGB图片，输出是一个有两个值的tuple，第一个值是大小等同于输入图片尺寸的confidence map，第二个值是大小等同于输入图片尺寸除以4的percentage map，它们的通道数都是2，值域都受到了softplus()函数的约束，从而是一切正实数。

lighting的预测代码如下：
```python
canon_light = self.netL(self.input_im).repeat(2,1)  # 2Bx4
self.canon_light_a = self.amb_light_rescaler(canon_light[:,:1])  # ambience term
self.canon_light_b = self.diff_light_rescaler(canon_light[:,1:2])  # diffuse term
canon_light_dxy = canon_light[:,2:]
self.canon_light_d = torch.cat([canon_light_dxy, torch.ones(b*2,1).to(self.input_im.device)], 1)
self.canon_light_d = self.canon_light_d / ((self.canon_light_d**2).sum(1, keepdim=True))**0.5  # diffuse light direction
```
self.netL的输入是RGB图片，输出是长度为4的向量，值域收到tanh()函数的约束，位于-1到1之间。这个长度为4的向量的第一项是self.canon_light_a，第二项是sefl.canon_light_b，分别代表ambience term和diffuse term，都受到了rescaler函数的约束，rescaler的函数定义都是$$lambda x: (1+x) / 2 \times max + (1-x)/2 \times min$$，而且默认的max都1，min都是0。输出向量的后两项是canon_light_dxy，也就是light的direction，再在末尾接上一个1，变成$$\left[x,y,1 \right]$$，再处理一下，最后结果为$$\left[x, y,1 \right] / \sqrt{x^2 + y^2 + 1}$$。

shading的计算代码如下：
```python
self.canon_normal = self.renderer.get_normal_from_depth(self.canon_depth)
self.canon_diffuse_shading = (self.canon_normal * self.canon_light_d.view(-1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
canon_shading = self.canon_light_a.view(-1,1,1,1) + self.canon_light_b.view(-1,1,1,1)*self.canon_diffuse_shading
self.canon_im = (self.canon_albedo/2+0.5) * canon_shading *2-1
