---
layout: post
comments: false
title: "installation of various systems, environments and platforms"
date: 2022-05-01 01:09:00

---

> 这个post是有关各种系统、环境、平台的安装指南。


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---


## 1. Linux系统用Anaconda安装pytorch + CUDA + cuDNN

**1. 安装Anaconda**

从Anaconda官网下载安装所用的的sh文件（比如说Anaconda3-5.3.1-Linux-x86_64.sh），上传至Linux系统。

在Linux系统里打开bash，cd到要上述安装文件所在目录下进行操作 ，然后输入：

```shell
bash -u Anaconda3-5.3.1-Linux-x86_64.sh
```

安装成功后，输入：

```shell
source ~/.bashrc
```

到此Anaconda安装成功。可以通过下面代码检查是否安装成功：

```shell
python
```

显示下面的样子，则说明安装正确：

![pic1]({{ '/assets/images/install-1.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

输入下面指令退出：

```shell
quit()
```

**2. 了解操作系统内核、显卡、编译器等基本信息**

**2.1 查看自己操作系统的版本信息**

```shell
cat /etc/issue
```

或者

```shell
cat /etc/lsb-release
```

![pic2]({{ '/assets/images/install-2.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

![pic3]({{ '/assets/images/install-3.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

**2.2 查看服务器显卡信息**

```shell
lspci | grep -i nvidia
```
用于查看全部的显卡信息。

![pic4]({{ '/assets/images/install-4.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

如果已经安装了显卡驱动，则可以用下述指令查看：

```shell
nvidia-smi
```
![pic5]({{ '/assets/images/install-5.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

**2.3 查看安装的显卡驱动的信息**

```shell
cat /proc/driver/nvidia/version
```

![pic6]({{ '/assets/images/install-6.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

**3. 显卡驱动安装**

如果要安装显卡驱动，需要根据操作系统的版本进行选择。

**3.1 多版本的gcc和g++**

gcc和g++是很多驱动安装过程中需要使用的编译器，很多时候由于编译器版本的不对应会使得安装出现很多莫民奇妙的错误，根据经验，现在的CUDA 10.1的话，也可以使用的是4.8，因此最好选择4.8-5.4之间的版本比较好，

首先查看系统的gcc和g++版本：

```shell
gcc --version
g++ --version
```

![pic7]({{ '/assets/images/install-7.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

之后安装新版本的gcc以及g++：

```shell
sudo apt-get update 
sudo apt get update                                                        /*对需要的软件包等进行必要的更新。*/
sudo apt-get install gcc-4.9
sudo apt-get install g++-4.9                                               /*用于安装对应版本的gcc以及g++。注意自己需要的版本自己修改。*/
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 20
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 20

/*这两行用于将刚安装的gcc和g++类似于注册的操作加入到bin中，用于可选择操作。也就是说通过这个操作不断向系统注册新的gcc和g++版本。*/

update-alternatives --config gcc
update-alternatives --config g++                                           /*用于对版本进行选择。进入之后根据提示完成选择即可。如果权限不够加sudo */
```

![pic8]({{ '/assets/images/install-8.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

一般我们是使用4.8-5.4之间的版本编译器即可。如果还是出现错误的话，首选的操作应该是卸载显卡驱动重装显卡驱动，成功率最高。

**3.2 显卡驱动安装**

如果需要安装显卡驱动的话，需要先将旧版本的显卡驱动卸载：

```shell
sudo apt-get remove --purge nvidia*
```

此外，安装之前，需要先禁用一个东西。nouveau。

打开文件blacklist.conf：

```shell
sudo vim /etc/modprobe.d/blacklist.conf
```

在文件的最后面加入以下的内容:

```shell
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
```

检查操作是否成功：

```shell
lsmod | grep nouveau   /* 没显示即成功。 */
```

显卡驱动的安装比较简单，直接到官网进行对应的驱动的下载。

![pic9]({{ '/assets/images/install-9.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

![pic10]({{ '/assets/images/install-10.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

将下载好的显卡驱动上传到服务器进行安装。

![pic11]({{ '/assets/images/install-11.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

采用该命令进行驱动的安装：

```shell
sudo ./NVIDIA-Linux-x86_64-430.34.run
```

安装完成之后，可以采用以下命令进行检查：

```shell
nvidia-smi
```

**4. CUDA及cuDNN的安装**

CUDA是GPU进行计算的运算平台，根据需要安装对应版本的cuda。

这里需要注意的是，最好在安装显卡驱动的时候选择对应的cuda版本，然后在安装cuda的时候的版本保持一致（虽然高版本的驱动也可以兼容低版本的cuda）。

首先，查看自己的显卡驱动版本以及支持的最大CUDA版本：

```shell
nvidia-smi
```

![pic12]({{ '/assets/images/install-12.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

进入英伟达提供的[GPU驱动和CUDA对应关系](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

确认自己目前的GPU驱动是否符合自己要安装的CUDA版本，符合就进入下一步，不符合就安装更加高级的驱动。

**4.1 CUDA安装包下载**

[这是最新版本的CUDA下载界面](https://developer.nvidia.com/cuda-downloads)
[这是之前版本的CUDA下载界面](https://developer.nvidia.com/cuda-toolkit-archive)

以CUDA-10.2为例：

![pic13]({{ '/assets/images/install-13.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

利用下述代码下载CUDA并安装：

```shell
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run
```

也可以选择在浏览器中输入wget后面的网址，然后在本地进行下载再上传至服务器。

**4.2 cuDNN安装包下载**

通过[网址](https://developer.nvidia.com/rdp/cudnn-download)下载CUDNN，下载需要注册账号，登录以后，如下图选择合适的CUDA版本对应的CUDNN并选择CUDNN Library for Linux，开始下载，下载好以后将文件后缀名改为.tgz后上传至服务器。

![pic14]({{ '/assets/images/install-14.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

![pic15]({{ '/assets/images/install-15.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

注意Archived cuDNN Releases就可以找到过去版本的cudnn。


**4.3 CUDA的安装（非root用户安装）**

先给cuda可执行权限：

```shell
chmod +x cuda_10.2.89_440.33.01_linux.run
```

运行run文件:

```shell
sh cuda_10.2.89_440.33.01_linux.run
```

通过键盘方向键（↑，↓）和Enter键可以进行选择和进入（确定）。选择Continue并进入:

![pic16]({{ '/assets/images/install-16.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

输入accept进入:

![pic17]({{ '/assets/images/install-17.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

利用上下键与Enter勾选对话框，只安装CUDA Toolkit:

![pic18]({{ '/assets/images/install-18.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

选择Options并进入，然后我们需要修改Toolkit Options 、Library install path这两项的路径。

![pic19]({{ '/assets/images/install-19.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

修改Toolkit Options路径，选择Change Toolkit Install Path:

![pic20]({{ '/assets/images/install-20.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

将默认路径修改至个人目录下，点击Enter确认:

![pic21]({{ '/assets/images/install-21.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

将下面的选项取消选定，选择Done，确认退出:

![pic22]({{ '/assets/images/install-22.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

选择Library install path （Blank for system default）

![pic23]({{ '/assets/images/install-23.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

添加之前一样的路径并Enter确认退出

![pic24]({{ '/assets/images/install-24.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

选择Done返回上一层目录，修改路径完成，选择Install开始安装

![pic25]({{ '/assets/images/install-25.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

出现如下所示的安装信息则说明安装成功

![pic26]({{ '/assets/images/install-26.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

**4.4 修改环境变量**

输入vim ~/.bashrc进行环境变量的修改；
添加一下信息（每个人的路径是不同的，我的是/home/zhaoqc/cuda-10.2）并保存退出。

```shell
export CUDA_HOME=$CUDA_HOME:/home/zhaoqc/cuda-10.2
export PATH=$PATH:/home/zhaoqc/cuda-10.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zhaoqc/cuda-10.2/lib64
```

输入source ~/.bashrc

**4.5 安装cuDNN**

解压cuDNN，输入以下命令进行解压

```shell
tar -zxvf cudnn-10.2-linux-x64-v8.0.0.39.tgz
```

此时当前目录下回出现一个./cuda的文件夹

![pic27]({{ '/assets/images/install-27.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

复制文件到CUDA安装目录

```shell
cp cuda/include/cudnn.h ../cuda-10.2/include/
cp cuda/lib64/libcudnn* ../cuda-10.2/lib64/
```

修改权限，cudnn安装完成

```shell
chmod a+r ../cuda-10.2/include/cudnn.h ../cuda-10.2/lib64/libcudnn*
```

查看是否安装成功，输入nvcc -V

![pic28]({{ '/assets/images/install-28.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}


**5. 安装PyTorch**

现在Anaconda里构建一个虚拟环境：

```shell
conda create -n pytorch_env python=3.8
```

上述指令会生成一个叫pytorch_env的虚拟环境，相当于在Anaconda里开了个小房间，我们之后的包都可以在这个地方安装，即使有问题也不会影响其它的小房间以及Anaconda，而且可以很容易删除，注意上述指令还强调了我们用的是python3.8环境。

安装完成之后，输入conda activate pytorch_env进入虚拟环境。下图里左边的括号内的名字就是环境的名字，请确认切换到pytorch_env这个名字的环境后再继续安装pytorch。

![pic29]({{ '/assets/images/install-29.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

**6. 常用conda命令**

新建一个Pyhon版本为3.8的环境

```shell
conda create -n name_of_environment python=3.8
```

删除掉一个环境

```shell
conda remove -n name_of_environment –all
```

查看安装了的环境

```shell
conda info –envs
```

安装一个软件

```shell
conda install some_package
```

查看当前环境装了哪些软件

```shell
conda list
```

复制一个环境并重新命名（老环境不会删除）

```shell
conda create –name new_name –clone old_name
```

清理包缓存

```shell
conda clean -p
```

清理tar包缓存

```shell
conda clean -t
```

**7. 参考文献**
* https://zhuanlan.zhihu.com/p/198161777
* https://blog.csdn.net/kingfoulin/article/details/98872965
* https://panpili.com/?p=1166
* https://blog.csdn.net/weixin_44100850/article/details/103308367


## 2. Linux安装tmux

tmux是一个 terminal multiplexer（终端复用器），它可以启动一系列终端会话。

简单来说，安装tmux之前，一旦与服务器断开连接或者关闭xhell或其他shell终端，我们的服务器上运行的程序就会终止，而且输入的历史消息全部消失。因此如果我们希望整晚在服务器上跑代码，我们的电脑也要整晚一直连接着服务器。而安装了tmux之后，即使我们关闭了shell终端或者不幸与服务器断开连接，我们在服务器上的程序依然在运行。

**1. 有sudo权限的用户**

很简单，就一行：

```shell
sudo apt-get install tmux
```

**2. 无sudo权限的非root用户**

**2.1 下载**

下载tmux及其依赖软件。

```shell
wget -c https://github.com/tmux/tmux/releases/download/3.0a/tmux-3.0a.tar.gz
wget -c https://github.com/libevent/libevent/releases/download/release-2.1.11-stable/libevent-2.1.11-stable.tar.gz
wget -c https://ftp.gnu.org/gnu/ncurses/ncurses-6.2.tar.gz
```
**2.2 解压下载的包**

```shell
tar -xzvf tmux-3.0a.tar.gz
tar -xzvf libevent-2.1.11-stable.tar.gz
tar -xzvf ncurses-6.2.tar.gz
```

**2.3 分别源码安装，先安装两个依赖包**

libevent会安在 /tmux_depend/lib：

```shell
cd  libevent-2.1.11-stable
/* $HOME/tmux_depend是我的安装路径，大家可以修改 */
./configure --prefix=$HOME/tmux_depend --disable-shared
make && make install
```

ncurses会安在 /tmux_depend/include：

```shell
cd
cd  ncurses-6.2
./configure --prefix=$HOME/tmux_depend
make && make install
```

**2.4 安装tmux**

```shell
cd
cd  tmux-3.0a
./configure CFLAGS="-I$HOME/tmux_depend/include -I/$HOME/tmux_depend/include/ncurses" LDFLAGS="-L/$HOME/tmux_depend/lib -L/$HOME/tmux_depend/include/ncurses -L/$HOME/tmux_depend/include"
make
cp tmux  $HOME/tmux_depend/bin
```

**2.5 设置环境变量（此步骤建议手动添加到bashrc文件中）**

```shell
export PATH=$HOME/tmux_depend/bin:$PATH
source ~/.bashrc
```

到此即完成安装

**3. tmux常见指令以及用法**

1）新建会话，比如新创建一个会话以"ccc"命名

```shell
[root@Centos6 ~]# tmux new -s ccc
/*加上参数-d，表示在后台新建会话*/
root@bobo:~# tmux new -s shibo -d
root@bobo:~# tmux ls
shibo: 1 windows (created Tue Oct  2 19:22:32 2018) [135x35]
```
 
 
2）查看创建得所有会话

```shell
[root@Centos6 ~]# tmux ls
0: 1 windows (created Wed Aug 30 17:58:20 2017) [112x22](attached)    
/* 这里的attached表示该会话是当前会话 */
aaa: 2 windows (created Wed Aug 30 16:54:33 2017) [112x22]
ccc: 1 windows (created Wed Aug 30 17:01:05 2017) [112x22]
```
   
 
3）登录一个已知会话。即从终端环境进入会话。
 
第一个参数a也可以写成attach。后面的aaa是会话名称。

```shell
[root@Centos6 ~]# tmux a -t aaa 
```
 
4）退出会话不是关闭：
 
登到某一个会话后，先按键ctrl+b启动快捷键，再按d，这样就会退出该会话，但不会关闭会话。
 
如果直接ctrl + d，就会在退出会话的通话也关闭了该会话！
 
   
 
5）关闭会话（销毁会话）

```shell
[root@Centos6 ~]# tmux ls
aaa: 2 windows (created Wed Aug 30 16:54:33 2017) [112x22]
bbb: 1 windows (created Wed Aug 30 19:02:09 2017) [112x22]   
[root@Centos6 ~]# tmux kill-session -t bbb
[root@Centos6 ~]# tmux ls
aaa: 2 windows (created Wed Aug 30 16:54:33 2017) [112x22]
```  
 
6）重命名会话

```shell
[root@Centos6 ~]# tmux ls  
wangshibo: 1 windows (created Sun Sep 30 10:17:00 2018) [136x29] (attached)
[root@Centos6 ~]# tmux rename -t wangshibo kevin
[root@Centos6 ~]# tmux ls
kevin: 1 windows (created Sun Sep 30 10:17:00 2018) [136x29] (attached)
```

**4. 参考文献**
* https://zhuanlan.zhihu.com/p/155662601



## 3. git的使用

**1. 介绍Git**

git是世界上最先进的分布式版本控制系统。

举个例子，如果曾用Microsoft word写文章，可能会有如下经历：想对当前文章进行修改，但又不确定是否应该修改，只能修改后另存为一个文件。而这样的次数多了， 就会出现很多个类似的文件。而过了一段时间，想找到某种特定修改情况下的文件，还得打开每个文件来寻找，十分复杂。而且会存下来很多相似但有不同的文件，使得文件管理十分混乱。如果还有其他人想要共同修改文件，那就会使得情况更加复杂。

如果有一个系统，能够自动记录每次文件的改动，还可以让其他人也同时进行编辑，而不需要自己手动管理储存一系列类似的改动的文件，也不需要在很多人之间传文件，那就会变得很方便。如果想查看之前的版本，只需要在文件改动栏里看一眼就行，这就会十分方便。

这样的系统就叫版本控制系统，而git是其中做的最好的。

Linus在发明了Linux之后，全世界的程序员都在对Linux进行贡献，而之前Linus是手动合并这些代码的。但到了后来过于繁杂，但Linus并不喜欢CVS，SVN这种集中式的版本控制系统，所以选择了BitKeeper。而之后BitKeeper不再免费提供使用，于是Linus本人用C写了一个分布式版本控制系统，就是git。

>集中式和分布式版本控制系统的区别
>集中式版本控制系统，其版本库是存放在中央服务器的，而干活的时候，用的都是自己的电脑，所以要先从中央服务器取得最新的版本，然后再在自己的电脑上进行改动。改动完毕，再将文件推送给中央服务器。
>集中式版本控制系统的最大的问题就是必须得联网才能工作，如果在局域网内还好，带宽和速度都可很快，但在互联网上，就会收到网速等因素的影响。

![1]({{ '/assets/images/GIT-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

>而分布式版本控制系统根本没有中央服务器，每个人的电脑都是一个完整的版本库，这样每个人修改文件的时候就不需要联网了，因为版本库就在自己的本地电脑上。既然每个人都有一个完整的版本库，那如何进行多人协作呢？比如说A在本地电脑上修改了文件file，B也在其本地电脑上修改了文件file，他们只需要将各自的修改推送给对方，就可以互相查看对方的修改了。

![2]({{ '/assets/images/GIT-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

>和集中式版本控制系统相比，分布式版本控制系统的安全性要高很多，因为每个人电脑里都有完整的代码库，如果某个代码库坏了，去其他人电脑拷贝一份就行。而集中式版本控制系统的中央服务器出现问题，那所有人就无法工作。
>在实际使用分布式版本控制系统的时候，其实很少在两人之间的电脑上推送版本库的修改，因为可能这两台电脑并不在一个局域网内，互相访问不了，等情况。因此，分布式版本控制系统通常也有一台充当”中央服务器“的电脑，但这个服务器的作用仅仅是用来方便交换大家的修改，没有的话也不要紧，只是交换修改不方便。
>Git的优势不仅是不需要联网这么简单，Git还有很强大的分支管理器，是SVN这种集中式版本控制系统所没有的。
>CVS是最早的集中式版本控制系统，但是后来被更稳定且开源的SVN取代了。
>分布式版本控制系统除了Git和BitKeeper以外，还有Mercurial和Bazaar，但Git仍然是最流行最好用的。


**2. 安装Git**

最早Git是在Linux上开发的，很长一段时间，Git也只能在Linux上运行。但现在Git已经可以在Linux、Mac和Windows上正常运行了。

**2.1 在Linux上安装Git**

先输入git，看看系统有没有安装Git：

```shell
$ git
The program 'git' is currently not installed. You can install it by calling:
sudo apt-get install git
```

如果是Debian和Ubuntu系统，直接使用：

```shell
sudo apt-get install git
```

对于其他linux版本，则直接从Git官网下载源码，解压后使用源码安装。

**2.2 在MacOS上安装Git**

Xcode已经集成了Git，不需要额外安装、


**2.3 在Windows上安装Git**

在Windows上安装Git，直接在Git官网下载安装程序，然后默认选项安装即可。

安装完成后，在菜单找到Git -> Git Bash，会有类似于命令行窗口的界面，说明安装完成。

![3]({{ '/assets/images/GIT-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

在安装完成后，还需设置，在命令行输入：

```shell
$ git config --global user.name "Your name"
$ git config --global user.mail "email@example.com"
```

因为Git是分布式版本控制系统，所以每个机器都需要有标签：名字和Email地址。

注意上述git config命令里用了--global参数，使用这个参数表明这台机器上所有的Git仓库都会使用这个名字和Email。当然也可以对某个仓库指定不同的用户名和Email地址。


**3. 创建版本库**

版本库又名仓库，repository，可以简单理解为一个目录，这个目录里所有的文件都可以被Git管理起来，每个文件的修改、删除，Git都能跟踪，以便任何时候都可以追踪历史版本，或者在将来某个时刻还原历史版本。

创建一个版本库非常简单。首先，选择一个合适的地方，打开git bash，创建一个空目录：

```shell
$ mkdir learngit
$ cd learngit
$ pwd             # pwd命令用于显示当前目录
```

第二步，通过git init命令把这个目录变成Git可以管理的仓库：

```shell
$ git init
Initialized empty Git repository in /Users/Alan/learngit/.git/
```

上述/Users/Alan/learngit/是仓库的位置，名字取决于本地电脑。

上述操作就将仓库建好了，而且告诉用户是一个空的仓库，empty Git repository。之后就可以发现，目录下会多一个.git文件夹，这个文件夹是Git来跟踪管理版本库的，所以不要轻易手动修改这个目录下的文件。

而如果不可见这个.git目录，那是因为这个目录是默认隐层的，使用ls -ah命令就可以看见。

除了在空目录下创建Git仓库，也可以在已有文件的目录下创建Git仓库。

**3.1 将文件添加到版本库**

所有的版本控制系统，其实只能跟踪文本文件的改动，比如TXT文件，网页，所有的程序代码等，Git也不例外。版本控制系统可以告诉用户每次的改动，比如在某文件第几行添加了什么单词等。而图片、视频这种二进制文件，虽然也能由版本控制系统管理，但无法跟踪文件的变化，只能知道图片从100kb改成120kb这种信息，但不知道具体改动了什么。

Microsoft word的格式是二进制文件，所以版本控制系统是无法跟踪word文件的改动的。所以如果要真正使用版本控制系统，就要以纯文本的方式编写文件。

>文本文件和二进制文件的区别：
>识别算法把字节码0到255分为三大类：
>（1）白名单：文本型字节码，包括9(TAB)、10(LF)、32(SPACE)到255
>（2）灰名单：可容忍的字节码，包括7(BEL)、8(BS)、11(VT)、12(FF)、26(SUB)和27(ESC)
>（3）黑名单：不能容忍的、非文本型字节码，包括0(NULL)到6，14到31
>如果文件至少有一个字节属于白名单，且没有字节属于黑名单，则文件被认为是纯文本的，否则就是二进制文件。文件为空，自动识别为二进制文件。

文本文件是有编码的，建议使用标准的UTF-8编码。

>如果使用Windows自带的记事本编辑文本文件，会在每个文件开头加上0xefbbbf的字符，会导致一些不可预见的错误，所以建议使用visual studio code来替代记事本编辑文本文件。

举个例子，编写一个readme.txt文件（两行），内容如下：

```shell
Git is a version control system
Git is free software
```

其需要放在之前创建的learngit目录或者子目录下，因为这样才能被Git找到。

将文件放到Git仓库只需要两步：

第一步，在git bash里用命令git add告诉Git，将文件添加到仓库：

```shell
$ git add readme.txt
```

执行上面的命令，没有任何显示（因为Unix哲学就是没有显示说明正确）。

第二步，在git bash里用命令git commit告诉Git，将文件提交到仓库：

```shell
$ git commit -m "wrote a readme file"
[master (root-commit) eaadf4e] wrote a readme file
 1 file changed, 2 insertions(+)
 create mode 100644 readme.txt
```

git commit命令，-m后面输入的是本次提交的说明，可以输入任意内容，但最好是有意义的，这样就可以从历史记录里方便的找到改动记录。

如果不想添加任何说明，使用git commit -m是不行的，系统还是会让你输入说明，可以使用下面的方法避免输入说明：

```shell
$ git commit -a --allow-empty-message -m ""
```

git commit命令执行成功之后会显示，1 file changed：一个文件被改动（新添加的readme.txt文件），2 insertions：插入两行内容，因为创建的txt文件有两行。

而为什么Git添加文件需要add和commit两步呢？因为commit可以一次添加很多文件，所以可以多次add不同的文件，比如：

```shell
$ git add file1.txt
$ git add file2.txt file3.txt
$ git commit -m "add 3 files"
```

**4. 回溯**

我们已经成功添加并提交了一个readme.txt文件，现在继续工作，修改readme.txt文件。改为：

```shell
Git is a distributed version control system
Git is free software
```

现在，在git bash里运行git status查看结果：

```shell
$ git status
On branch master
Changes not staged for commit:
    (use "git add <file>..." to update with what will be committed)
    (use "git checkout -- <file>..." to discard changes in working directory)


     modified:    readme.txt


no changes added to commit (use "git add" and/or "git commit -a")
```

git status可以让我们时刻掌握仓库当前的状态，上面的命令的输出告诉我们，readme.txt被修改了，但还没有准备提交。

虽然Git告诉我们readme.txt被修改了，但如果还想查看具体修改了什么内容就更好。需要用git diff这个命令来查看：

```shell
$ git diff readme.txt
diff --git a/readme.txt b/readme.txt
index 46d49bf..9247db6 100644
--- a/readme.txt
+++ b/readme.txt
@@ -1,2 +1,2 @@
-Git is a version control system
+Git is a distributed version control system
 Git is free software
```

git diff就是查看文件前后的difference，显示的格式是Unix通用的diff格式。可以从上面的输出看到，在第一行添加啦一个distributed单词。

在知道了readme.txt作了什么修改之后，就可以将其提交到仓库中去了。提交修改和提交新文件是一样的两个步骤，第一步是先git add：

```shell
$ git add readme.txt
```

同样没有任何输出。在执行第二步git commit之前，再运行git status看看当前仓库的状态：

```shell
$ git status
On branch master
Changes to be commited:
    (use "git reset HEAD <file>..." to unstage)
    
     modified:    readme.txt
```

git status的输出告诉我们，将要被提交的修改包括readme.txt，下一步就可以放心提交了：

```shell
$ git commit -m "add distributed"
[master e475afc] add distributed
 1 file changed, 1 insertion(+), 1 deletion(-)
```

提交后，再用git status命令查看仓库当前状态：

```shell
$ git status
On branch master
nothing to commit, working tree clean
```

Git告诉我们当前没有需要提交的修改，而且，工作目录是干净的（working tree clean）。


**4.1 版本回退**

现在我们已经知道了如何修改文件，并将修改后的文件提交到Git版本库。现在我们再修改一次readme.txt，改成如下内容：

```shell
Git is a distributed version control system
Git is a free software under the GPL
```

然后提交：

```shell
$ git add readme.txt
$ git commit -m "append GPL"
[master 1094adb] append GPL
 1 file changed, 1 insertion(+), 1 deletion(-)
```

像上述操作这样，不断对文件进行修改，然后不断提交到版本库里，就像玩RPG游戏，每通过一关就会自动将游戏状态存盘，如果某一关没过去，还可以选择读取前一关的状态。有些时候，还可以手动存盘，以便失败的时候从最近的地方重新开始。Git也是一样，每当你觉得文件修改到一定程度的时候，就可以保存一个快照，这个快照再Git中被称为commit。一旦把文件改乱了，或者误删了文件，还可以从最近的一个commit恢复，然后继续工作，而不是将之前的工作全部丢失。

readme.txt文件已经有三个版本被提交到Git仓库里了：

版本1：wrote a readme file

```shell
Git is a version control system
Git is free software
```

版本2：add distributed

```shell
Git is a distributed version control system
Git is free software
```

版本3：append GPL

```shell
Git is a distributed version control system
Git is free software distributed under the GPL
```

在实际工作里，记住几千行的代码每次都修改了什么内容是不现实的，而版本系统就是帮助我们记住这些修改的内容。版本系统里有命令可以告诉我们历史记录，在Git里，使用git log命令查看：

```shell
$ git log
commit 1094adb7b9b3807259d8cb349e7df1d4d6477073 (HEAD -> master)
Author: Alan Turing <mail@example.com>
Date: Fri May 18 21:06:15 2018 +0800


    append GPL
    
commit e475afc93c209a690c39c13a46716e8fa000c366
Author: Alan Turing <mail@example.com>
Date: Fri May 18 21:03:36 2018 +0800


    add distributed

commit eaadf4e385e865d25c48e7ca9c8395c3f7dfaef0
Author: Alan Turing <mail@example.com>
Date: Fri May 18 20:59:18 2018 +0800


    wrote a readme file
```

git log命令显示从最近到最远的提交日志。我们可以看到3次提交，最近的一次是append GPL，上一次是add distributed，最早的一次是wrote a readme file。

如果嫌输出信息太多，可以加上--pretty=oneline参数：

```shell
$ git log --pretty=oneline
1094adb7b9b3807259d8cb349e7df1d4d6477073 (HEAD -> master) append GPL
e475afc93c209a690c39c13a46716e8fa000c366 add distributed
eaadf4e385e865d25c48e7ca9c8395c3f7dfaef0 wrote a readme file
```

上述一大串字符，比如1094adb7b9b3807259d8cb349e7df1d4d6477073，这是commit id（版本号）。和SVN不一样，Git的commit id不是1，2，3这种递增的数字，而是一个SHA1计算出来的很大的数字，用十六进制表示。为什么commit id需要使用这种方式呢？因为其是分布式的版本控制系统，如果有多人在一个版本库里工作，大家都用1，2，3来命名，就会起冲突。

每提交一个版本，Git就会把它们自动串成一条时间线。如果使用可视化工作查看Git历史，就可以更清楚地看到提交历史的时间线：

![4]({{ '/assets/images/GIT-4.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

现在我们将readme.txt回退到上一个版本，也就是add distributed的版本，应该怎么做呢？

首先，Git必须知道当前版本是哪个版本，在Git中用HEAD表示当前版本，也就是最新提交的1094adb7b9b3807259d8cb349e7df1d4d6477073，上一个版本就是HEAD^，上上个就是HEAD^^，以此类推。而往前100个版本不能重复100个^，所以写成HEAD~100。

现在，我们要把当前版本append GPL回退到上一个版本add distributed，就可以使用git reset命令：

```shell
$ git reset --hard HEAD^
HEAD is now at e475afc add distributed
```

现在查看readme.txt的内容是不是版本add distributed（注意，我的电脑是Mac的，所以使用cat命令查看，Windows没有，需要使用别的方法）

```shell
$ cat readme.txt
Git is a distributed version control system
Git is free software
```

用git log来看看现在版本库的状态：

```shell
$ git log
commit e475afc93c209a690c39c13a46716e8fa000c366 (HEAD -> master)
Author: Alan Turing <mail@example.com>
Date: Fri May 18 21:03:36 2018 +0800


    add distributed

commit eaadf4e385e865d25c48e7ca9c8395c3f7dfaef0
Author: Alan Turing <mail@example.com>
Date: Fri May 18 20:59:18 2018 +0800


    wrote a readme file
```

最新的那个版本append GPL已经看不到了。

注意，现在我们的原来的readme.txt已经回到了上一个版本，也就是说里面的内容并没有append GPL这些句子了，也就是说真实的文件随之改变了。

如果想要再找到这个append GPL版本，只要目前的git bash窗口还没关闭，就可以找到那个append GPL版本对应的commit id，从而指定回到某个版本：

```shell
$ git reset --hard 1094a
HEAD is now at 83b0afe append GPL
```

注意到上述代码里，我们并没有完整打出append GPLde commit id，只是给了前几位，因为Git会自动去找。但如果给的太少，他可能会找到好几个版本，就无法确定是哪一个了。

我们现在再来看readme.txt的内容：

```shell
$ cat readme.txt
Git is a distributed version control system
Git is free software distributed under the GPL
```

Git的回退速度非常快，因为在Git内部有个指向当前版本的HEAD指针，当回退版本的之后，Git仅仅是把HEAD从指向append GPL改到指向add distributed。然后顺便将工作区的文件更新。所以HEAD指向哪个版本号，就会将当前版本定位在哪。

而如果现在回退到了某个版本，关掉了git bash，但之后又想回到新版本怎么办呢？因为回到某个版本就需要该版本的commit id。

Git提供了一个命令，git reflog来记录每一次git bash里的命令：

```shell
$ git reflog
e475afc HEAD@{1}: reset: moving to HEAD^
1094adb (HEAD -> master) HEAD@{2}: commit append GPL
e475afc HEAD@{3}: commit: add distributed
eaaaf4e HEAD@{4}: commit: (initial): wrote a readme file
```

从而可以获知append GPL的commit id的前几位是1094adb，就可以使用git reset回到那个版本了。

**4.2 工作区和暂存区**

Git和其它版本控制系统如SVN的一个不同之处就在于有暂存区的概念。

*工作区（working directory）*

就是在本地电脑里能看到的目录，比如learngit文件夹就是一个工作区。

*版本库（repository）*

工作区有一个隐藏目录.git，这个不算工作区，而是Git的版本库。
Git的版本库里存了很多东西，其中最重要的就是称为stage（或者叫index）的暂存区，还有Git自动创建的第一个分支master，以及指向master的一个指针HEAD。

![5]({{ '/assets/images/GIT-5.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

前面所述，将文件往Git版本库里添加的时候，是分两步进行的：

第一步是用git add把文件添进去，实际上就是把文件修改添加到暂存区；
第二步是用git commit提交修改，实际上就是把暂存区的所有内容提交到当前分支。

因为我们创建Git版本库的时候，Git自动为我们创建了唯一一个master分支，所以现在git commit就是往master分支上提交更改。

从而上述过程可以理解为，需要提交的文件修改都放在暂存区，然后一次性提交暂存区所有的修改。

我们对readme.txt加上一行内容：

```shell
Git is a distributed version control system
Git is free software distributed under the GPL
Git has a mutable index called stage
```

然后，在工作区新增一个LICENSE文本文件（内容随意）

先用git status查看一下状态：

```shell
$ git status
On branch master
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   readme.txt

Untracked files:
  (use "git add <file>..." to include in what will be committed)

	LICENSE

no changes added to commit (use "git add" and/or "git commit -a")
```

Git告诉我们，readme.txt被修改了，而LICENSE从未被添加过，所以状态是untracked的。

现在，使用两次命令git add，把readme.txt和LICENSE都添加之后，用git status再看一下：

```shell
$ git status
On branch master
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

	new file:   LICENSE
	modified:   readme.txt
```

现在，暂存区的状态就变成如下的样子了：

![6]({{ '/assets/images/GIT-6.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

所以，git add命令实际上就是把要提交的所有修改放到暂存区（stage），然后，执行git commit就可以一次性将暂存区所有的修改都提交到分支。

```shell
$ git commit -m "understand how stage works"
[master e43a48b] understand how stage works
 2 files changed, 2 insertions(+)
 create mode 100644 LICENSE
```

一旦提交后，如果没有再对工作区做什么修改，那么工作区就是干净的：

```shell
$ git status
On branch master
nothing to commit, working tree clean
```

现在版本库变成了这样，暂存区就没有任何内容了：

![7]({{ '/assets/images/GIT-7.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

**4.3 管理修改

为什么Git比其它版本控制系统设计的更加优秀呢？因为Git跟踪并管理的是修改，而不是文件。

什么是修改？比如新增了一行，这是修改，删除了一行，也这是修改，更改了某些字符，也是修改。甚至创建了一个新文件，这也是修改。

首先，我们对readme.txt做一个修改，比如加一行内容：

```shell
$ cat readme.txt
Git is a distributed version control system
Git is free software distributed under the GPL
Git has a mutable index called stage
Git tracks changes
```

然后，将文件添加到暂存区：

```shell
$ git add readme.txt
$ git status
On branch master
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)
      modified:   readme.txt
```

然后，再修改readme.txt：

```shell
$ cat readme.txt
Git is a distributed version control system
Git is free software distributed under the GPL
Git has a mutable index called stage
Git tracks changes of files
```

提交：

```shell
$ git commit -m "git tracks changes"
[master 519219b] git tracks changes
 1 file changed, 1 insertion(+)
```

提交后，再查看状态：

```shell
$ git status
On branch master
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   readme.txt

no changes added to commit (use "git add" and/or "git commit -a")
```

会发现，并不是之前在commit之后工作区就clean的状态，而是还有没提交的修改。

因为我们上述操作是：第一次修改 -> git add -> 第二次修改 -> git commit

Git管理的是修改，使用git add之后，第一次修改被放入暂存区，准备提交，但是工作区的第二次修改并没有被放暂存区，git commit只负责将暂存区的修改提交到仓库，也就是第一次修改提交，第二次修改没提交。

可以用git diff HEAD --readme.txt来查看工作区和版本库里最新版本的区别：

```shell
$ git diff HEAD -- readme.txt 
diff --git a/readme.txt b/readme.txt
index 76d770f..a9c5755 100644
--- a/readme.txt
+++ b/readme.txt
@@ -1,4 +1,4 @@
 Git is a distributed version control system.
 Git is free software distributed under the GPL.
 Git has a mutable index called stage.
-Git tracks changes.
+Git tracks changes of files.
```

可以看出第二次修改确实没有被提交。

**4.4 撤销修改

假设我们在readme.txt里又添加了一行：

```shell
$ cat readme.txt
Git is a distributed version control system
Git is free software distributed under the GPL
Git has a mutable index called stage
Git tracks changes of files
My stupid boss still prefers SVN
```

在准备使用git add提交之前，发现想要删除最后一行，那直接删除就可以了，因为其还在工作区。用git status查看一下：

```shell
$ git status
On branch master
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   readme.txt

no changes added to commit (use "git add" and/or "git commit -a")
```

可以发现，Git说，git checkout --file可以丢弃工作区的修改：
 
```shell
$ git checkout -- readme.txt
```

命令git checkout -- readme.txt的意思就是，把readme.txt文件在工作区的修改全部撤销，这里有两种情况：

一种是readme.txt自修改之后还没有放到暂存区，现在，撤销修改就回到和版本库一模一样的状态
第二种是readme.txt已经添加到了暂存区，又做了修改，现在，撤销修改就是回到添加到暂存区之后的状态

总之，就是让文件回到最近一次git commit或者git add时的状态。

在做了git checkout -- readme.txt操作之后的readme.txt文件就是：

```shell
$ cat readme.txt
Git is a distributed version control system.
Git is free software distributed under the GPL.
Git has a mutable index called stage.
Git tracks changes of files.
```

果然回到了之前。

注意，git checkout -- file里的--很重要，如果没有的话，就变成了切换到另一个分支的命令。

如果，我们在工作区修改了文件，还git add到了暂存区了：

```shell
$ cat readme.txt
Git is a distributed version control system.
Git is free software distributed under the GPL.
Git has a mutable index called stage.
Git tracks changes of files.
My stupid boss still prefers SVN.

$ git add readme.txt
```

但是还没有执行git commit。用git status查看，修应只是放到了暂存区，还没有提交：

```shell
$ git status
On branch master
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

	modified:   readme.txt
```

Git同样告诉我们，使用命令git reset HEAD file就可以把暂存区的修改撤销掉（unstage），重新放回工作区：

```shell
$ git reset HEAD readme.txt
Unstaged changed after reset:
M    readme.txt
```

git reset命令既可以回退版本，也可以把暂存区的修改回退到工作区。当我们使用HEAD时候，表示最新的版本。

再用git status查看，可以发现现在暂存区是干净的，工作区有修改：

```shell
$ git status
On branch master
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   readme.txt
```

再使用git checkout -- file来丢弃工作区的修改：

```shell
$ git chekcout -- readme.txt

$ git status
On branch master
nothing to commit, working tree clear
```

假设，现在在工作区修改了文件，用git add添加到了暂存区，还用git commit推送到了版本库，就可以使用之前的版本回退那一节的方法，回退到之前的版本。不过如果已经将本地的版本库推送到远程，那就覆水难收了。

**4.5 删除文件

在Git中，删除也是一个修改操作。

我们先添加一个test.txt到Git并提交：

```shell
$ git add test.txt

$ git commit -m "add test.txt"
[master b84166e] add test.txt
 1 file changed, 1 insertion(+)
 create mode 100644 test.txt
```

在本地工作区，如果要删除文件，直接删了，或者用rm命令删了：

```shell
$ rm test.py
```

这个时候，Git知道你删除了文件，因此，工作区和版本库就不一致了，git status命令会告诉你哪些文件被删除了：

```shell
$ git status
On branch master
Changes not staged for commit:
  (use "git add/rm <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	deleted:    test.txt

no changes added to commit (use "git add" and/or "git commit -a")
```

现在有两个选择，一是确实要从版本库里删除该文件，那就使用git rm删除，并且git commit：

```shell
$ git rm test.txt
rm 'test.txt'

$ git commit -m "remove test.txt"
[master d46f35e] remove test.txt
 1 file changed, 1 deletion(-)
 delete mode 100644 test.txt
```

现在，文件就从版本库中被删除了。

另一种情况是本地工作区删错了，因为版本库里还有这个文件，所以可以轻松的将文件恢复到最新版本：

```shell
$ git checkout -- test.txt
```

git checkout实际上是用版本库里的版本替换掉工作区的版本，无论工作区是修改还是删除，都可以一键还原。

**5. 远程仓库

我们现在已经掌握了如何在Git仓库里对一个文件进行“时光穿梭”，再也不用担心文件备份或者丢失的问题了。

但是其实这些功能SVN也有，并不是Git独特的地方。如果只是在一个仓库里管理历史文件，那么Git和SVN确实没有区别。但本章介绍的Git的一个重要功能却是Git独有的：远程仓库。

Git是分布式版本控制系统，同一个Git仓库，可以分布到不同的机器上。怎么分布呢？最早，只有一台机器有一个原始版本库，此后，别的机器可以“克隆”这个原始版本库，而且每台机器的版本库其实都是一样的，没有主次之分。

其实一台电脑也可以克隆多个版本库，只要不在一个目录下就行。但这么做并没有太大的现实意义。

实际情况往往是，找一台电脑充当服务器的角色，其他每个人都从这个“服务器”仓库克隆一份到自己的电脑上，并且各自把各自的提交推送到服务器仓库里，也从服务器仓库中拉去别人的提交。

完全可以免费自己搭建一个运行Git的服务器，但是会有些繁琐。好在Github提供了Git仓库托管服务，所以只需要注册一个Github账号，就可以免费获得Git远程仓库。

在注册了Github帐号之后，由于本地Git仓库和Github仓库之间的传输是通过SSH加密的，所以需要一些设置：

第一步，创建SSH Key。在用户主目录下，看看有没有.ssh目录，如果有，再看这个目录下有没有id_rsa和id_rsa.pub这两个文件，如果有了，可以直接跳到下一步，如果没有，打开git bash，创建SSH Key：

```shell
$ ssh-keygen -t rsa -C "youremail@example.com"
```

这个邮箱是注册GitHub的邮箱。

如果一切顺利，可以在用户主目录里找到.ssh目录，里面有id_rsa和id_rsa.pub两个文件，这两个就是SSH Key的密钥对，id_rsa是私钥，不可泄漏，id_rsa.pub是公钥，可以自由暴露。

第二步，登录Github，打开account settings，SSH Key界面：

然后，点Add SSH Key，填上任意的title，在Key文本框里粘贴id_rsa.pub的内容：

![8]({{ '/assets/images/GIT-8.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

点Add Key，就可以看到已经添加的Key：

![9]({{ '/assets/images/GIT-9.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

而为什么Github需要SSH Key呢？因为GitHub需要识别出你推送的提交确实是你推送的，而不是别人冒充的，Git支持SSH协议，所以Github只要知道了你的公钥，就可以确认只有你自己才能推送。

Github允许你添加多个Key。假定你有若干台电脑，只要把每台电脑的Key都添加到Github，就可以在每台电脑上往Github推送了。

在Github上免费托管的Git仓库，任何人都可以看到（但只有自己才能修改）。所以不要放入敏感信息。

如果不想别人看到Git库，有两个办法，一个是交钱让Github把公开的仓库变成私有的，这样其他人就不可读了。另一个办法是自己搭建Git服务器，这一般是公司的做法。

**5.1 添加远程库

现在的情形是，你已经在本地创建了一个Git仓库后，又想在GitHub创建一个Git仓库，并且让这两个Git仓库远程同步，这样Github上的仓库既可以作为备份，又可以让其他人通过该仓库来协作，一举多得。

首先，登录GitHub，然后在右上角点击create a new repo，创建一个新的仓库：

![10]({{ '/assets/images/GIT-10.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

在repository name填入项目的名字，比如learngit，其他保持默认值，点击create repository，就成功创建了一个新的Git仓库：

![11]({{ '/assets/images/GIT-11.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

目前，在GitHub上这个仓库还是空的，GitHub告诉我们，可以从这个仓库克隆出新的仓库，也可以把一个已有的本地仓库与之关联，然后，把本地仓库的内容推送到GitHub仓库。

现在，根据GitHub的提示，在本地仓库下用git bash运行以下命令：

```shell
$ git remote add origin git@github.com:alanturing/learngit.git
```

上面的alanturing是GitHub的用户名，替换成自己的，learngit是GitHub里这个仓库的名字。必须将上述用户名替换成自己的GitHub用户名，否则就会关联到其他人的GitHub仓库里，虽然关联没问题，但是因为其他人的GitHub并没有你的SSH Key的公钥，是推送不上去的。

添加之后，远程库的名字就是origin，这是Git默认的叫法，也可以改成别的。

下一步，就可以把本地库的所有内容推送到远程库上：

```shell
$ git push -u origin master
Counting objects: 20, done
Delta compression using up to 4 threads
Compressing objects: 100%(15/15), done
Writing objects: 100%(20/20), 1.64KiB | 560.00 KiB/s, done.
Total 20 (delta 5), reused 0 (delta 0)
remote: Resolving deltas: 100% (5/5), done.
To github.com:alanturing/learngit.git
 * [new branch]    master -> master
Branch 'master' set up to track remote branch 'master' from 'origin'
```

把本地内容推送到远程，用git push命令，实际上是把当前分支master推送到远程。

由于远程库是空的，第一次推送master分支时，加上了-u参数，Git不但会把本地的master分支内容推送到远程新的master分支，还会把本地的master分支和远程的master分支关联起来，在以后的推送或者拉取时就可以简化命令。

推送成功之后，就可以在GitHub界面看到远程库的内容已经和本地一模一样：

![12]({{ '/assets/images/GIT-12.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

从现在起，只要本地做了提交，就可以通过命令：

```shell
$ git push origin master
```

把本地master分支的最新修改推送至Github，现在就拥有了真正的分布式版本库了。

>SSH警告
>当第一次使用Git的clone或者push命令连接GitHub时，会有一个警告：
```shell
The authenticity of host 'github.com (xx.xx.xx.xx)' can't be established.
RSA key fingerprint is xx.xx.xx.xx.xx.
Are you sure you want to continue connecting (yes/no)?
```
>这是因为Git使用SSH连接，而SSH连接在第一次验证GitHub服务器的Key时，需要你确认GitHub的Key的指纹信息是否真的来自于GitHub的服务器，输入yes回车即可。
>Git会输出一个警告，告诉你已经把GitHub的Key添加到本机的一个信任列表里了：
```shell
Warning: Permanently added 'github.com' (RSA) to the list of known hosts.
```
>这个警告只会出现一次，后面的操作就不会有这个警告了。


**5.2 删除远程库

如果添加的时候地址写错了，或者就是想删除远程库，可以用git remote rm name命令。使用前，建议先用git remote -v来查看远程库信息：

```shell
$ git remote -v
origin git@github.com:alanturing/learngit.git(fetch)
origin git@github.com:alanturing/learngit.git(push)
```

然后根据名字删除，比如删除origin：

```shell
$ git remote rm origin
```

此处的删除，其实是解除了本地和远程的绑定关系，并不是物理上删除了远程库。远程库本身并没有任何改动。要真正删除远程库，需要登录到GitHub，在后台页面手动删除。

**5.3 从远程库克隆

我们已经了解了，先有本地库，后有远程库的时候，如何关联远程库。

现在，假设我们从零开发，那么最好的方式是先创建远程库，再从远程库克隆。

首先，登录GitHub，创建一个新的仓库，取个名字，比如gitskills：

![13]({{ '/assets/images/GIT-13.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

勾选Initialize this repository with a README，这样GitHub会自动为我们创建一个README.md文件。创建完毕后，可以看到README.md文件：

![14]({{ '/assets/images/GIT-14.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

现在，远程库已经准备好了，下一步是在某个目录下，比如/users/alan/，在git bash里用命令git clone克隆一个本地库：

```shell
$ git clone git@github.com:alanturing/gitskills.git
Cloning into 'gitskills'...
remote: Counting objects: 3, done.
remote: Total 3 (delta 0), reused 0 (delta 0), pack-reused 3
Receiving objects: 100% (3/3), done.
```

上述操作会在该目录下，/users/alan/，创建一个gitskills目录，而这个gitskills目录下，就会已经有README.md文件了：

```shell
$ cd gitskills
$ ls
README.md
```

使用git clone命令从GitHub的远程库上克隆到本地的仓库，本地的这个仓库就已经是Git仓库了，而且和这个远程的GitHub库是关联的。所以在本地进行修改，之后git add，git commit之后，再使用git push就可以将本地修改上传到GitHub的远程库里了。

如果有多个人协作开发，那每个人各自从远程克隆一份就可以了。

>所以说，5.1和5.3分别是两种远程库和本地库关联的方法，一种是先创建本地库，然后建立远程库，并关联。另一种是直接在GitHub上建立仓库，再克隆到本地库，这样就已经默认关联了。

**6. 分支管理

分支就如同平行宇宙，当你正在学习Git的时候，另一个宇宙的你正在学习SVN。如果两个平行宇宙互不干扰，那就没有什么区别，但如果在某个时间点，两个宇宙合并了，结果就是你既学会了Git又学会了SVN。

分支在实际中有什么作用呢？假设你准备开发一个新功能，但需要两周才能完成，第一周写了50%的代码，如果立刻提交，由于代码还没写完，不完整的代码库会导致其他人无法继续正常工作。但如果等代码全部写完再一次性提交，又存在丢失进度的风险。

现在有了分支，就可以解决这个问题。你创建了一个属于你自己的分支，别人看不到，还可以继续在之前的分支上正常工作，而你在自己的分支上干活，想提交就提交，直到开发完毕后，再一次性合并到之前的分支上，这样，既安全，又不影响别人工作。

其他版本控制系统如SVN等都有分支管理，但这些版本控制系统的分支创建、切换、合并等操作都极其繁琐，所以很少有人使用。

但Git的分支，无论创建、切换和删除分支，Git都能很快（若干秒或者更短）完成，不管版本库里有多少个文件都是一样。

**6.1 创建与合并分支

在之前的版本回退章节里，我们已经知道，每次提交，Git都将它们串成一条时间线，而这条时间线就是一个分支。截止到目前，只有一个时间线，在Git里，这个分支叫主分支，即master分支。HEAD严格来说不是指向提交，而是指向master，master才是指向提交的，所以HEAD指向的就是当前分支。

一开始的时候，master分支是一条线，Git用master指向最新的提交，再用HEAD指向master，就能确定当前分支，以及当前分支的提交点：

![15]({{ '/assets/images/GIT-15.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

每次提交，master分支都会向前移动一步，这样，随着你不断提交，master分支的线也越来越长。

当我们创建新的分支，例如dev时，Git新建了一个指针dev，指向master相同的提交，再把HEAD指向dev，就表示当前分支在dev上：

![16]({{ '/assets/images/GIT-16.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

Git创建一个分支很快，因为除了增加一个dev指针，更改HEAD的指向，工作区的文件都没有任何变化。

不过，从现在开始，对工作区的修改和提交就是针对dev分支了，比如新提交一次后，dev指针向前移动一步，而master指针不变：

![17]({{ '/assets/images/GIT-17.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

假如我们在dev上的工作完成了，就可以把dev合并到master上。Git怎么合并呢？最简单的方法，就是直接把master指向dev的当前提交，就完成了合并：

![18]({{ '/assets/images/GIT-18.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

所以Git合并也很快，就修改一下指针，工作区内容也不变。

合并完分支后，甚至可以删除dev分支。删除dev分支就是把dev指针删除掉，删掉后，我们就剩下了一条master分支：

![19]({{ '/assets/images/GIT-19.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

下面举个例子。

首先，我们创建dev分支，然后切换到dev分支：

```shell
$ git checkout -b dev
Switched to a new branch 'dev'
```

git checkout命令加上-b表示创建并切换，其等价于以下两条命令：

```shell
$ git branch dev
$ git checkout dev
Switched to branch 'dev'
```

然后，用git branch命令查看当前分支：

```shell
$ git branch
* dev
  master
```

git branch命令会列出所有分支，当前分支前面会标一个$$\ast$$号。

然后，我们就可以在dev分支上正常提交，比如对readme.txt做个修改，加上一行：

```shell
Creating a new branch is quick
```

然后提交：

```shell
$ git add readme.txt
$ git commit -m "branch test"
[dev b17d20e] branch test
 1 file changed, 1 insertion(+)
```

现在dev分支的工作完成，我们就可以切换回master分支：

```shell
$ git checkout master
Switched to branch 'master'
```

切换回master分支后，再查看readme.txt文件，发现刚添加的内容不见了，这是因为那个提交在dev分支上，而master分支此刻的提交点并没有变：

![20]({{ '/assets/images/GIT-20.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

现在，我们把dev分支的工作成果合并到master分支上：

```shell
$ git merge dev
Updating d46f35e..b17d20e
Fast-forward
 readme.txt | 1 +
 1 file changed, 1 insertion(+)
```

git merge命令用于合并指定分支到当前分支（之前已经用git checkout将当前分支改为master了）。合并后，再查看readme.txt的内容，就可以看到，和dev分支的最新提交是完全一样的。

注意到上面的Fast-forward信息，Git告诉我们，这次合并是快进模式，也就是直接把master指向dev的当前提交，所以合并速度非常快。

>对于Fast-forward的情况，Git里的合并分支，实际上就是将当前分支的指针设置为被合并的分支的指针。

当然，并不是每次合并都能够Fast-forward，之后会说其他方式的合并。

合并完成后，可以放心地删除dev分支了：

```shell
$ git branch -d dev
Deleted branch dev (was b17d20e).
```

删除后，查看branch，就只有master分支了：

```shell
$ git branch
* master
```

因为创建、合并和删除分支非常快，所以Git鼓励使用分支完成某个任务，合并后再删除分支，这和直接在master分支上工作效果是一样的，但更加安全。

我们注意到切换分支只用git checkout branch，而之前所说的撤销修改则是git checkout -- file，同一个命令有两种作用。比较令人迷惑。

实际上，切换分支这个动作，用switch更加科学。因此，最新版本的Git提供了git switch命令来切换分支：

```shell
$ git switch -c dev
```

直接切换到已有的master分支，可以用：

```shell
$ git switch master
```

上述git switch里的-c，是创建并切换的意思。

新的git swtich命令比之前的git checkout命令更加容易理解。


**6.2 解决冲突

合并分支有时候也会遇到问题。

准备新的feature1分支：

```shell
$ git switch -c feature1
Switched to a new branch 'feature1'
```

修改readme.txt最后一行，改为：

```shell
Creating a new branch is quick AND simple
```

在feature1分支上提交：

```shell
$ git add readme.txt

$ git commit -m "AND simple"
[feature1 14096d0] AND simple
 1 file changed, 1 insertion(+), 1 deletion(-)
```

切换到master分支：

```shell
$ git switch master
Switched to branch 'master'
Your branch is ahead of 'origin/master' by 1 commit
  (use "git push" to publish your local commits)
```

Git还会自动提醒我们本地的master分支比远程的master分支要超前一个提交。

在master分支上将readme.txt文件的最后一行改为：

```shell
Creating a new branch is quick & simple
```

提交：
```shell
$ git add readme.txt

$ git commit -m "$ simple"
[master 5dc6824] & simple
 1 file changed, 1 insertion(+), 1 deletion(-)
```

现在master分支和feature1分支都有了新的提交，变成了这样：

![21]({{ '/assets/images/GIT-21.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

这种情况下，Git无法执行快进合并，只能试图将各自的修改合并起来，但这种合并就可能会有冲突，尝试一下：

```shell
$ git merge feature1
Auto-merging readme.txt
CONFICT (content): Merge conflict in readme.txt
Automatic merge failed: fix conflicts and the commit the result
```

果然冲突了，Git告诉我们，readme.txt文件存在冲突，必须手动解决冲突后在提交。git status也可以告诉我们冲突的文件：

```shell
$ git status
On branch master
Your branch is ahead of 'origin/master' by 2 commits.
  (use "git push" to publish your local commits)

You have unmerged paths.
  (fix conflicts and run "git commit")
  (use "git merge --abort" to abort the merge)

Unmerged paths:
  (use "git add <file>..." to mark resolution)

	both modified:   readme.txt

no changes added to commit (use "git add" and/or "git commit -a")
```

我们可以直接查看readme.txt的内容：

```shell
$ cat readme.txt
Git is a distributed version control system.
Git is free software distributed under the GPL.
Git has a mutable index called stage.
Git tracks changes of files.
<<<<<<< HEAD
Creating a new branch is quick & simple.
=======
Creating a new branch is quick AND simple.
>>>>>>> feature1
```

我们发现Git使用<<<<<<<，=======和>>>>>>>标记出不同分支的内容。

我们手动修改readme.txt最后一行如下：

```shell
Creating a new branch is quick and simple.
```

再提交：

```shell
$ git add readme.txt
$ git commit -m "conflict fixed"
[master cf810e4] conflict fixed
```

现在，master和feature1的分支变成了如下形式：

![22]({{ '/assets/images/GIT-22.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

用带参数的git log也可以看到分支的合并情况：

```shell
$ git log --graph --pretty=online --abbrev-commit
$ git log --graph --pretty=oneline --abbrev-commit
*   cf810e4 (HEAD -> master) conflict fixed
|\  
| * 14096d0 (feature1) AND simple
* | 5dc6824 & simple
|/  
* b17d20e branch test
* d46f35e (origin/master) remove test.txt
* b84166e add test.txt
* 519219b git tracks changes
* e43a48b understand how stage works
* 1094adb append GPL
* e475afc add distributed
* eaadf4e wrote a readme file
```

最后，删除feature1分支：

```shell
$ git branch -d feature1
Deleted branch feature1 (was 14096d0)
```

所以说，当Git无法自动合并分支的时候，就必须首先解决冲突。解决冲突后，再提交，合并完成。

解决冲突就是把Git合并失败的文件手动编辑为我们希望的内容，再提交。

用git log --graph命令可以看到分支合并图。

**6.3 分支管理策略

通常，合并分支时，如果可能，Git会使用Fast forward模式，但这种模式下，删除分支后，会丢掉分支信息。

如果要强制禁用Fast forward模式，Git就会在merge时生成一个新的commit，这样，从分支历史上就可以看出来分支信息。

我们来尝试一下--no-ff方式的git merge：

```shell
$ git switch -c dev
Switched to a new branch 'dev'
```

修改readme.txt文件，并提交一个新的commit：

```shell
$ git add readme.txt
$ git commit -m "add merge"
[dev f52c633] add merge
 1 file changed, 1 insertion(+)
```

现在，我们切换回master：

```shell
$ git switch master
Switched to branch 'master'
```

准备合并dev分支，请注意--no-ff参数，表示禁用Fast forward：

```shell
$ git merge --no-ff -m "merge with no-ff" dev
Merge made by the 'recursive' strategy
 readme.txt | 1 +
 1 file changed, 1 insertion(+)
```

因为本次合并需要创建一个新的commit，所以加上-m参数，把commit描述写进去。

合并后，我们使用git log来看分支历史：

```shell
$ git log --graph --pretty=oneline --abbrev-commit
*   e1e9c68 (HEAD -> master) merge with no-ff
|\  
| * f52c633 (dev) add merge
|/  
*   cf810e4 conflict fixed
...
```

可以看到，不使用Fast forward，git merge之后就像下图：

![23]({{ '/assets/images/GIT-23.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

在实际开发中，我们应该按照几个基本原则进行分支管理：

首先，master分支应该是非常稳定的，也就是仅用来发布新版本，平时不能在上面干活；

干活都在dev分支上，也就是说，dev分支是不稳定的，到某个时候，比如说1.0版本发布时，再把dev分支合并到master上，在master分支发布1.0版本。

如果多人在dev分支上干活，每个人都有自己的分支，时不时的往dev分支上合并就可以了。所以，团队合作的分支看起来就像这样：

![24]({{ '/assets/images/GIT-24.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

**6.4 Bug分支

在软件开发中，经常会遇到bug。有bug就需要修复。在Git中，由于分支十分强大，所以每个bug都可以通过一个新的临时分支来修复，修复后，合并分支，然后删除临时分支。

当接到一个修复代号为101的bug的任务时，很自然的想创建一个分支isseu-101来修复它，但是，当前还在dev分支上进行的工作还没有提交：

```shell
$ git status
On branch dev
Changes to be commited:
    (use "git reset HEAD <file>..." to usage)
    
      new file:   hello.py

Changes not staged for commit:
    (use "git add <file>..." to update what will be commited)
    (use "git chekcout -- <file>..." to discard changeds in working directory)
    
      modified: readme.txt
```

在dev分支上的工作刚进行到一半，还无法立刻提交。但是要求此bug必须在两小时内修复，这时该如何做呢？

Git提供了一个stash功能，可以把当前工作先从储藏起来，等以后恢复现场后继续工作：

```shell
$ git stash
Saved working directory and index state WIP on dev: f52c633 add merge
```

现在用git status查看工作区，就是干净的（除非有没有被Git管理的问及那），因此可以放心的创建分支来修复bug了。

首先确定要在哪个分支上修复bug，如果在master分支上修复，就从master创建临时分支：

```shell
$ git checkout master
Switched to branch 'master'
Your branch is ahead of 'origin/master' by 6 commits.
    (use "git push" to publish your local commits)
$ git checkout -b issue-101
Switched to a new branch 'issue-101'
```

现在修复bug，将readme里的Git is free software，改成Git is a free software。然后提交：

```shell
$ git add readme.txt

$ git commit -m "fix bu 101"
[issue-101 4c805e2] fix bug 101
 1 file changed, 1 insertion(+), 1 deletion(-)
```

修复完成后，切换到master分支，并完成合并，最后删除issue-101分支：

```shell
$ git switch master
Switched to branch 'master'
Your branch is ahead of 'origin/master' by 6 commits.
    (use "git push" to publish your local commits)

$ git merge --no-ff -m "merged bug fix 101" issue-101
Merge made by the 'recursive' strategy.
    readme.txt | 2 + -
    1 file changed, 1 insertion(+), 1 deletion(-)
```

现在bug修复完成了，我们可以回到之前的dev分支继续干活了：

```shell
$ git switch dev
Switch to branch 'dev'

$ git status
nothing to commit, working tree clean
```

工作区是干净的，用git stash list命令查看之前存的工作现场：

```shell
$ git stash list
stash@{0}: WIP on dev: f52c633 add merge
```

工作现场还在，Git把stash内容存在某个地方了，需要恢复，有两种方法：

一是用git stash apply恢复，但是恢复后，stash的内容并不删除，需要用git stash drop来手动删除；另一种方式是用git stash pop，恢复的同时将stash内容也删除：

```shell
$ git stash pop
On branch dev
Changes to be commited:
    (use "git reset HEAD <file>..." to usage)
    
      new file:   hello.py

Changes not staged for commit:
    (use "git add <file>..." to update what will be commited)
    (use "git chekcout -- <file>..." to discard changeds in working directory)
    
      modified: readme.txt

Dropped ref/stash@{0}: (5d677e2ee266f39ea296182fb2354265b91b3b2a)
```

再用git stash list查看，就找不到该存储的工作区了：

```shell
$ git stash list
```

可以多次stash，恢复的时候，先用git stash list查看，再恢复指定的stash：

```shell
$ git stash apply stash@{0}
```

我们在master上修复了bug，但注意到dev分支也是从master上分出来的，所以该bug在dev分支上也有，如果合并的话，还会将这个带有bug的文件合并到master分支，需要再修复一次一样的bug。那是否有简单的方法在dev分支上修复同样的bug呢？

同样的bug，要在dev上修复，只需要把4c805e2 fix bug 101这个提交所做的修改“复制”到dev分支。注意：我们只想复制4c805e2 fix bug 101这个提交所做的修改，而不是将整个master分支merge过来。

Git专门提供了一个cherry-pick命令，让我们能复制一个特定的提交到当前分支：

```shell
$ git branch
*dev
master

$ git cherry-pick 4c805e2
[master 1d4b803] fix bug 101
 1 file changed, 1 insertion(+), 1 deletion(-)
```

Git自动给dev分支做了一次提交，这次提交的commit的id是1d4b803，并不同于master分支的4c805e2的这次commit，因为这两个commit只是内容相同，但是是两个不同的commit。用git cherry-pick，我们就不需要再dev分支上再手动把已经修复的bug再修复一次。

既然可以在master分支上修复bug，再在dev分支上重复这个修复过程，那么直接在dev分支上修复这个bug，再在master分支上重复是否可行呢？当然也是可行的，但一样的，如果当前dev分支还有工作没做完不能提交，在切换到master分支上之前，还是需要git stash保存dev分支现场。

**6.5 Feature分支

在软件开发过程中，总有无穷无尽的新的功能要不断的添加进来。

添加一个新功能时，肯定不希望一些实验性质的代码将主分支弄乱，所以每添加一个新的功能，最好新建一个feature分支，在上面开发，完成后，合并，最后，删除该feature分支。

现在，接到一个新任务，开发代号为Vulcan的新功能。

创建一个新分支准备开发：

```shell
$ git switch -c feature-vulcan
Switched to a new branch 'feature-vulcan'
```

开发完毕后：

```shell
$ git add vulcan.py

$ git status
On branch feature-vulcan
Changes to be commited:
    (use "git reset HEAD <file>..." to unstage)

      new file:   vulcan.py

$ git commit -m "add feature vulcan"
[feature-vulcan 2877773e] add feature vulcan
 1 file changed, 2 insertions(+)
 create mode 100644 vulcan.py
```

切回dev分支，准备合并：

```shell
$ git switch dev
```

一切顺利的话，feature-vulcan分支被合并到dev分支上，然后被删除。

但是这时候接到通知，经费不足，该功能取消，而且该分支包含机密资料，需要被销毁：

```shell
$ git branch -d feature-vulcan
error: The branch 'feature-vulcan' is not fully merged.
If you are sure you want to delete it, run 'git branch -D feature-vulcan'.
```

销毁失败。Git提醒，feature-vulcan分支还没被合并，如果删除的话就会丢失修改，如果要强行删除，需要使用大写的-D参数。

```shell
$ git branch -D feature-vulcan
Deleted branch feature-vulcan (was 287773e).
```

删除成功。

**6.6 多人协作

当你从远程仓库克隆时，实际上Git自动把本地的master分支和远程的master分支对应起来了，并且，远程仓库的默认名称是origin。

要查看远程库的信息，用git remote：

```shell
$ git remote
origin
```

或者，用git remote -v显示更详细的信息：

```shell
$ git remote -v
orogin git@github.com:alanturing/learngit.git (fetch)
origin git@github.com:alanturing/learngit.git (push)
```

上面显示了可以抓取和推送的origin地址，如果没有推送权限，就看不到push的地址。

**6.6.1 推送分支

推送分支，就是把该分支上的所有本地提交推送到远程库。推送时，要制定本地分支，这样，Git就会把该分支推送到远程库对应的远程分支上：

```shell
$ git push origin master
```

如果要推送其他分支，比如dev，就改成：

```shell
$ git push origin dev
```

但是实际上并不是一定要把本地分支都推送到远程去，一般来说，master分支是主分支，因此需要时刻与远程同步；dev分支是开发分支，团队所有成员都要在上面工作，所以也需要与远程同步；bug分支只用于在本地修复bug，就没必要推送了；feature分支是否推送到远程，取决于你是否和其他同事合作在上面开发。

分支完全可以在本地不推送到远程，是否推送，取决于具体情况。

**6.6.2 抓取分支

多人协作时，大家都会往master和dev分支上推送各自的修改。

现在，假设有另一个同事，在另一台电脑（需要将SSH Key添加到Github）或者同一台电脑的另一个目录下克隆：

```shell
$ git clone git@github.com:alanturing/learngit.git
Cloning into 'learngit'
remote: Counting objects: 40, done.
remote: Compressing objects: 100% (21/21), done.
remote: Total 40 (delta 14), reused 40 (delta 14), pack-reused 0
Receiving objects: 100% (40/40), done.
Resolving deltas: 100% (14/14), done.
```

当你的同事从远程库clone时，默认情况下，他只能看到本地的master分支：

```shell
$ git branch
* master
```

现在你的同事要在dev分支上开发，就必须创建远程origin的dev分支到本地，于是他用这个命令创建本地dev分支：

```shell
$ git checkout -b dev origin/dev
```

现在，他就可以在本地dev分支上修改，然后，时不时的将本地dev分支push到远程库的dev分支上了：

```shell
$ git add env.txt

$ git commit -m "add env"
[dev 7a5e5dd] add env
 1 file changed, 1 insertion(+)
 create mode 100644 env.txt

$ git push origin dev
Counting objects: 3, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (2/2), done.
Writing objects: 100% (3/3), 308 bytes | 308.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0)
To github.com:alanturing/learngit.git
    f52c633...7a5e5dd  dev -> dev
```

现在你的同事已经向origin/dev推送了他的提交，而碰巧你也对同样的文件做了修改，并试图推送：

```shell
$ cat env.txt
env

$ git add env.txt

$ git commit -m "add new env"
[dev 7bd91f1] add new env
 1 file changed, 1 insertion(+)
 create mode 100644 env.txt

$ git push origin dev
To github.com:alanturing/learngit.git
  ! [rejected]      dev -> dev (non-fast-forward)
error: failed to push some refs to 'git@github.com:alanturing/learngit.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details
```

推送失败，因为你的同事的最新提交和你试图推送的提交有冲突，解决办法也很简单，Git已经提示了，先用git pull把最新版本的提交从origin/dev上抓取下来，然后，在本地合并，解决冲突，再推送：

```shell
$ git pull
There is no tracking information for the current branch.
Please specify which branch you want to merge with.
See git-pull(1) for details.

    git-pull <remote> <branch>

If you wish to set tracking information for this branch you can do so with:

    git branch --set-upstream-to=origin/<branch> dev
```

git pull也失败了，因为没有指定本地dev分支与远程origin/dev分支的链接，根据提示，设置dev与origin/dev的链接：

```shell
$ git branch --set-upstream-to=origin/dev dev
Branch 'dev' set up to track remote branch 'dev' from 'origin'
```

现在再pull：

```shell
$ git pull
Auto-mergin env.txt
CONFLICT (add/add): Merge conflict in env.txt
Automatic merge failed; fix conflicts and then commit the result.
```

这次git pull成功，但是合并有冲突，需要手动解决，解决的办法和分支管理中的解决冲突完全一样。解决后，提交，再push：

```shell
$ git commit-m "fix env conflict"
[dev 57c53ab] fix env conflict

$ git push origin dev
Counting objects: 6, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (4/4), done.
Writing objects: 100% (6/6), 621 bytes | 621.00 KiB/s, done.
Total 6 (delta 0), reused 0 (delta 0)
To github.com:michaelliao/learngit.git
   7a5e5dd..57c53ab  dev -> dev
```

因此，多人协作的工作模式通常是这样：

* 1. 首先，可以试图用git push origin branch-name推送自己的修改；
* 2. 如果推送失败，则因为远程分支比本地要更新，需要先用git pull试图合并；
* 3. 如果本病有冲突，则解决冲突，并在本地提交；
* 4. 没有冲突或者冲突解决后，再用git push origin branch-name，推送就能成功

如果git pull提示no tracking information，则说明本地分支和远程分支的链接关系没有创建，用命令git branch --set-upstream-to branch-name origin/branch-name来创建。

**6.7 Rebase

在上一节我们看到，多人在同一个分支上协作时，很容易出现冲突。即使没有冲突，后push的人也不得不先pull，在本地合并，然后再能push成功。

每次合并再push，分支变成了这样：

```shell
$ git log --graph --pretty=oneline --abbrev-commit
* d1be385 (HEAD -> master, origin/master) init hello
*   e5e69f1 Merge branch 'dev'
|\  
| *   57c53ab (origin/dev, dev) fix env conflict
| |\  
| | * 7a5e5dd add env
| * | 7bd91f1 add new env
| |/  
* |   12a631b merged bug fix 101
|\ \  
| * | 4c805e2 fix bug 101
|/ /  
* |   e1e9c68 merge with no-ff
|\ \  
| |/  
| * f52c633 add merge
|/  
*   cf810e4 conflict fixed
```

看起来很乱。所以，能不能让Git的提交历史是一条干净的直线呢？

用Git里的rebase操作就可以做到，翻译为变基。

从实际问题出发，看看怎么把分叉的提交变成直线。

在和远程分支同步后，我们对hello.py这个文件做了两次提交。用git log查看：

```shell
$ git log --graph --pretty=oneline --abbrev-commit
* 582d922 (HEAD -> master) add author
* 8875536 add comment
* d1be385 (origin/master) init hello
*   e5e69f1 Merge branch 'dev'
|\  
| *   57c53ab (origin/dev, dev) fix env conflict
| |\  
| | * 7a5e5dd add env
| * | 7bd91f1 add new env
...
```

注意到，Git用(HEAD -> master)和(origin/master)标识出当前分支的HEAD和远程origin的位置分别是582d922 add author和d1be385 init hello，本地比远程快两个提交。

现在我们尝试推送本地分支：

```shell
$ git push origin master
To github.com:alanturing/learngit.git
 ! [rejected]        master -> master (fetch first)
error: failed to push some refs to 'git@github.com:michaelliao/learngit.git'
hint: Updates were rejected because the remote contains work that you do
hint: not have locally. This is usually caused by another repository pushing
hint: to the same ref. You may want to first integrate the remote changes
hint: (e.g., 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.
```

失败了，说明有人先于我们推送了远程分支。按照经验，先pull一下：

```shell
$ git pull
remote: Counting objects: 3, done.
remote: Compressing objects: 100% (1/1), done.
remote: Total 3 (delta 1), reused 3 (delta 1), pack-reused 0
Unpacking objects: 100% (3/3), done.
From github.com:michaelliao/learngit
   d1be385..f005ed4  master     -> origin/master
 * [new tag]         v1.0       -> v1.0
Auto-merging hello.py
Merge made by the 'recursive' strategy.
 hello.py | 1 +
 1 file changed, 1 insertion(+)
```

再用git status查看状态：

```shell
$ git status
On branch master
Your branch is ahead of 'origin/master' by 3 commits
   (use "git push" to publish your local commits)

nothing to commit, working tree clean
```

加上git pull合并的提交，现在本地分支比远程分支超前三个提交。

用git log查看：

```shell
$ git log --graph --pretty=oneline --abbrev-commit
*   e0ea545 (HEAD -> master) Merge branch 'master' of github.com:michaelliao/learngit
|\  
| * f005ed4 (origin/master) set exit=1
* | 582d922 add author
* | 8875536 add comment
|/  
* d1be385 init hello
...
```

现在将本地分支push到远程没有问题，但是提交历史分叉了，不是很好看。这时候，rebase就可以被使用了。我们输入命令git rebase试试：

```shell
$ git rebase
First, rewinding head to replay your work on top of it...
Applying: add comment
Using index info to reconstruct a base tree...
M	hello.py
Falling back to patching base and 3-way merge...
Auto-merging hello.py
Applying: add author
Using index info to reconstruct a base tree...
M	hello.py
Falling back to patching base and 3-way merge...
Auto-merging hello.py
```

输出了很多信息，用git log来查看一下：

```shell
$ git log --graph --pretty=oneline --abbrev-commit
* 7e61ed4 (HEAD -> master) add author
* 3611cfe add comment
* f005ed4 (origin/master) set exit=1
* d1be385 init hello
...
```

原本分叉的提交现在变成一条直线了。这是怎么实现的呢？Git把本地的提交挪动了位置，放到了f005ed4 (origin/master) set exit=1之后，这样，整个提交历史就变成了一条直线。rebase操作前后，最终的提交内容是一致的，但是，本地的commit修改内容已经变化了，它们的修改不再基于dlbe385 init hello，而是基于f005ed4 (origin/master) set exit=1，但最后的提交7e61ed4的内容是一致的。

这就是rebase操作的特点：把分叉的提交历史整理成一条直线，看上去更直观。缺点是本地的分叉提交已经被修改过了。

最后，通过push操作把本地分支推送到远程：

```shell
$ git push origin master
Counting objects: 6, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (5/5), done.
Writing objects: 100% (6/6), 576 bytes | 576.00 KiB/s, done.
Total 6 (delta 2), reused 0 (delta 0)
remote: Resolving deltas: 100% (2/2), completed with 1 local object.
To github.com:michaelliao/learngit.git
   f005ed4..7e61ed4  master -> master
```

再用git log看看效果：

```shell
$ git log --graph --pretty=oneline --abbrev-commit
* 7e61ed4 (HEAD -> master, origin/master) add author
* 3611cfe add comment
* f005ed4 set exit=1
* d1be385 init hello
...
```

远程分支的提交历史也是一条直线。


**7. 标签管理

发布一个版本时，我们通常先在版本库中打上一个标签（tag），这样，就唯一确定了打标签时刻的版本。将来无论什么时候，取某个标签的版本，就是把那个打标签时刻的历史版本取出来。所以，标签也是版本库的一个快照。

Git的标签虽然是版本库的快照，但实际上它就是指向某个commit的指针，和分支很像，但是分支可以移动，而标签不能，所以，创建和删除标签都是瞬间完成的。

Git已经有commit可以追溯历史版本了，为什么还要tag呢？

因为commit的id是一串很长的代码，与其说将commit号是6a5819e...的版本取出来，不如说将tag v1.2的版本取出来比较简洁，而且有明确含义。

所以说，tag就是一个有意义的名字，其与某个commit绑定在一起。


>tag 对应某次 commit, 是一个点，是不可移动的。
>branch 对应一系列 commit，是很多点连成的一根线，有一个HEAD 指针，是可以依靠 HEAD 指针移动的。
>所以，两者的区别决定了使用方式，改动代码用 branch ,不改动只查看用 tag。
>tag 和 branch 的相互配合使用，有时候起到非常方便的效果，例如 已经发布了 v1.0 v2.0 v3.0 三个版本，这个时候，我突然想不改现有代码的前提下，在 v2.0 的基础上加个新功能，作为 v4.0 发布。就可以 检出 v2.0 的代码作为一个 branch ，然后作为开发分支。



**7.1 创建标签

在Git上打标签非常简单，首先，切换到需要打标签的分支上：

```shell
$ git branch
* dev
master
$ git checkout master
Switched to branch 'master'
```

然后，用命令git tag name就可以打一个新标签：

```shell
$ git tag v1.0
```

用命令git tag查看所有标签：

```shell
$ git tag
v1.0
```

默认标签是打在这个分支最新提交的commit上的。如果想给过去的commit打标签的话，要找到历史提交的commit id，然后打上就行：

```shell
$ git log --pretty=oneline --abbrev-commit
12a631b (HEAD -> master, tag: v1.0, origin/master) merged bug fix 101
4c805e2 fix bug 101
e1e9c68 merge with no-ff
f52c633 add merge
cf810e4 conflict fixed
5dc6824 & simple
14096d0 AND simple
b17d20e branch test
d46f35e remove test.txt
b84166e add test.txt
519219b git tracks changes
e43a48b understand how stage works
1094adb append GPL
e475afc add distributed
eaadf4e wrote a readme file
```

比如说要对add merge这次提交打标签，它对应的commit id是f52c633：

```shell
$ git tag v0.9 f52c633
```

再用git tag查看标签：

```shell
$ git tag
v0.9
v1.0
```

>git tag tagname用于新建一个标签，默认在HEAD上新建，也可以指定一个commit id。

>HEAD就是当前活跃分支的游标。形象的记忆就是：你现在在哪儿，HEAD就指向哪儿，所以Git才知道你在那儿！不过HEAD并非只能指向分支的最顶端（时间节点距今最近的那个），实际上它可以指向任何一个节点，它就是Git内部用来追踪当前位置的东西。

标签不是按时间顺序列出，而是按字母顺序排列的。可以用git show tagname来查看标签信息：

```shell
$ git show v0.9
commit f52c63349bc3c1593499807e5c8e972b82c8f286 (tag: v0.9)
Author: Alan Turing <alanturing@example.com>
Date:   Fri May 18 21:56:54 2018 +0800

    add merge

diff --git a/readme.txt b/readme.txt
...
```

可以看到，v0.9确实打在add merge这次提交上。

>需要注意，git tag name这个命令会将tag打在本分支的最后一个commit上，如果想打在历史commit上，则需要用git log --pretty=oneline --abbrev-commit来查看历史commit，但是这个命令查看的仍然是本分支的历史commit。所以说在上面的例子里是master分支，如果想在dev分支某个commit上打标签，在master分支上用git log查不到dev分支的commit的id。但是如果假设你事先就记下来了dev分支上某个commit的id，那在master分支上，是否可以直接给这个dev分支的commit打标签呢？

>需要说明的是，创建 tag 是基于本地分支的commit，而且与分支的推送是两回事，就是说分支已经推送到远程了，但是你的tag并没有，如果把tag推送到远程分支上，需要另外执行tag的推送命令。

我们还可以创建带有说明的标签，用-a指定标签名，-m指定说明文字：

```shell
$ git tag -a v0.1 -m "version 0.1 released" 1094adb
```

用命令git show tagename可以看到说明文字：

```shell
$ git show v0.1
tag v0.1
Tagger: Alan Turing <alanturing@example.com>
Date:   Fri May 18 22:48:43 2018 +0800

version 0.1 released

commit 1094adb7b9b3807259d8cb349e7df1d4d6477073 (tag: v0.1)
Author: Alan Turing <alanturing@example.com>
Date:   Fri May 18 21:06:15 2018 +0800

    append GPL

diff --git a/readme.txt b/readme.txt
...
```

**7.2 操作标签

如果标签打错了，也可以删除：

```shell
$ git tag -d v0.1
Deleted tag 'v0.1' (was f15b0dd)
```

创建的标签都只存储在本地，不会自动推送到远程。所以，打错的标签可以在本地安全删除。

如果要推送某个标签到远程，使用命令git push origin tagname：

```shell
$ git push origin v1.0
Total 0 (delta 0), reused 0 (delta 0)
To github.com:alanturing/learngit.git
 * [new tag]        v1.0 -> v1.0
```

或者一次性推送全部尚未推送到远程的本地标签：

```shell
$ git push origin --tags
Total 0 (delta 0), reused 0 (delta 0)
To github.com:alanturing/learngit.git
 * [new tag]         v0.9 -> v0.9
```

如果标签已经推送到远程，要删除远程标签就麻烦一点，先从本地删除：

```shell
$ git tag -d v0.9
Deleted tag 'v0.9' (was f52c633)
```

然后，从远程删除。删除命令也是push，但格式如下：

```shell
$ git push origin :refs/tags/v0.9
To github.com:alanturing/learngit.git
 - [deleted]         v0.9
```

**8. 使用Github

我们一直用GitHub作为免费的远程仓库，如果是个人的开源项目，放到Github上是完全没有问题的。其实Github还是一个开源协作社区，通过GitHub，既可以让别人参与你的开源项目，也可以参与别人的开源项目。

在GitHub出现之前，开源项目开源容易，但让别人参与进来比较困难。因为要参与，就要提交代码，而给每个想要提交代码的人都创建一个账号是不现实的，因此，对于大多数群体来说，也只能举报bug，即使要改bug，也只能通过邮件的方式，很不方便。

但是在GitHub上，利用Git很强大的克隆和分支功能，群众就可以自由的参与各种开源项目了。

如何参与一个开源项目呢？比如人气很高的bootstrap项目，这是一个而非常强大的CCS框架，可以访问[它的主页](https://github.com/twvs/bootstrap)，点fork，就在自己的账号下克隆两个一个bootstrap仓库，然后从自己的账号下clone到本地：

```shell
$ git clone git@github.com:alanturing/bootstrap.git
```

一定要从自己的账号下clone仓库，这样你才能提交修改。如果从bootstrap的作者的仓库地址git@github.com:twbs/bootstrap.git克隆，因为没有权限，将不能提交修改。

bootstrap的官方仓库twbs/bootstrap，你在GitHub上自己的账号里clone的仓库alanturing/bootstrap，以及你克隆到本地电脑的仓库，之间的关系如下图所示：

![25]({{ '/assets/images/GIT-25.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

如果你想修复bootstrap的一个bug，或者新增一个功能，立刻就可以开始在本地干活，干完后，推送到自己的远程仓库。

如果你希望bootstrap官网库接受你的修改，你就可以在Github上发起一个pull request。这需要对方接受你的pull request，才能将你的修改加入官方库里。


## 4. 在服务器使用Tensorboard的方法

Tensorboard在本机可以方便使用，但使用服务器时需要设置一下。

**1. windows系统**

在Windows系统装一个Xshell，在文件->属性->ssh->隧道->添加，类型local，源主机填127.0.0.1（意思是本机），端口设置一个，比如12345，目标主机为服务器，目标端口一般是6006，如果6006被占了可以改为其他端口。

在服务器上运行

```text
tensorboard --logdir='logs' --port=6006
```

在本机打开网页127.0.0.1:12345 ，即可查看远程的tensorboard。

**2. Mac或Linux系统**

在登录远程服务器的时候使用命令：

```text
ssh -L 16006:127.0.0.1:6006 account@server.address
```

（代替一般ssh远程登录命令：ssh account@server.address）

训练完模型之后使用如下命令：

```text
tensorboard --logdir="/path/to/log-directory"
```

（其中，/path/to/log-directory为自己设定的日志存放路径，因人而异）

最后，在本地访问地址：http://127.0.0.1:16006/


**3. 参考资料**
* https://zhuanlan.zhihu.com/p/359626009
* https://zhuanlan.zhihu.com/p/403439895


## 5. 通过跳板机查看内网服务器的tensorbord的方法

先用`16006`端口接收跳板机的`8008`端口：

```shell
ssh -L 16006:127.0.0.1:8008 account@跳板机ip
```

再用跳板机的`8008`端口接收服务器的`6006`端口：

```shell
ssh -L 8008:127.0.0.1:6006 account@服务器ip
```

`cd`到保存模型的目录，终端输入:

```shell
tensorboard --logdir=./
```

然后直接在本地浏览器打开链接：`http://127.0.0.1:16006/`



## 6. scp和ssh用法

`scp`简称是security copy，通过`ssh`来上传和下载文件（目录）

```shell
# 下载文件或者目录
scp -P ssh端口 -r 远程服务用户名@服务器ip:远程文件或目录 本地文件或目录 
# 接着输入密码即可

# 如果远程服务器ssh端口是默认的22，可以不指定端口，命令简化为
scp -r 远程服务用户名@服务器ip:远程文件或目录 本地文件或目录

# 上传文件或者目录
scp -P ssh端口 -r 本地文件或目录 远程服务用户名@服务器ip:远程文件或目录 
# 接着输入密码即可

# 如果远程服务器ssh端口是默认的22，可以不指定端口，命令简化为
scp -r 本地文件或目录 远程服务用户名@服务器ip:远程文件或目录 
```

由此可见`scp`的上传和下载，只是把远程和本地换下位置而已。而`ssh`更为简单：

```shell
ssh -P ssh端口 远程服务用户名@服务器ip

# 如果远程服务器ssh端口是默认的22，可以不指定端口，命令简化为
ssh 远程服务用户名@服务器ip
```

如果不能直接连接远程服务器，需要使用跳板机的话，那么`ssh`和`scp`都需要使用跳板机才能实现。假设跳板机地址为`host1：192.168.1.100 admin`，目标服务器`host2：192.168.1.110 user`，

通过跳板机使用`ssh`登录目标服务器命令如下：

```shell
# 自己本地主机执行
ssh -J admin@192.168.1.100 user@192.168.1.110
 #回车后先输入跳板机密码，再输入目标机器密码
```

这样就可以通过`ssh`连接到目标主机了，发现只是通过参数`-J`来指定了跳板机，如果需要多次跳转呢？那么命令格式就是`ssh -J 跳板机1 跳板机2 跳板机3 目标服务器`

那么`scp`通过跳板机命令如下：

```shell
# 下载
scp -P 22 -o 'ProxyJump 跳板机' -r 目标服务器:目标文件或目录 本地文件或目录
scp -P 22 -o 'ProxyJump admin@192.168.1.100 -p 22' -r user@192.168.1.110:/user/bin/12533.dump ./12533.dump
# 简化
scp -o 'ProxyJump admin@192.168.1.100' -r user@192.168.1.110:/user/bin/12533.dump ./12533.dump

#上传
scp -P 22 -o 'ProxyJump 跳板机' -r 本地文件或目录 目标服务器:目标文件或目录 
scp -P 22 -o 'ProxyJump admin@192.168.1.100 -p 22' -r ./12533.dump user@192.168.1.110:/user/bin/12533.dump 

# 简化
scp -o 'ProxyJump admin@192.168.1.100' -r ./12533.dump user@192.168.1.110:/user/bin/12533.dump 
```

发现`scp`通过跳板机和目标服务器传输文件，只是增加了个参数`-o`，用于指定代理， 其它和`scp`完全相同。如果有多个跳板机，那么就在`ProxyJump`增加多个跳板机。



## 6. Anaconda相关方法

查现有环境：`conda info --env`

有两种办法复制环境：

在本机上，直接使用`conda create -n 新环境名 --clone 旧环境名`，复制既有环境

如果要复制到其他机器，就要考虑导出当前环境到文件，利用文件再次创建环境。
* 首先，激活要导出的环境：`conda activate 环境名`
* 之后，导出环境：`conda env export > 环境名.yaml`，利用`conda env export`导出的是个`yaml`格式的文件，该文件记录了环境名，软件源地址以及安装包列表
* 最后，在新机器上使用该`yaml`配置文件创建新环境：`conda env create -f 环境名.yaml`，生成的环境与复制源完全一样（包括环境名），如果想在同一台机器上复制，需要把`yaml`文件中的环境名修改为一个新的名字，否则会冲突。


## 7. faiss module安装问题

使用`pip install faiss-gpu`安装`faiss`包之后，使用的时候可能会报错，可以考虑使用`pip install https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`来安装`faiss`

参考文献：
* https://github.com/facebookresearch/faiss/issues/2064

## 8. pytorch3D module安装问题

[这里](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)是pytorch3D的官方指南。

安装完自己的所需的其他环境后（PyTorch、fvcore、iopath是必须的），编译安装是最不容易出错的：

```shell
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

> 注意一定要预先安装好`Pytorch, fvcore, iopath`，再按照上述代码安装`pytorch3d`，顺序不能反。


参考
* https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
* https://blog.csdn.net/weixin_43357695/article/details/126063091


## 9. xformers module安装问题

使用`diffusers`模块的时候，一般都需要安装`transformers, accelerate`模块，以及`xformers`模块。`transformers`和`accelerate`模块可以直接使用`pip`来安装对应的版本，只需要和`pytorch`版本兼容，就可以正常使用。即使安装的`transformers`和`accelerate`模块和`pytorch`不兼容，也只是在运行的时候报错，并不会因为安装这两个模块而破坏已经安装好的`pytorch`环境。

但是在使用`pip`安装`xformers`模块的时候，是默认会同步安装`pytorch`的，如果安装的`xformers`模块所默认的`pytorch`版本正好和已经安装好的`pytorch`版本一致，那`xformers`就不会安装新的`pytorch`了，但如果不一致，就会删掉之前已有的`pytorch`版本，重新安装新的`pytorch`，以及对应的`xformers`，这样就破坏了已经安装好的`pytorch`环境。

所以，当我们已经知道所需要安装的`pytorch`版本，以及也需要安装`xformers`模块的时候，使用`pip`**同时**安装`pytorch`和`xformers`，就会自动安装好`pytorch`以及兼容的`xformers`。比如说如下所示：

```shell
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 xformers --index-url https://download.pytorch.org/whl/cu118
```

参考
* https://www.felixsanz.dev/articles/compatibility-between-pytorch-cuda-and-xformers-versions


## 10. meshplot module安装问题

和`pytorch3d`一样，在已经安装好了其它模块的情况下，直接从github编译安装`meshplot`是最靠谱的：

```shell
git install "git clone https://github.com/skoch9/meshplot.git"
```

参考：
* https://blog.csdn.net/weixin_43486780/article/details/120433063


