---
layout: post
comments: false
title: "LaTex"
date: 2021-12-14 01:09:00

---

> This post is a summary of common used LaTex grammer.


<!--more-->



---


## 1. Latex括号


| 功能 | 语法 | 显示 |
| :------: | :------: | :------: |
| 圆括号，小括号 | `\left( \frac{a}{b} \right)` | $$\left( \frac{a}{b} \right)$$ |
| 方括号，中括号 | `\left[ \frac{a}{b} \right]` | $$\left[ \frac{a}{b} \right]$$ |
| 花括号，大括号 | `\lbrace \frac{a}{b} \rbrace` | $$\lbrace \frac{a}{b} \rbrace$$ |
| 尖括号 | `\langle \frac{a}{b} \rangle` | $$\langle \frac{a}{b} \rangle$$ |
| 双竖线，范式 | `\lVert \frac{a}{b} \rVert` | $$\lVert \frac{a}{b} \rVert$$ |
| 取整符号 | `\lfloor \frac{a}{b} \rfloor` | $$\lfloor \frac{a}{b} \rfloor$$ |
| 取顶符号 | `\lceil \frac{a}{b} \rceil` | $$\lceil \frac{a}{b} \rceil$$ |
| 混合符号 | `\left( 0,1 \right]` | $$\left( 0,1 \right]$$ |
| 单侧括号 | `\left. \frac{a}{b} \right\}` | $$\left. \frac{a}{b} \right\}$$ |

注意`(`和`)`是普通的小括号，而`\left(`和`\left)`可以使得括号大小和括号内的部分相对于，比如说有分式的时候。


## 2. Latex字体


| 功能 | 语法 | 显示 |
| :------: | :------: | :------: |
| 显示直立文字 | `\textup{text}` | $$\textup{text}$$ |
| 意大利斜体 | `\textit{text}` | $$\textit{text}$$ |
| slanted斜体 | `\textsl{text}` | $$\textsl{text}$$ |
| 显示小体大写文本 | `\textsc{text}` | $$\textsc{text}$$ |
| 中等字体 | `\textmd{text}` | $$\textmd{text}$$ |
| 加粗 | `\textbf{text}` | $$\textbf{text}$$ |
| 默认值 | `\textnormal{text}` | $$\textnormal{text}$$ |
| 斜体 | `\emph{text}` | $$\emph{text}$$ |
| 细体字 | `\textlf{text}` | $$\textlf{text}$$ |
| 使用等宽字体 | `\texttt{text}` | $$\texttt{text}$$ |
| 使用无衬线字体 | `\textsf{text}` | $$\textsf{text}$$ |
| 所有字母大写 | `\uppercase{text}` | $$\uppercase{text}$$ |
| 空心字母 | 先导入包`\usepackage{amsfonts,amssymb}`，`\mathbb{t}` | $$\mathbb{t}$$ |
| 花体字母 | `\mathcal{L}` | $$\mathcal{L}$$ |


## 3. Latex表示数学里的集合关系


| 功能 | 语法 | 显示 |
| :------: | :------: | :------: |
| 集合中的竖线 | `\mid` | $$\mid$$ |
| 属于 | `\in` | $$\in$$ |
| 不属于 | `\not\in`或者`\notin` | $$\not\in$$ |
| A包含于B | `A\subset B` | $$A\subset B$$ |
| A真包含于B | `A\subsetneqq B` | $$A\subsetneqq B$$ |
| A包含B | `A\supset B` | $$A\supset B$$ |
| A真包含B | `A\supsetneqq B` | $$A\supsetneqq B$$ |
| A不包含于B | `A\not\subset B` | $$A\not\subset B$$ |
| A交B | `A\cap B` | $$A\cap B$$ |
| A并B | `A\cup B` | $$A\cup B$$ |
| A大交 | `\bigcap A` | $$\bigcap A$$ |
| A大并 | `\bigcup A` | $$\bigcup A$$ |
| A的闭包/补集 | `\overline{A}` | $$\overline{A}$$ |
| A减去B | `A\setminus B` | $$A\setminus B$$ |
| 空集 | `\emptyset` | $$\emptyset$$ |


## 4. Latex逻辑


| 功能 | 语法 | 显示 |
| :------: | :------: | :------: |
| 存在 | `\exists` | $$\exists$$ |
| 存在且仅存在一个 | `\exists!` | $$\exists!$$ |
| 不存在 | `\nexists` | $$\nexists$$ |
| 所有，任意 | `\forall` | $$\forall$$ |
| 否 | `\neg` | $$\neg$$ |
| 或，析取 | `\lor` | $$\lor$$ |
| 与，合取 | `\land` | $$\land$$ |
| 除 | `\div` | $$\div$$ |
| 蕴含 | `\implies` | $$\implies$$ |
| 当且仅当 | `\iff` | $$\iff$$ |
| 等价 | `\Leftrightarrow` | $$\Leftrightarrow$$ |
| 小于 | `\le` | $$\le$$ |
| 大于 | `\ge` | $$\ge$$ |
| 小于等于 | `\leq` | $$\leq$$ |
| 大于等于 | `\geq` | $$\geq$$ |
| 不等于 | `\neq` | $$\neq$$　|

## 5. Latex数学符号

| 功能 | 语法 | 显示 |
| :------: | :------: | :------: |
| 无穷 | `\infty` | $$\infty$$ |
| 虚数 | `\imath` | $$\imath$$ |
| 虚数 | `\jmath` | $$\jmath$$ | 
| 上箭头 | `\uparrow` | $$\uparrow$$ |
| 上箭头 | `\Uparrow` | $$\Uparrow$$ |
| 下箭头 | `\downarrow` | $$\downarrow$$ |
| 下箭头 | `\Downarrow` | $$\Downarrow$$ |
| 左箭头 | `\leftarrow` | $$\leftarrow$$ |
| 左箭头 | `\Leftarrow` | $$\Leftarrow$$ |
| 右箭头 | `\rightarrow` | $$\rightarrow$$ |
| 右箭头 | `\Rightarrow` | $$\Rightarrow$$ |
| 底端对齐的省略号 | `\ldots` | $$1,2,\ldots,n$$ |
| 中线对齐的省略号 | `\cdots` | $$x_1^2 + x_2^2 + \cdots + x_n^2$$ |
| 竖直对齐的省略号 | `\vdots` | $$\vdots$$ |
| 斜对齐的省略号 | \ddots | $$\ddots$$ |
| 上取整 | `\lceil` 和 `\rceil` | $$\lceil x \rceil$$ |
| 下取整 | `\lfloor` 和 `\rfloor` | $$\lfloor x \rfloor$$ |
| 求和 | `\sum` | $$\sum_{r=1}^n$$ |
| 积分 | `\int` | $$\int_{r=1}^\infty$$ |
| 多重积分 | `\iint` 和 `\iiint`等，有多少个`i`就是多少个多重积分 | $$\iint_{i=1}^\infty$$ |
| 连乘 | `\prod` | $$\prod_{i=1}^{K}$$ |
| 最大值 | `\max` | $$\max_{c_k}$$ |
| 最小值 | `\min` | $$\min_{c_k}$$ |
| 开四次方 | `\sqrt\[4\]{}` | $$\sqrt\[4\]{\frac xy}$$ |

## 6. Latex分类表达式


## 7. Latex上标

| 功能 | 语法 | 显示 |
| :------: | :------: | :------: |
| 帽子 | `\hat{a}` | $$\hat{a}$$ |
| 大帽 | `\widehat` | $$\widehat{a}$$ |
| 反帽子 | `\check{a}` | $$\check{a}$$ |
| 反圆帽子 | `\breve{a}` | $$\breve{a}$$ | 
| 波浪号 | `\tilde{a}` | $$\tilde{a}$$ |
| 大波浪 | `\widetilde` | $$\widetilde{a}$$ |
| 横线 | `\bar{a}` | $$\bar{a}$$ |
| 大横线 | `\overline` | $$\overline{a}$$ |
| 矢量 | `\vec{a}` | $$\vec{a}$$ |
| 升调 | `\acute{a}` | $$\acute{a}$$ |
| 降调 | `\grave{a}` | $$\grave{a}$$ |
| 圆圈 | `\mathring{a}` | $$\mathring{a}$$ |
| 一阶导数符号 | `\dot{a}` | $$\dot{a}$$ |
| 二阶导数符号 | `\ddot{a}` | $$\ddot{a}$$ |

## 8. Latex如何表示argmax和argmin

| 功能 | 语法 | 显示 |
| :------: | :------: | :------: |
| 最大值 | `\mathop{\arg\max}\limits_{\theta}` | $$\mathop{\arg\max}\limits_{\theta}$$ |
| 最小值 | `\mathop{\arg\min}\limits_{\theta}` | $$\mathop{\arg\min}\limits_{\theta}$$ |

