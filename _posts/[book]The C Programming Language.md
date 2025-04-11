---
layout: post
comments: false
title: "[book]The C Programming Language"
date: 2021-11-29 01:09:00
tags: book-reading
---

> 这是The C Programming Language(second edition)这本书的翻译版，加上了个人一些拙略的理解。


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---


## Preface

自从1978年 The C Programming Language 这本书出版以来，计算机领域经历了一场革命。大型计算机的功能越来越强大，而个人计算机的性能也可以与十多年前的大型机相媲美。在此期间，C语言也在悄悄地演进，其发展早已超出了它仅仅作为UNIX操作系统编程语言的初衷。

C语言的普及程度的逐渐增加以及该语言本身的发展，加之很多组织开发出了与其设计有所不同的编译器，所有的这一切都要求对C语言有一个比本书第一版更加精确、更适应其发展的定义。1983年，美国国家协会（ANSI）成立了一个委员会，其目标是制定一个“无歧义性的且与具体机器无关的C语言定义”，而同时又要保持C语言原有的“精神”。结果产生了C语言的ANSI标准。

ANSI标准规范了一些在本书第1版中提及但没有具体描述的结构，特别是解构赋值和枚举。该标准还提供了一种新的函数声明形式，允许在使用过程中对函数的定义进行交叉检查。标准中还详细说明了一个具有标准输入/输出、内存管理和字符串操作等扩展函数集的标准库。它精确地说明了在C语言原始定义中并不明晰的某些特性的行为，同时还明确了C语言中与具体机器相关的一些特性。

本书第2版介绍的是ANSI标准定义的C语言。尽管我们已经注意到了该语言中已经变化了的地方，但我们还是决定在这里只列出它们的新形式。最重要的原因是，新旧形式之间并没有太大的差别；最明显的变化是函数的声明和定义。目前的编译器已经能够支持该标准的大部分特性。

我们将尽力保持本书第1版的简洁性。C语言并不是一种大型语言，也不需要用一本很厚的书来描述。我们在讲解一些关键特性（比如指针）时做了改进，它是C语言程序设计的核心。我们重新对以前的例子进行了精炼，并在某些章节里增加了一些新例子。例如，我们通过实例程序对复杂的声明进行处理，以将复杂的声明转换为描述性的说明或反之。

## Preface for the first edition

C语言是一种通用的程序设计语言，其特点包括简洁的表达式、流行的控制流和数据结构、丰富的运算符集等。C语言不是一种很“高级”的语言，也不“庞大”，并且不专用于某一特定的应用领域。但是，C语言的限制少，通用性强，这使得它比一些公认为更强大的语言使用更方便、效率更高。

C语言最初是由Dennis Ritchie为UNIX操作系统设计的，并在DEC PDP-11计算机上实现。UNIX操作系统、C编译器和几乎所有的UNIX应用程序都是用C语言写的。同时，还有一些适用于其他机器的编译器产品，比如IBM System/370、Honeywell 6000、Interdata 8/32等。但是，C语言不受限于任何特定的机器或系统，使用它可以很容易地编写出不经修改就可以运行在所有支持C语言的机器上的程序。

本书的目的是帮助读者学习如何用C语言编写程序。本书的开头有一个指南性的引言，目的是使新用户能尽快地开始学习；随后在不同的章节中介绍了C语言的各种主要特性；本书的附录中还包括一份参考手册。本书并不仅仅只是讲述了语言的一些规则，而是采用阅读别人的代码、自己编写代码、修改某些代码等不同的方式来指导读者进行学习。书中的大部分例子都可以完整的运行，而不只是孤立的程序段。

本书并不是一本有关程序设计的入门性手册，他要求读者熟悉基本的程序设计概念，如变量、赋值语句、循环和函数等。尽管如此，初级的程序员仍然可以阅读此书，并借此学会C语言。

C语言是一种令人愉快的、具有很强表达能力的通用的语言，适合于编写各种程序。它容易学习，并且随着使用经验的增加，使用者会越来越感到得心应手。


## Introduction

C语言是一种通用性的编程语言。C语言在开发的过程中和UNIX系统有着很紧密的联系，因为UNIX和绝大多数UNIX上的程序都是用C语言编写的。C语言，并不局限于某一特定的操作系统或者机器；而且尽管因为它是被用于编写操作系统和编译器而叫做“system programming language"，C语言其实在很多领域都有很大的作用。

很多C语言里的重要想法来源于Martin Richards开发的BCPL语言。BCPL语言对C语言的影响，通过B语言来间接实现，而B语言是由Ken Thompson在1970年在DEC PDP-7上为第一个UNIX系统开发的。

BCPL和B是typeless语言。相对来说，C提供了一系列不同的data types。基础的types包括characters，intergers以及不同的size的floating-point numbers。而且，还有由pointers，arrays，structures和unions构造的derived data types。Expressions是由operations和operands组成的；任何expression都可以是一个statement（包括assignment，function call等等）。pointers提供和机器无关的地址计算。

C语言为良好的结构的programs提供了基础的control-flow constructions：statement grouping，decision making（if-else），selecting one of a set of possible cases（switch），looping with the termination test at the top（while，for）或者at the bottom（do），以及early loop exit（break）。

functions会返回基本data types，structures，unions或者pointers。任何function都可以被递归调用。function的定义不可以是nested，但variables可以用block-strcutured的方式来声明。一个C program的functions可以在多个分开编译的source files里。variables可能是一个function的internal的，也可以是external但是只在一个source file里的，也可以是在整个program里都可见的。

C是一种相对来说low-level的语言。这个特征并不是贬低C语言；它只是表示C语言处理的objects和大多数computers所处理的objects是类似的，也就是characters，numbers，以及addresses。

C并没有提供operations直接处理复合objects比如character strings，sets，lists，arrays等。并没有能够治理一整个array或者string的operations。除了static definition和functions的local variable提供的stack discipline，C语言也并不定义任何storage allocation。而且C语言也没有heap或者garbage collection。C语言自身不提供input/output工具；并没有READ或者WRITE statements，以及并没有built-in的file access methods。所有的这些high-level机制都必须通过调用functions来提供。大多数的C语言的implementations都会包含一定数量的这样的functions。

相似的，C只提供直接的，single-thread的控制流：tests，loops，grouping和subprograms，并没有multiprogramming，parallel operations，synchronization，或者coroutines。

虽然说上述这些特征的缺失看似是很大的缺陷（比较两个character strings我居然还需要调用函数？），但是将一个语言的体量压缩到合适的大小有很大的好处。因为C语言本身很小，所以它可以很快的被学会。

很多年来，C的定义都是由第一版的the C programming language里的reference manual来定义的。1983年，ANSI建立了一个委员会来提供一个现代的，全面的C的定义。ANSI C作为C的定义，在1988年完成。这个定义的大多数特征都已经被现代编译器所支持。

ANSI C标准或者定义是基于原版的reference manual的。语言只有很少的改动；ANSI C的一个要求是之前已经存在了的programs需要仍然保持有效，或者说如果失效了，那么编译器也应该能够给出错误的提示。

对于大多数程序员来说，最重要的一个改变就是声明以及定义function的语法变了。现在一个function的声明可以包含这个function的arguments的描述；从而定义的语法改变了。这个新多出来的信息使得编译器能够更好的发现由不匹配的arguments导致的错误；在我们的使用经验里，这个改动是很有意义的。

还有一些小规模的改动。structure assignment和enumeration，现在正式成为了语言的一部分。floating-point computation可以在单一精度下完成了。算术的性质，特别是unsigned types，也被正式提出。preprocessor变得更加精细了。绝大部分的这些改变对于程序员来说基本没有影响。

ANSI C标准的第二个重要的贡献是提出了一个用来辅助C的library的定义。它将用来access operation system（比如，read或者write files）的functions，formatted input和output，memory allocation，string manipulation，like列入说明。一个standard headers的集合用来提供functions和data types声明的统一的access。用这个library来和操作系统进行交互的programs可以确保没有问题。这个library里的大部分内容也和UNIX系统里的标准I/O library兼容。

因为C提供的control structures和data types直接被绝大多数电脑支持，用来支持自带programs的run-time library十分小。标准library functions必须明确调用，这样才能避免在不需要的时候不会出现。这些标准库函数绝大多数用C来写（除去那些它们隐藏了的操作系统的细节），并且是可以跨平台移动的。

尽管C能够和绝大多数电脑兼容，它并不是为了任何特殊的电脑系统架构而设计的。只需要注意，我们就很容易写出可以移植的programs，也就是说，这些programs在任何硬件平台上都是一样的效果。ANSI C标准定义了这种可移植性，并且规定了一些常数来描述program在哪个机器上运行。

C并不是一个strongly-typed language，但是它进化了，它的type-checking被强化了。最初的C语言不喜欢，但是也不禁止，pointers和integers互相交换数据；但新的标准要求清晰的变量声明以及明确的转换定义。Compilers会警告所有的type errors，并且并不会对不兼容的数据类型进行转换，会直接报错。函数的声明也有类似的强化。但是，C语言还是保持了程序员自己知道自己在做什么事的原则；C语言只需要程序员清晰的把他们的意图说出来。

C语言，和任何其它语言一样，也有它的缺点。有一些operators有着错误的precedence；有一些部分的语法可以改得更好。但是，C语言已经被证明是一个极为高效且极具表达能力的语言，并且对于绝大多数应用领域都是如此。

这本书的结构如下。Chapter1是C核心部分的一个tutorial。目的是让读者尽快的开始，因为学习语言的最好方法就是尽快的用它来写program。这个tutorial认为读者对于编程有着最基本的认识；对于电脑，编译，或者类似n=n+1这种expression并没有任何的解释。这本书并不是为了变成一本数据结构或者算法的书，我们要集中注意力在语言本身。

Chpater2到Chapter6细节的讨论了C语言各个方面的内容，要比Chapter1更加详细，更加正式。Chapter2介绍基本的数据类型，operators和expressions。Chapter3介绍control flow：if-else，switch，while，for。Chapter4介绍了functions和program structure——external variables，scope规则，多个source files等——并且也接触到了preprocessor。Chapter5讨论了pointers和address算术。Chapter6介绍了structures和unions。

Chapter7介绍了标准库，其提供了和操作系统交互的interface。这个library是由ANSI标准定义的，并且应该能够支持所有可以支持C的机器，从而用它来操作input，output以及其它access操作系统的programs在任何机器上都应该是保持不变的，不需要随着机器的不同而更改代码。

Chapter8介绍了C programs和UNIX操作系统之间的一个interface，主要集中在input/output，file system和storage allocation。尽管这章的内容是为了UNIX系统设计的，用其他系统的程序员仍然可以在其中找到有用的内容，比如说一个标准库是如何被应用的。

Appendix A包含了一个语言reference manual。C语言语法和语义的官方文件应该是ANSI标准。然而ANSI标准是为了编译器设计的人写的。而此处的reference manual用更简洁的方式来介绍了语言的定义，舍去了那些标准化的内容。Appendix B是标准库的一个总结，也是为了程序员而不是编译器设计员而写的。Appendix C是现在这版C语言和原版的改动的总结。

ANSI标准拥有对C语言的最终解释权。


## Chapter 1: A Tutorial Introduction

让我们对C做一个快速的入门。我们的目的是在真实的programs显示这个语言的最核心的部分，而不要去管一些细枝末节的东西。此时，我们并不打算全面甚至不打算精确的介绍内容。我们只希望能尽快让读者写出有用的programs，为了做到这点，我们需要集中注意力在基础上：variables和constants，arithmetic，control flow，functions，以及input和output基础。我们这一章并不打算介绍写大型C程序所需要的内容，这些内容包括pointers，structures，C语言里丰富的operators，一些control flow statements，以及标准库。

我们上述这种介绍方式也有着缺点，最显著的就是C语言任何一部分的完整解释在这里都是没有的。而且这一章里的programs因为并没有用到C语言的全部特性（故意为之），所以并不一定是实现起来效率最高最简洁的。另一个缺点就是接下来的章节可能会重复这一章节的内容。

不管怎样，有经验的程序员能在这一章里学到有用的新思想，新手能够学会自己写一些有用的小的C程序。


### 1.1 Getting Started

学习一门新的编程语言的唯一方法就是用它来写程序。对所有语言来说，第一个程序都是print hello world!，在C语言里这样实现：

```c
# include <stdio.h>

main()
{
    printf("hello, world!\n");
}
```

如何运行这个C程序取决于其所在的系统。举例来说，如果在UNIX操作系统上，你需要创建一个.c的程序，比如说hello.c，然后用如下command来编译：

```c
cc hello.c
```

如果程序本身没有问题，那么编译的过程会安静的进行，然后生成一个executable文件，叫做a.out，用如下command来运行：

```c
a.out
```

他会输出hello world!。

在不同的操作系统里，上述过程会有所不同。

现在我们来对上面那段程序进行一些解释。一个C程序，不管它大小是多少，都有着functions和variables。一个function有着statements来指示需要做的计算，并且variables会记下计算过程中需要的值。C语言里的functions类似于Fortran语言里的subroutines和functions或者Pascal语言里的procedures和functions。在我们上面的程序里，我们有一个function叫做main。通常情况下，你可以随意命名你的程序，但是main是很特殊的：你的程序永远会从main function开始执行。这也表明任何C程序都必须在某个地方有一个main function。

main function经常会在内部调用其它的function来完成它的任务，有些是你自己定义的，有些是库里提供的。上面程序里的第一行#include <stdio.h>告诉编译器要将标准input/output库里的信息包含进来；这一行在很多C程序的开头都会出现。标准库会在Chapter7和Appendix B里被介绍。

一个在functions之间交换数据的方法就是一个function，提供一系列的values，叫做arguments，给被调用的另一个function。一个function的arguments放在这个function的名字后面，用括号括起来。在我们上面的程序里，main function并没有参数，于是在function名字后面是一对空的括号。

一个function里面的statements用花括号括起来，main function只有一句statement，printf("hello, world!\n")。一个function可以通过叫他的名字来调用，这个名字后面跟着用括号括起来的arguments，所以说这个statement里用到的function，printf，通过叫他的名字来调用，并且提供了括号括起来的arguments："hello, workd!\n"。printf function是一个库函数，用来print输出的。

一列用双引号包起来的characters，就像"hello, world!\n"，叫做一个character string或者string constant。

上面的character string里的\n是C语言里的newline character，表示换行。如果你用下面这个方式来写，C语言编译器会报错：

```c
printf("hello, world!
");
```

printf function并不会自动提供换行，所以说如果用多个printf function也是一样的效果：

```c
# include <stdio.h>

main()
{
    printf("hello, ");
    printf("world!");
    print("\n");
}
```

注意到，\n作为一个长度为2的character string，其仅仅表达一个character的意思（换行本身就是一个character），还有很多这样的，叫做escape sequence，这些是用来实现那些常用的但却很难打出来的或者不可见的characters，比如说\t表示制表，\b表示退一格，\"表示双引号本身，\\表示反斜杠本身，等等。在section2.3里有一个全面的表格来介绍escape sequence。

我们来看一下我们所学习的第一个C语言程序：

```c
#include <stdio.h>               # include information about standard library

main()                           # define a function named main that receives no argument values
{                                # statements of main are enclosed in braces
    printf("hello, world!\n");   # main calls library function printf to print this sequence of characters; \n represents the newline character
}
```


### 1.2 Variables and Arithmetic Expressions

我们下一个程序用公式$$C = (5/9)(F - 32)$$来输出下面的华氏度表格以及其摄氏度对应的温度：

```c
0    -17
20   -6
40   4
60   15
80   26
100  37
120  48
140  60
160  71
180  82
200  93
220  104
240  115
260  126
280  137
300  148
```

这个程序仍然只含有一个function，也就是main function，虽然比hello world那个程序长一些，但并不更加复杂。这个程序将会介绍一些新的内容，包括comment，declaration，variable，arithmetic expression，loop以及formatted output。

```c
# include <stdio.h>

/* print Fahrenheit-Celsius table
   for fahr = 0, 20, ..., 300 */

main()
{
    int fahr, celsius;
    int lower, upper, step;
    
    lower = 0;      /* lower limit of temperature table */
    upper = 300;    /* upper limit */
    step = 20;      /* step size */
    
    fahr = lower;
    while (fahr <= upper){
        celsius = 5 * (fahr-32) / 9;
        printf("%d\t%d\n", fahr, celsius);
        fahr = fahr + step;
    }
}
```
上面的两行
```c
/* print Fahrenheit-Celsius table
   for fahr = 0, 20, ..., 300 */
```
是comment，也就是简要的解释一下这个程序是做什么的。任何在/\*和\*/之间的内容都会被编译器忽略掉；它们可以让一个程序的可读性更强。comment可以出现在程序的任何位置。

在C语言里，所有的variables在使用之前都需要被声明，它们通常出现在function的开头，也就是在后面需要使用这些variables的statements之前。一个declaration会宣布variables的性质；它由一个type名字和一系列variables组成，比如：

```c
int fahr, celsius;
int lower, upper, step;
```

type int表明variables是integers（整数），而type float表明variables是浮点数，也就是带小数点的数。int和float类型的variables的范围都是由操作系统决定的。

C语言除了int和float type，还提供了很多其它基础的types：

```c
char    字符，一个字节
short   短整数
long    长整数
double  双精度浮点数
```

上述type的variables的大小仍然是由操作系统决定的。C语言里还有arrays，structures，unions，这些是由基础的types组合成的；还有pointers，它表示某个variable的地址；还有functions，会返回这些不同type的variables。

上述程序里，最开始是assignment：

```c
lower = 0;
upper = 300;
step = 20;
fahr = lower;
```

这会给variables初始值。每个statement都由分号结尾。

因为我们想要打印的表格每一列的数字都是由相同的计算公式算出来的，所以我们使用一个while循环来做：

```c
while (fahr <= upper){
    ...
}
```

while循环按照下面的逻辑来运行：首先括号内的条件要被判断。如果条件为真（fahr确实不比upper大），循环体会被执行。之后括号内的条件再次被判断。如果条件为真，那么循环体再次被执行。重复这两个步骤直到循环体内的条件不满足（fahr比upper大），此时循环停止，while循环后面紧接着的statement将会被执行。

while循环的循环体可以是一句或者多句statements，如果是多句statements，那么就一定要在花括号里面，如果是一句statement，那么可以不用花括号（也可以用）：

```c
while (i < j)
    i = 2 * i;
```

>C语言和python不一样，并不是形式型语言，也就是说缩进多少并不影响代码的正确性，但是为了好看，还是按照常用的缩进（4格）。同样的，每行只写一句statement也是为了代码好看，写多句也是可以的，但可阅读性会被降低，这也是一种习惯。还有个习惯就是对于operator，一般都会前后空一格，更加清晰，比如a + b。花括号的位置也不重要。

上述代码里最重要的部分就是celsius = 5 * (fahr - 32) / 9;这一句，之所以写成先乘以5再除以9而不是直接乘以5/9是因为在C语言里，如同很多其它语言一样，integer除法会将小数部分去掉，因为5和9都是整数，所以5/9的结果就变成了0，是不正确的。

在上述代码里，又用到了printf function。此处用到了printf function更多的功能。printf function本身是一个通用的output formatting function，我们将会在Chapter7里详细介绍。它的第一个argument是要打印的一个character string，用%来表示占位符，%后面的字母表示输出的格式，比如说%d表示按照整数来输出这个argument：

```c
printf("%d\t%d\n", fahr, celsius);
```

会以整数的格式输出两个arguments：fahr和celsius，并且中间隔一个制表符（也就是空四格）。

printf第一个argument里的每一个%都需要对应好后续的arguments，而且格式也需要注意。

但我们还需要注意，printf并不是C语言的一部分；C语言里并没有处理input或者output的工具。printf只是一个标准库里十分常见的function。printf function的行为是由ANSI标准规定的，所以任何按照ANSI标准编写的C程序如果使用printf function那么都是相同的。

为了集中注意力在C语言本身，我们直到Chapter7之前都不会过多深入input和output的工具。我们会在那里再说formatted input的内容。scanf function可以用来处理formatted input，其和printf相似，但是是处理input而不是output的。

对于上述代码，我们发现其还存在着某些问题。比如说，输出的结果并不是很好看，因为输出的数字并没有对齐。这很容易解决，我们只需要在printf function的%后面再加上一个数字表述输出宽度，那么这个占位符的输出就按照指定的宽度按右对齐输出，比如说：

```c
printf("%3d %6d\n", fahr, celsius);
```

那么结果将会是：

```c
  0    -17
 20     -6
 40      4
 60     15
 80     26
100     37
120     48
140     60
160     71
180     82
200     93
220    104
240    115
260    126
280    137
300    148
```

比格式问题更复杂的问题是，我们上面使用的都是整数运算，不够精确，比如0华氏度实际上是-17.8摄氏度，而不是-17摄氏度。为了获得更精确的结果，我们需要用浮点数而不是整数来计算。这需要对代码进行一些调整：

```c
#include <stdio.h>

/* print Fahrenheit-Celsius table
    for fahr = 0, 20, ..., 300; floating-point version */
main()
{
    float fahr, celsius;
    int lower, upper, step;
    
    lower = 0;           /* lower limit of temperature table */
    upper = 300;         /* upper limit */
    step = 20;           /* step size */
    
    fahr = lower;
    while (fahr <= upper) {
        celsius = (5.0 / 9.0) * (fahr - 32.0);
        printf("%3.0f %6.1f\n", fahr, celsius);
        fahr = fahr + step;
    }
}
```

上述代码和前面那个版本的差不多，除了fahr和celsius被声明为float，并且while循环里计算的公式写的更加自然。我们并不能使用(5 / 9)因为其算的结果是0。而带有小数点，即使是.0仍然说明其是float而不是integer，所以说5.0 / 9.0的结果并不会将小数点后的部分截掉，因为它是两个浮点数的计算结果。

如果一个arithmetic operator的operands是integers，那么这个operation就是一个integer operation，比如说5 / 9。如果一个arithmetic operator只要有一个floating-point operand，那么即使其它的还是integer operands，这些integers会被转换为floating-point数，然后再进行operation的计算，比如说5.0 / 9。在上面的代码里，即使我们写成fahr - 32，其中32是一个integer，因为fahr在声明的时候是float，所以在这个arithmetic operation，减法，里，32被转换为floating-point数。但为了观感，我们还是写成了farh - 32.0。

合适integers会被转换为floating point数的完整的细节的规定在Chapter2里会详细说明。对于现在来说，注意这个assignment：

```c
fahr = lower;
```

以及这个：

```c
while (farh <= upper)
```

它两也在operator里遇到了type不同的variables，而在这两种情况下，int仍然被转换为了float来进行operation。

在printf function里，%3.0f说明按照floating-point的形式输出，其宽度至少是3个字符，但不输出小数点以及小数点之后的内容。$6.1f表明按照floating-point的形式输出，输出宽度至少是6个字符，小数点后保持一位。上述代码执行之后的结果是：

```c
  0  -17.8
 20   -6.7
 40    4.4
 ...
```

我们还可以用更省略的方式来表示，%6f表示按照floating-point方式输出，宽度至少为6，不带小数点，%.2f表示按floating-point的方式输出，小数点后保留两位，但是宽度不做限制，%f只是表示按照floating-point的方式输出。

```c
%d     按照十进制整数的形式输出
%6d    按照十进制整数的形式输出，至少6个字符宽
%f     按照浮点数输出
%6f    按照浮点数输出，至少6个字符宽
%.2f   按照浮点数输出，小数点后保留两位
%6.2f  按照浮点数输出，至少6个字符宽，小数点后保留两位
```

printf里还会用%o表示八进制输出，%x表示十六进制输出，%c表示字符，%s表示字符串，%%表示%本身。


### 1.3 The For Statement

对于同一个任务会有很多各种各样的方法来写一个program实现它。对于我们这个温度转换的例子，我们尝试另一种方法。

```c
# include <stdio.h>

/* print Fahrenheit-Celsius table */
main()
{
    int fahr;
    
    for (farh = 0; farh <= 300; fahr = fahr + 20)
        printf("%3d %6.1f\n", fahr, (5.0 / 9.0) * (fahr - 32));
}
```

上述代码会生成一样的结果，但显然代码写的方式和之前的不一样。一个最主要的区别就是绝大多数的variables不需要了，只有fahr保留了下来，被声明为int。lower，upper和step仅仅作为常数在for语句里出现，而且Celsius度的计算作为printf的第三个argument出现，而不需要单独一个statement来表示了。

上述的第二个变化是C语言里很常见的一个rule：在任何地方如果允许用某一种type的值，那么你可以用很复杂的expression来替换，只需要这个expression的值也是这个type的就行了。在上述例子里，printf的第三个argument需要是一个float，所以任何floating-point expression都可以作为这个argument。

for statement是一个循环，是while循环的一个generalization。在for statement的括号里，有三个部分，用分号隔开。第一个部分，fahr = 0，是initialization，在for循环开始的时候作为最先执行的语句只执行一次。然后第二个部分是test或者condition，其是用来控制循环的，fahr <= 300。这个condition将会被测试，如果满足，那么for循环的主体，也就是这个例子里的printf语句就会被执行。再之后，第三个部分，increment step，fahr = fahr + 20被执行。再之后，第二部分，condition，再次被测试。以此循环。而在第二部分condition不满足的时候，for循环停止。

for循环正如while循环那样，循环主体可以是单独一个statement或者是用花括号括起来的一组statements。而for循环的三个部分，initialization，condition和increment可以是任意expression。

while循环和for循环之间的选择是随意的，取决于你喜欢哪一种。for循环对于initialization和increment都是简单的statements而且有联系的时候比较适合。for循环要比while循环更加简洁，并且将循环condition和initialization，increment都写在了一起。


### 1.4 Symbolic Constants

这是我们离开温度转换例子前最后一个重点了。在一个program里直接用数字，比如说上面例子里的300，20等实际上是个不好的习惯；它对于之后要看代码的人是不友好的，而且不利于代码的系统性修改。一个解决办法是，给这些数值有意义的名字。一个#define行会定义一个symbolic name或者symbolic constant成为一个特定的字符串：

```c
#define name replacement list
```

在此之后，*name*出现的任何地方（这个*name*不能带引号，也不能是其它名字的一部分）将会被相对应的*replacement text*所取代。一个*name*有着和*variable name*一样的命名规则：一串字母和数字并且要以字母开头。这个*replacement text*可以是任何字符串，不仅限于数字。

```c
#include <stdio.h>

#define LOWER 0    /* lower limit of table */
#define UPPER 300  /* upper limit */
#define STEP  20   /* step size */

/* print Fahrenheit-Celsius table */
main()
{
    int fahr;
    
    for (fahr = LOWER; fahr <= UPPER; fahr = fahr + STEP)
        printf("%3d %6.1f\n", fahr, (5.0 / 9.0) * (fahr - 32));
}
```

LOWER，UPPER和STEP都是symbolic constants，它们并不是variables，所以并不需要声明。symbolic constants的名字经常是大写的，从而它们可以轻易的和那些通常小写的variable名字区分开。注意#define语句的结尾并没有分号。


### 1.5 Character Input and Output

我们来考虑一簇用来处理字符数据的programs。你将会发现很多的这类programs只是我们这里所提到的prototype的扩展或者改进。

标准库所支持的input和output模型是十分简单的。text input或者output，不管是从哪来的或者到哪去，都被当作字符流来处理。一个text stream是一系列按行排列的字符；每一行都包含0个或更多的字符，并且以换行符结尾。

标准库为一次read或者write一个字符提供了几个functions，而在这个functions中getchar和putchar又是最简单的。每次被调用的时候，getchar都会读取text stream里的下一个input字符，并且返回它。

```c
c = getchar();
```

variable c就包含了下一个input character。字符一般情况下都来自键盘输入；而从文件来的input将会在Chapter7里被提到。

function putchar在每次被调用时打印一个character：

```c
putchar(c)
```

上述代码会将variable c里包含的内容以一个字符的形式打印出来，通常是直接在屏幕上显示。putchar和printf经常交叉使用。


#### 1.5.1 File Copying

用getchar和putchar，你可以写出很多有用的代码，甚至都不需要知道任何有关input和output的事情。最简单的例子就是一个将input挨个字符复制到output的代码：

伪代码如下：

```c
read a character
    while (character is not end-of-file indicator)
        output the character just read
        read a character
```

将其用c语言来实现：

```c
#include <stdio.h>

/* copy input to output; 1st version */
main()
{
    int c;
    
    c = getchar();
    while (c != EOF) {
        putchar(c);
        c = getchar();
    }
}
```

上面代码里的!=表示不等于。

如果我们上述代码用的variable c声明为char类型，更加符合我们之前对任务以及对getchar，putchar function的描述。但是这里用的是int类型，实际上任何整数类型都可以。这里用int而不用char实际上有很重要的原因。

原因就是我们需要将input的结尾和有效的数据分开。解决办法是getchar会返回一个特殊的值用来表示不再有更多的数据了，而这个值也不能和任意其它的character混淆。这个值叫做EOF，end of file的简写。我们要将c声明为一个范围足够大的type，其能够包含的下getchar返回的任何值。而我们如果用char类型的话，其就包含不了EOF这个值，从而会出问题，所以我们用了int。

>即使c声明为int也没关系，读入的字符的值会存入这个variable里，然后在putchar的时候，因为putchar会将任何variable里存的值都当作一个字符来对待，所以输出还是字符，即使这个variable被声明为不是char类型。

EOF是<stdio.h>里定义的一个整数，它的值只要和任何的char类型的值不一样就行。通过用#define来为EOF定义一个symbolic constant，我们在program里就不需要每次都明确的写出具体的数值了。

上述的代码还可以写的更精炼一点。在C语言里，任何的assignment，比如说

```c
c = getchar()
```

是一个expression而且有一个值，这个值就是左边在assignment之后的值。这个性质表明一个assignment可以是一个更大的expression的一部分。所以说我们可以将c的assignment放在while loop的test里：

```c
#include <stdio.h>

/* copy input to output; 2nd version */
main()
{
    int c;
    
    while ((c = getchar()) != EOF)
        putchar(c);
}
```

上述的while语句，获得了一个character，将它assign给c，然后再测试这个character是不是EOF。如果不是，while loop的主体就被执行，输出那个character。之后while loop再重复。当输入的character是EOF时，while loop停止，main function结束。

上述的版本使得代码更加精炼，更便于阅读。

上述代码里assignment语句外面的那层括号是有必要的。因为!=的优先级要高于assignment符号=，所以如果没有外面那层括号，那么!=会先执行，之后再执行=，所以说:

```c
c = getchar() != EOF
```

等价于

```c
c = (getchar() != EOF)
```

因为c声明的是int类型，所以上述语句会导致c是0或者1，并不是我们想要的结果。


#### 1.5.2 Character Counting

我们下一个program用来计算字符数，和上面那个copy program类似：

```c
#include <stdio.h>

/* count characters in input; 1st version */
main()
{
    long nc;
    
    nc = 0;
    while (getchar() != EOF)
        ++nc;
    printf("%ld\n", nc)
}
```

我们上述的program里的statement：

```c
++nc
```

用了一个新的operator，++，指的是增加1。它等价于

```c
nc = nc + 1
```

但是++nc更加简洁并且更高效。还有个类似的operator，--，表示减小1。而operators ++和--即可以用在前面，也可以用在后面，++nc和nc++都是一个意思，但是这两种expression的数值是不一样的，我们在chapter2里会提到。但是++nc和nc++都是将nc的值加上1。

上述character counting program用一个long variable来累计计算值，而不是int。long variable至少有32bits。因为int variable是16bits的，最大数值就是32767，作为计数器在这种情况下很容易就超出范围了。而且在printf function里使用%ld也告诉了printf function以long integer的形式输出这个值。

我们还可以用double（double precision float）类型来表示更大的数值：

```c
#include <stdio.h>

/* count characters in input; 2nd version */
main()
{
    double nc;
    
    for (nc = 0; getchar() != EOF; ++nc)
        ;
    printf("%.0f\n", nc);
}
```

printf function表示float和double都是%f；而%.0f就不会输出小数点后面的部分了（虽然本来也就是0）。

上述for loop的循环体是空的，因为所有的任务在test condition和increment部分就做完了。但是C语言的语法要求for statement必须有一个循环体。独立的分号;，叫做null statement，就是为了满足这个条件的。我们将它单独成行，为了更便于辨识。

在我们离开character counting program之前，注意到如果输入不含字符，那么while或者for的test对于第一次getchar的调用不满足condition，从而循环体不执行，program会输出0，是正确的。这一点很重要。while和for循环一个比较优越的点在于它们在循环体执行前就会进行test condition的测试。如果条件一开始就不满足，那就不执行循环体了。programs要对于零长输入也表现得很聪明。while和for statements就做得很好，对于这种特殊的边界情况也能给出合理的值。


#### 1.5.3 Line Counting

下一个program用来计算输入的行数。正如我们之前所说的，标准库确保输入的text流以一系列的行的形式出现，每一行由一个换行符结尾。因此，计算行数也就等价于计算换行符的个数：

```c
#include <stdio.h>

/* count lines in input */
main()
{
    int c, nl;
    
    nl = ;
    while ((c = getchar()) != EOF)
        if (c == '\n')
            ++nl;
    printf("%d\n", nl);
}
```

while loop的主体包含了一个if，它是用来控制什么时候做++nl的。if statement会检测用括号括起来的condition，如果condition满足，就会执行if statement内的statement（或者用花括号括起来的statements）。

注意到，虽然while loop的循环体里有两行而不是一行，但这两行还是被认为是一个statement，所以并不需要花括号。

==在C语言里表示等于（如同Pascal里的=或者Fortran里的.EQ），这是用来与assignment进行区分的。

一个在单引号之间的字符表示的是一个整数值，而这个整数值是这个字符在机器的字符集里对应的值（比如说ASCII）。这叫做一个character constant。比如说，'A'是一个character constant；在ASCII字符集里它的值是65，也就是机器内部表示character A所用的值。但是显然用不一样的字符集，'A'的值就可能会不一样。

而escape sequences，也就是类似于'\n'的，它也是character constant，'\n'表示换行符的值，在ASCII里，是10。我们需要注意，'\n'是一个单个的character，在expression里它只是一个整数；换句话说，'\n'是一个string constant，但只含有一个character。string和character的关系我们在chapter2里会说。


#### 1.5.4 Word Counting

一个word被定义为一个不包含空格、tab或者换行符的字符串。下面是个简单的例子：

```c
#include <stdio.h>

#define IN  1   /* inside a word */
#define OUT 0  /* outside a word */

/* count lines, words, and characters in input */
main()
{
    int c, nl, nw, nc, state;
    
    state = OUT;
    nl = nw = nc = 0;
    while ((c = getchar()) != EOF){
        ++nc;
        if (c == '\n')
            ++nl;
        if (c == ' ' || c == '\n' || c == '\t')
            state = OUT;
        else if (state == OUT){
            state = IN;
            ++nw;
        }
    }
    printf("%d %d %d\n", nl, nw, nc);
}
```

每次program遇到一个word的第一个字符，它就会多算一次。state这个variable记录这个program目前是在一个word内还是外；一开始的时候是不在的，所以初始化为OUT。我们用symbolic constant来将IN和OUT指定为0和1，这样会有更好的阅读效果。对于很大的programs，我们会发现使用symbolic constants来表示这些常数会使得修改代码变得简单很多。

```c
nl = nw = nc = 0;
```

这一行将三个variables都assign为0。这种现象是因为assignment是一个有值的expression而且assignment是从右往左进行的。上述代码和下面的等价：

```c
nl = (nw = (nc = 0));
```

在C语言里，||表示OR。对于AND，用&&表示，它的优先级高于||。用||或者&&连接的expressions是从左到右衡量的，而且一旦True或者False的值确定了，就不再继续算了（比如说全是||连接的expressions，只要有一个True的，后续不用衡量了，结果是True）。

上述代码里还出现了else，它在if statement的condition是False的时候提供了可选择的操作。普遍的形式是这样的：

```c
if (expression)
    statement;
else 
    statement;
```

在if-else结构里，只有一个statement会被执行。每个statement都可以是单个的statement，也可以是用花括号括起来的多个statements。在上面的例子里，else后面跟的是另一个if statement，而这个if statement用花括号括起来了两个statements作为自己的主体。所以说其实对于这个else，它的statement就是这个if statement，所以整个if statement看成一个整体还是只是一个statement，所以并不需要花括号。


### 1.6 Arrays

让我们来写一个program，用来计算每个数字、每个white space characters（blank，tab和newline）、以及所有其它字符组成的整体，这三个部分所出现的次数。也就是说，我们的input一共有12各种类（数字10种，后面2种）。我们用一个array来统计每个数字出现的次数，要比使用十个单独的variables来统计要容易很多。

```c
#include <stdio.h>

/* count digits, white space, others */
main()
{
    int c, i, nwhite, nother;
    int ndigit[10];
    
    nwhite = nother = 0;
    for (i = 0; i < 10; ++i)
        ndigit[i] = 0;
    
    while ((c = getchar()) != EOF)
        if (c >= '0' && c <= '9')
            ++ndigit[c-'0'];
        else if (c == ' ' || c == '\n' || c == '\t')
            ++nwhite;
        else
            ++nother;
    printf("digits =");
    for (i = 0; i < 10; ++i)
        printf("%d", ndigit[i]);
    printf(", white space = %d, other = %d\n",
                nwhite, nother);
}
```

上述代码里的声明

```c
int ndigit[10];
```

声明ndigit是一个由10个int类型组成的array。在C语言里，array的下标从0开始计数，所以这个array的elements是ndigit$$\[0\]$$，ndigit$$\[1\]$$，ndigit$$\[2\]$$，ndigit$$\[3\]$$，ndigit$$\[4\]$$，ndigit$$\[5\]$$，ndigit$$\[6\]$$，ndigit$$\[7\]$$，ndigit$$\[8\]$$，ndigit$$\[9\]$$。这一点在初始化array的for loop里也能看得出来。

下标可以是任何integer expression，包括如i这样的integer variable，也包括integer constant。

上述的代码基于表示digits的字符的特殊的性质。比如说，

```c
if (c >= '0' && c <= '9')
```

就可以确定c是否是一个表示数字的字符。如果是的话，那么它的值应该是c - '0'。而这种做法仅仅对于'0', '1', ... , '9'有着连续的值的时候才可行。幸运的是，这对于绝大多数character set来说都是可行的。

根据定义，char类型其实是由整数来表示的，所以char variables和char constants在代数expressions里和int类型没啥区别。这是很自然也是很方便的；比如c-'0'是一个值在0到9之间的integer expression，从而是ndigit的一个有效的下标。

在上述代码里，决定一个character是digits，还是white spaces还是其它的characters是用下面来实现的：

```c
if (c >= '0' && c <= '9')
    ++ndigit[c-'0'];
else if (c == ' ' || c == '\n' || c == '\t')
    ++nwhite;
else
    ++nother;
```

这样一种写法：

```c
if (condition)
    statement;
else if (condition);
    statement;
else
    statement;
```

在程序需要表示多条路的选择的时候经常出现。conditions按照从上到下的顺序进行衡量，直到某个condition满足；然后相对应的statement被执行，然后整个结构结束。如果没有任何一个条件满足，最后一个else对应的statement就会被执行。如果最后一个else和它的statement被省略了，那么就没有任何操作会被执行。在初始的if和最后的else之间可以有任意多个else if。

我们建议使用上面这种写的方式，因为如果在每个else if的地方，都换一行并且缩进一个tab来写if，对于很长的这种结构，缩进就会太多。

switch statement将会在chapter4里被提到，它提供了另一种方式来处理有多条选择的情况。


### 1.7 Functions

在C语言里，一个function等价于Fortran语言里的一个subroutine或者function，或者是Pascal里的一个procedure或者function。一个function提供了一个简单的方法将一些计算包装起来，之后我们再使用的时候就不必关心内部的计算细节了。拥有正确设计的functions，我们甚至可以忽略这个function内部做了什么，只需要知道这个function能做什么就可以了。C语言使得使用function很简单高效。

到现在为止，我们仅仅用过那些提供给我们的functions，比如说printf，getchar，putchar等；现在我们可以自己写几个functions了。既然C语言里没有提供求乘方的function，那我们可以通过写一个叫做power(m,n)的function来实现乘法功能，从而解释function的工作机理。power(m,n)是求m的n次方，m和n都是正整数。（标准库里有pow(x,y) function，计算的是$$x^y$$）

下面就是power function和使用它的main function构成的program：

```c
#include <stdio.h>

int power(int m, int n);

/* test power function */
main()
{
    int i;
    
    for (i = 0; i < 10; ++i)
        printf("%d %d %d\n", i, power(2, i), power(-3, i));
    return 0;
}

/* power: raise base to n-th power; n >=0 */
int power(int base, int n)
{
    int i, p；
    
    p = 1;
    for (i = 1; i <= n; ++i)
        p = p * base;
    return p;
}
```

一个function定义有如下的格式：

```c
return-type function-name(parameter declarations, if any)
{
    declarations
    statements
}
```

function的定义可以以任意的顺序出现，也可以在一个source file或者多个source file里，但是一个function并不能被分割到多个files里进行定义。如果一个program分散在多个source files里，你可能需要在编译和加载的时候多注意，但这是操作系统的问题，不是语言本身的问题。

上述定义的power function在main function里被调用了两次：

```c
printf("%d %d %d\n", i, power(2, i), power(-3, i));
```

每一次对power function的调用都要传递两个arguments，而会输出一个integer。

上述代码的第一行

```c
int power(int base, int n)
```

声明了参数类型和名字，以及function会返回什么类型的结果。power function的参数的名字是local的，也就是说它只能在power function内部使用，在power function之外就看不到了。这也是说，在power function外我们也可以用同样名字的variables而不会引起问题。我们可以看到，在上述代码里，power function和main function里都有i，但没有引起冲突。

对于一个function的变量，我们用parameter来命名，当然formal argument和actual argument也是一样的意思。

上述代码里，power function用return来返回计算的值。return后面可以接任意的expression。

一个function不一定需要return一个值；可以仅仅用一个return，用来表示这个function执行完毕了。调用一个function的function也可以忽略被调用的function返回的值。

注意到，main function的结尾处也有个return statement。main function也可以和其它function一样，给它的caller返回一个值，而它的caller则是这个program所在的操作系统环境。对于main function，return 0;表示正常的结束；return一个非零的值则表示出现异常。我们之前的代码都省略了main function的return statement，但实际上应该要有，这样的代码更便于维护，而且main function应该向它的caller传递执行的信息。

在main function之前的声明

```c
int power(int base, int n);
```

说的是power是一个function，期待两个int arguments，并且返回一个int。这个声明，叫做function prototype，需要和power function的定义以及使用相匹配。如果一个function的定义或者任何使用和它的prototype不匹配，就会报错。但prototype的parameter名字不需要匹配。而且，在一个function prototype里，parameter名字实际上是可选的，所以说：

```c
int power(int, int)
```

代替上面的那行也是一样的。但是在prototype里写上恰当的名字可以使得代码可读性增强。

对于过去版本的function定义，在ANSI C里是这样的：

```c
/* power: raise base to n-th power; n >= 0 */
/*         (old-version)                   */
power(base, n)
int base, n;
{
    int i, p;
    
    p = 1;
    for (i = 1; i <= n; ++i)
        p = p * base;
    return p;
}
```

可以看到，变量名在函数名后的括号里被确定，并且在花括号之前，变量的类型就要被声明。而没有被声明的变量一律以int类型对待。

并且代码开头power function的声明在旧版本里变成这样：

```c
int power()
```

旧版本会被取代，所以强烈建议使用新版本。



### 1.8 Arguments - Call by Value

C语言的functions的一个方面可能很多人不了解。在C语言里，所有的function arguments都是通过value传递的。这个的意思是原始的variables将值给暂时的variables，然后再作为arguments给函数。也就是说函数的arguments是其对应的variables的副本。这和通过引用的方式来获取arguments的值有很大的不同，在那种情况下，被调用的函数是可以获取arguments对应的原始的variables的，而不是只能获取一个副本。

通过值来调用实际上是有很多好处的。它可以使得program更加简单，因为arguments就可以被看成在被调用的function里新构造的local variables。比如说，下面这个power function就使用了这个性质：

```c
/* power: raise base to n-th power; n >= 0; version 2 */
int power(int base, int n)
{
    int p;
    
    for (p = 1; n > 0; --n)
        p = p * base
    return p;
}
```

在上述代码里，argument n就被用作了一个暂时的variable，并且逐步减少直到变成0；从而就不再需要另一个variable i来计数了。而在power function内对n进行的操作，对于power function所需要的这个argument n对应的外部的variable是没有任何影响的。

当有必要的时候，让function能够改变其argument对应的外部variable的值也是可以的。这个时候caller需要提供这个argument对应的variable的address（也就是这个variable的pointer），然后被调用的function一定要声明这个parameter是一个pointer，并采用通过访问address的形式获取这个值。我们将会在chapter5里介绍pointer。

但是对于array来说情况又不一样了。当一个array的名字被当作一个argument时，传递给function的值就是这个array开头的address，并没有任何array内部的值会传递给这个function。通过下标，function可以获取这个array的值。这是我们下一节马上要说的内容。


### 1.9 Character Arrays

C语言里最常见的array就是字符组成的array。为了解释character array的使用以及functions如何操作它们，我们来写一个program用来读取一个text lines的集合并且打印出最长的那个。

伪代码如下：

```c
while (there is another line)
    if (it is longer than the previous longest)
        (save it)
        (save its length)
print longest line
```

上述的伪代码将program自然的分割成了几部分。一部分获取新行，一部分储存，剩下的控制过程。

让我们先写一个独立的function getline，用来获取input的下一行。getline需要对于end of file返回一个信号；或者说返回line的长度，这样返回的是0就是end of file了。

当我们找到了一行，比之前找到的最长的行要长的时候，它需要被存下来。从而我们需要第二个function copy，用来将某行复制到一个地方存下来。

最后我们需要一个main program来控制getline和copy。

```c
#include <stdio.h>
#define MAXLINE 1000   /* maximum input line length */

int getline(char line[], int maxline);
void copy(char to[], char from[]);

/* print the longest input line */
main()
{
    int len;                /* current line length */
    int max;                /* maximum length seen so far */
    char line[MAXLINE];     /* current input line */
    char longest[MAXLINE];  /* longest line saved here */
    
    max = 0;
    while ((len = getline(line, MAXLINE) > 0)
        if (len > max){
            max = len;
            copy(longest, line);
        }
    if (max > 0)            /* there is a line */
        printf("%s", longest);
    return 0;
}

/* getline: read a line into s, return length */
int getline(char s[], int lim)
{
    int c, i;
    
    for (i = 0; i < lim - 1 && (c = getchar()) != EOF && c != '\n'; ++i)
        s[i] = c;
    if (c == '\n'){
        s[i] = c;
        ++i;
    }
    s[i] = '\0';
    return i;
}

/* copy: copy 'from' into 'to'; assume to is big enough */
void copy(char to[], char from[])
{
    int i;
    
    i = 0;
    while ((to[i] = from[i]) != '\0')
        ++i
}
```


































        
        


                


















































---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*
