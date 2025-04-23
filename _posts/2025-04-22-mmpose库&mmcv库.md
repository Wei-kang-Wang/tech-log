---
layout: post
comments: false
title: "mmpose库&mmcv库"
date: 2025-04-22 01:09:00
tags: language

---
<!--more-->

{: class="table-of-content"}
* TOC
{:toc}
---

<!--more-->


## **\[ECCV 2024 \]**SCAPE: A Simple and Strong Category-Agnostic Pose Estimator官方代码

以ECCV 2024里的论文SCAPE: A Simple and Strong Category-Agnostic Pose Estimator的官方代码为例，介绍一下`mmpose`和`mmcv`这两个库在实践上的应用。代码仓库地址：https://github.com/tiny-smart/SCAPE/tree/main

`mmpose 0.29.0`和`mmcv 1.6.2`为例。

### 数据集

数据集相关代码在`scape/datasets`下，包含`datasets, pipelines`两个文件夹，以及`__init__.py, builder.py`两个函数。

#### `__init__.py`

`__init__.py`从`builder.py`以及两个文件夹下`import`所有内容，从而可以直接使用`from datasets import XX`来`import`这些内容。

#### `builder.py`

`builder.py`里定义了`build_dataset()`函数，并且定义了`_check_valid()`和`_concat_cfg`两个辅助函数。`build_dataset`接收`cfg, default_args`作为输入，其作用是使用输入的cfg文件构建数据集（`default_args`一般都是`None`），使用的cfg文件是`configs/mp100/scape/scape_split1.py`里的`data.train`，是一个字典，在这个cfg文件里，`type`是一个字符串，`'TransformerPoseDataset'`。`build_dataset`先使用`_check_valid`函数判断`data.train.cfg_train`这个字典下的`'num_joints', 'dataset_channel'`是否是`list/tuple`，如果是的话，只取它们的第一项返回，此处的cfg文件在`_check_valid`函数前后不会改变。`build_dataset`函数最后使用`build_from_cfg(cfg, DATASETS, default_args)`返回数据集。`build_from_cfg`函数由`from mmcv.utils import build_from_cfg`引进，`DATASET`由`from mmpose.datasets.builder import DATASETS`引进。

在`mmpose.datasets.builder.py`里，定义了`DATASETS`：`DATASETS = Registry('dataset')`，其中`Registry`是由`from mmcv.utils import Registry`引进。`Registry`是在`mmcv.utils.registry.py`里定义的一个类，如其名所示，`Registry`的作用就是为一类东西总结为一个注册表，比如说models，datasets等，其常用的方法是`build`和`register_modules()`，其中后者通常作为装饰器使用，将某些新定义的对象加入该`Registry`作为children管理，而`build`函数则是使用`Registry`初始化的时候接收的`build_func`参数来构建实例。如下是个简单的例子：

```python
MODELS = Registry('models')
@MODELS.register_module()
class ResNet:
    pass
resnet = MODELS.build(dict(type='ResNet'))

@MODELS.register_module()
def resnet50():
    pass
resnet = MODELS.build(dict(type='resnet50'))
```

`build_from_cfg`同样也定义在`mmcv.utils.registry.py`里。其源码不长：

```python
def build_from_cfg(cfg: Dict,
                   registry: 'Registry',
                   default_args: Optional[Dict] = None) -> Any:
    """Build a module from config dict when it is a class configuration, or
    call a function from config dict when it is a function configuration.

    Example:
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> resnet = build_from_cfg(dict(type='Resnet'), MODELS)
        >>> # Returns an instantiated object
        >>> @MODELS.register_module()
        >>> def resnet50():
        >>>     pass
        >>> resnet = build_from_cfg(dict(type='resnet50'), MODELS)
        >>> # Return a result of the calling function

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f'but got {cfg}\n{default_args}')
    if not isinstance(registry, Registry):
        raise TypeError('registry must be an mmcv.Registry object, '
                        f'but got {type(registry)}')
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError('default_args must be a dict or None, '
                        f'but got {type(default_args)}')

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')
    try:
        return obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f'{obj_cls.__name__}: {e}')
```

`build_dataset`函数最后使用`build_from_cfg(cfg, DATASETS, default_args)`返回数据集，cfg是`data.train`，从而返回的是`TransformerPoseDataset(**cfg)`。

#### `datasets`文件夹

`scape/datasets/datasets`下由一个文件夹`mp100`和一个`__init__.py`组成，后者从`mp100`里`import FewShotKeypointDataset, FewShotBaseDataset, TransformerBaseDataset, TransformerPoseDataset`从而可以直接使用`from datasets.datasets import xx`来调用这些定义数据集的类。`mp100`文件夹下也有一个`__init__.py`文件，以及`fewshot_base_dataset.py, fewshot_dataset.py, transformer_base_dataset.py, transformer_dataset.py`四个文件分别定义数据集。`mp100/__init__.py`分别从这四个文件里`import`对应的数据集类：`from .fewshot_dataset import FewShotKeypointDataset, from .fewshot_base_dataset import FewShotBaseDataset, from .transformer_dataset import TransformerPoseDataset, from .transformer_base_dataset import TransformerBaseDataset`

##### `datasets/mp100`文件夹

下面来挨个看`datasets/mp100`文件夹下四个定义数据集的scripts。

首先来看`transformer_base_dataset.py`。该文件里定义了一个`TransformerBaseDataset`类，以及一个字典`cata_pair`，定义了点的连接情况。该`TransformerBaseDataset`类由装饰器`@DATASETS.register_module()`控制，而在`mmpose.datasets.builder.py`里，定义了`DATASETS`：`DATASETS = Registry('dataset')`。该`TransformerBaseDataset`类继承自`torch.utils.data.Dataset`。




`mmpose.models.builder.py`里定义了`build_posenet()`函数，其用到了`mmcv.utils.Registry`的`build()`来从输入的`cfg`文件构建模型。
