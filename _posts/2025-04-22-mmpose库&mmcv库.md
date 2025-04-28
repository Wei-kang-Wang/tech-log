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

`mmpose`库是港中文MMLab开发的一个用于pose estimation的库，针对众多pose estimation的任务，包括2D关键点检测、2D物体检测、2D mask检测等等。其和另一个库`mmcv`紧密联系。这两个库建立在PyTorch基础上，定义了一系列更上层的包装，来使得pose estimation相关的任务可以更快速的部署和调试。

以`mmpose 0.29.0`和`mmcv 1.6.2`为例来介绍这两个库。

## 一些general的上层设计：`Config`类、`Registry`类和`build_from_cfg()`函数

`mmcv`使用大量的configuration文件来创建模块，或者调用函数，其使用`Config`类来管理这些configuration文件，使用`Registry`类来分类管理类似功能的模块（比如models，datasets），而`build_from_cfg`是最常见的从configuration文件来实例化模块的函数。所以这三个是`mmcv`库的核心。

### `Config`类

`Config`类是用来管理config文件的，其支持从多种类型文件里读取configuration信息，包括：`.py`、`.json`和`.yaml`。其返回的是一个字典。

下面是一个例子。假设我们定义好了一个`test.py`文件如下：

```python
a = 1
b = dict(b1=[0, 1, 2], b2=None)
c = (1, 2)
d = 'string'
```

使用`Config`类来加载`test.py`里的configuration信息：

```shell
>>> cfg = Config.fromfile('test.py')
>>> print(cfg)
>>> dict(a=1, b=dict(b1=[0, 1, 2], b2=None), c=(1, 2), d='string')
```



### `Registry`类

`Registry`类是`mmcv`库里最重要、最基础的一个设计，其目的是使用一个`Registry`实例去管理具有相似功能的不同的模块，比如说`Registry("models")`来去管理各自不同的network，`Registry("dataset")`去管理不同的数据集等。

总的来说，`Registry`是一个将每个`class`或者`function`映射到某个`string`的映射。一般来说，我们都通过某个`Registry`实例来管理某一类型具有相似功能的`class`或者`function`，通过`Registry`，我们就可以使用每个`class`或者`function`对应的`string`来找到它们，并且实例化该`class`或者调用该`function`。

使用`Registry`一般包含三步：
* 自定义一个`my_build`函数，`Registry`在实例化的时候，除了需要提供一个名字，还可以选择为`build_func`可选参数提供一个新的值，比如`my_registry = Registry("example", build_func=my_build)`，default的就是`build_from_cfg()`。在`Registry`的`build`方法里，`build_func`用于实例化`Registry`里的每个`class`，或者为`function`提供参数。
* 创建实例，比如`my_registry = Registry("example")
* 使用`my_registry`去管理一系列模块（使用装饰器的格式，下面会介绍）

以一个简单的例子来介绍。假设我们想要管理一系列的Dataset Converters，它们的功能都是对于不同的数据集进行数据操作，我们希望通过一个`Registry`来统一管理它们。

首先，创建一个文件夹`my_dataset_converters`，在该文件夹下，首先创建一个`builder.py`文件：

```python
from mmcv.utils import Registry
# create a registry for converters
CONVERTERS = Registry('converters')
```

然后我们就可以在该文件夹下创建`class`或者`function`了，假设我们创建一个`converter1.py`文件，里面包含一个`Converter1`类，创建一个`converter2.py`文件，里面包含一个`converter2`函数：

```python
## converter1.py
from .builder import CONVERTERS

# use the registry to manage the module
@CONVERTERS.register_module()
class Converter1(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

# converter2.py
from .builder import CONVERTERS
from .converter1 import Converter1

# 使用注册器管理模块
@CONVERTERS.register_module()
def converter2(a, b)
    return Converter1(a, b)
```

使用`Registry`类来管理模块的关键就在于，在创建模块的时候（模块可以是`class`或者`function`），使用上述装饰器的方式，使用创建的`Registry`实例里的`register_module()`方法来为新定义的`class`或者`function`在该`Registry`实例里进行注册。通过该装饰器的方式，上述定义的`Conveter1`类和`converter2`函数，就分别和某个`string`建立起了映射关系：`'Converter1' -> <class 'Converter1'>, 'converter2' -> <function 'converter2'>`，以后就可以使用`CONVERTERS`里的`<class 'Converter1'>`和`<function 'converter2'>`来分别reference它们了。

> 需要注意的是，上述方式定义的模块（`Converter1`类和`converter2`函数），只有当该模块所在的文件被imported的时候，才会触发`CONVERTERS`对这模块的注册。这就是为什么很多时候，在文件夹下，还需要一个`__init__`文件来`import`这些模块（这里有更详细的解释：https://github.com/open-mmlab/mmdetection/issues/5974）。比如说，在上述`my_dataset_converters`文件夹下，定义一个`__init__`函数如下：
> ```python
> from converter1 import Converter1
> from converter2 import converter2
> ```
> 如此操作之后，`Converter1`和`converter2`就已经被注册在`CONVERTERS`里了。

在模块被成功注册之后，我们就可以来创建它们了：

```python
# `CONVERTERS`并没有给build_func参数传递值，从而使用的是default的build_from_cfg，
# 所以使用下属字典类型的converter1_cfg和converter2_cfg可以用来创建converter1实例和获取converter2的返回值（注意converter2是个函数）

converter1_cfg = dict(type='Converter1', a=a_value, b=b_value)
converter2_cfg = dict(type='converter2', a=a_value, b=b_value)
converter1 = CONVERTERS.build(converter1_cfg)
# returns the calling result
result = CONVERTERS.build(converter2_cfg)
```

实际上，一个`Registry`类被实例化的时候，有四个可选参数：`name`、`build_func`、`parent`和`scope`。`name`就是给该实例取的名字，`build_func`是`Registry`的`build`方法用来创建模块时候使用的函数，default的是`build_from_cfg`，下面来看一下使用自定义的函数值传递给`build_func`的例子：

```python
from mmcv.utils import Registry

# create a build function
def build_converter(cfg, registry, *args, **kwargs):
    cfg_ = cfg.copy()
    converter_type = cfg_.pop('type')
    if converter_type not in registry:
        raise KeyError(f'Unrecognized converter type {converter_type}')
    else:
        converter_cls = registry.get(converter_type)

    converter = converter_cls(*args, **kwargs, **cfg_)
    return converter

# create a registry for converters and pass ``build_converter`` function
CONVERTERS = Registry('converter', build_func=build_converter)
```

> 在该例子里，这个新定义的`build_converter`实际上和default的`build_from_cfg`功能类似。在绝大多数情况下，`build_from_cfg`就够用了。`mmcv`还提供了一个`build_model_from_cfg`函数，可以用来对于`PyTorch`的`nn.Sequential`里的`modules`进行创建，也是很常用的。

`Registry`还有个经常使用的方法：`get`，其


如下是`Registry`的源码（定义在`mmcv.utils.registry.py`里）：

```python
class Registry:
    """A registry to map strings to classes or functions.

    Registered object could be built from registry. Meanwhile, registered
    functions could be called from registry.

    Example:
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> resnet = MODELS.build(dict(type='ResNet'))
        >>> @MODELS.register_module()
        >>> def resnet50():
        >>>     pass
        >>> resnet = MODELS.build(dict(type='resnet50'))

    Please refer to https://mmcv.readthedocs.io/en/latest/understand_mmcv/registry.html for advanced usage.

    Args:
        name (str): Registry name.
        build_func(func, optional): Build function to construct instance from
            Registry, func:`build_from_cfg` is used if neither ``parent`` or
            ``build_func`` is specified. If ``parent`` is specified and
            ``build_func`` is not given,  ``build_func`` will be inherited
            from ``parent``. Default: None.
        parent (Registry, optional): Parent registry. The class registered in
            children registry could be built from parent. Default: None.
        scope (str, optional): The scope of registry. It is the key to search
            for children registry. If not specified, scope will be the name of
            the package where class is defined, e.g. mmdet, mmcls, mmseg.
            Default: None.
    """

    def __init__(self, name, build_func=None, parent=None, scope=None):
        self._name = name
        self._module_dict = dict()
        self._children = dict()
        self._scope = self.infer_scope() if scope is None else scope

        # self.build_func will be set with the following priority:
        # 1. build_func
        # 2. parent.build_func
        # 3. build_from_cfg

        if build_func is None:
            if parent is not None:
                self.build_func = parent.build_func
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func
        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_children(self)
            self.parent = parent
        else:
            self.parent = None

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + f'(name={self._name}, ' f'items={self._module_dict})'
        return format_str

    @staticmethod
    def infer_scope():
        """Infer the scope of registry.

        The name of the package where registry is defined will be returned.

        Example:
            >>> # in mmdet/models/backbone/resnet.py
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            The scope of ``ResNet`` will be ``mmdet``.

        Returns:
            str: The inferred scope name.
        """
        # We access the caller using inspect.currentframe() instead of
        # inspect.stack() for performance reasons. See details in PR #1844
        frame = inspect.currentframe()
        # get the frame where `infer_scope()` is called
        infer_scope_caller = frame.f_back.f_back
        filename = inspect.getmodule(infer_scope_caller).__name__
        split_filename = filename.split('.')
        return split_filename[0]

    @staticmethod
    def split_scope_key(key):
        """Split scope and key.

        The first scope will be split from key.

        Examples:
            >>> Registry.split_scope_key('mmdet.ResNet')
            'mmdet', 'ResNet'
            >>> Registry.split_scope_key('ResNet')
            None, 'ResNet'

        Return:
            tuple[str | None, str]: The former element is the first scope of
            the key, which can be ``None``. The latter is the remaining key.
        """
        split_index = key.find('.')
        if split_index != -1:
            return key[:split_index], key[split_index + 1:]
        else:
            return None, key

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def children(self):
        return self._children

    def get(self, key):
        """Get the registry record.

        Args:
            key (str): The class name in string format.

        Returns:
            class: The corresponding class.
        """
        scope, real_key = self.split_scope_key(key)
        if scope is None or scope == self._scope:
            # get from self
            if real_key in self._module_dict:
                return self._module_dict[real_key]
        else:
            # get from self._children
            if scope in self._children:
                return self._children[scope].get(real_key)
            else:
                # goto root
                parent = self.parent
                while parent.parent is not None:
                    parent = parent.parent
                return parent.get(key)

    def build(self, *args, **kwargs):
        return self.build_func(*args, **kwargs, registry=self)

    def _add_children(self, registry):
        """Add children for a registry.

        The ``registry`` will be added as children based on its scope.
        The parent registry could build objects from children registry.

        Example:
            >>> models = Registry('models')
            >>> mmdet_models = Registry('models', parent=models)
            >>> @mmdet_models.register_module()
            >>> class ResNet:
            >>>     pass
            >>> resnet = models.build(dict(type='mmdet.ResNet'))
        """

        assert isinstance(registry, Registry)
        assert registry.scope is not None
        assert registry.scope not in self.children, \
            f'scope {registry.scope} exists in {self.name} registry'
        self.children[registry.scope] = registry

    @deprecated_api_warning(name_dict=dict(module_class='module'))
    def _register_module(self, module, module_name=None, force=False):
        if not inspect.isclass(module) and not inspect.isfunction(module):
            raise TypeError('module must be a class or a function, '
                            f'but got {type(module)}')

        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                raise KeyError(f'{name} is already registered '
                               f'in {self.name}')
            self._module_dict[name] = module

    def deprecated_register_module(self, cls=None, force=False):
        warnings.warn(
            'The old API of register_module(module, force=False) '
            'is deprecated and will be removed, please use the new API '
            'register_module(name=None, force=False, module=None) instead.',
            DeprecationWarning)
        if cls is None:
            return partial(self.deprecated_register_module, force=force)
        self._register_module(cls, force=force)
        return cls

    def register_module(self, name=None, force=False, module=None):
        """Register a module.

        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.

        Example:
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(ResNet)

        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
            module (type): Module class or function to be registered.
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')
        # NOTE: This is a walkaround to be compatible with the old api,
        # while it may introduce unexpected bugs.
        if isinstance(name, type):
            return self.deprecated_register_module(name, force=force)

        # raise the error ahead of time
        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(
                'name must be either of None, an instance of str or a sequence'
                f'  of str, but got {type(name)}')

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module):
            self._register_module(module=module, module_name=name, force=force)
            return module

        return _register
```


### `build_from_cfg`函数

`build_from_cfg`函数和`Registry`类一样，也在`mmcv.utils.registry.py`里定义，同样通过`from mmcv.utils import build_from_cfg`来import。

下面是`build_from_cfg`源码：

```python
def build_from_cfg(cfg: Dict, registry: 'Registry', default_args: Optional[Dict] = None) -> Any:
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
            raise KeyError('`cfg` or `default_args` must contain the key "type", 'f'but got {cfg}\n{default_args}')
    if not isinstance(registry, Registry):
        raise TypeError('registry must be an mmcv.Registry object, 'f'but got {type(registry)}')
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError('default_args must be a dict or None, 'f'but got {type(default_args)}')

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f'type must be a str or valid type, but got {type(obj_type)}')
    try:
        return obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f'{obj_cls.__name__}: {e}')
```




## hooks

## runners

`mmcv`里的runner类，是用来管理训练过程的，其主要的内容如下：
* 其支持`EpochBasedRunner`和`IterBasedRunner`两个子类，同时用户也可以自定义runner类
* 其支持在训练过程中使用不同的workflow，到目前为止，支持两种：`train`和`val`
* 其可以通过不同的hooks来扩展功能

下面来具体介绍两个子类：`EpochBasedRunner`和`IterBasedRunner`。

### `EpochBasedRunner`

该类定义在`mmcv.runner.epoch_based_runner`文件里，可以由`from mmcv.runner import EpochBasedRunner`来import。

如该类的名字所示，该类内部的workflow是基于epoch来设置的，比如说`[('train', 2), ('val', 1)]`表示训练两个epoch，测试一个epoch，交替进行，而每个epoch可以包含指定数量的iterations、

> `MMDetection`目前使用`EpochBasedRunner`作为default的runner

`EpochBasedRunner`的核心逻辑是：

```python
# the condition to stop training
while curr_epoch < max_epochs:
    # traverse the workflow.
    # e.g. workflow = [('train', 2), ('val', 1)]
    for i, flow in enumerate(workflow):
        # mode(e.g. train) determines which function to run
        mode, epochs = flow
        # epoch_runner will be either self.train() or self.val()
        epoch_runner = getattr(self, mode)
        # execute the corresponding function
        for _ in range(epochs):
            epoch_runner(data_loaders[i], **kwargs)
```

目前`EpochBasedRunner`仅仅支持两种mode：`train`和`val`，以`train`为例，其核心逻辑是：

```python
# Currently, epoch_runner could be either train or val
def train(self, data_loader, **kwargs):
    # traverse the dataset and get batch data for 1 epoch
    for i, data_batch in enumerate(data_loader):
        # it will execute all before_train_iter function in the hooks registered. You may want to watch out for the order.
        self.call_hook('before_train_iter')
        # set train_mode as False in val function
        self.run_iter(data_batch, train_mode=True, **kwargs)
        self.call_hook('after_train_iter')
   self.call_hook('after_train_epoch')
```

### `IterBasedRunner`

和`EpochBasedRunner`不同的是, `IterBasedRunner`里的workflow是基于iterations设计的。比如说`[('train', 2), ('val', 1)]`表示训练两个iterations，测试一个iteration，交替进行。

> `MMSegmentation`目前使用`IterBasedRunner`作为默认的runner

`IterBasedRunner`的核心逻辑如下：

```python
# Although we set workflow by iters here, we might also need info on the epochs in some using cases. That can be provided by IterLoader.
iter_loaders = [IterLoader(x) for x in data_loaders]
# the condition to stop training
while curr_iter < max_iters:
    # traverse the workflow.
    # e.g. workflow = [('train', 2), ('val', 1)]
    for i, flow in enumerate(workflow):
        # mode(e.g. train) determines which function to run
        mode, iters = flow
        # iter_runner will be either self.train() or self.val()
        iter_runner = getattr(self, mode)
        # execute the corresponding function
        for _ in range(iters):
            iter_runner(iter_loaders[i], **kwargs)
```

和`EpochBasedRunner`类似，`IterBasedRunner`现在也仅支持两个mode：`train`和`val`。以`val`为例，其核心逻辑是：

```python
# Currently, iter_runner could be either train or val
def val(self, data_loader, **kwargs):
    # get batch data for 1 iter
    data_batch = next(data_loader)
    # it will execute all before_val_iter function in the hooks registered. You may want to watch out for the order.
    self.call_hook('before_val_iter')
    outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
    self.outputs = outputs
    self.call_hook('after_val_iter')
```

> 除了上述提到的那些基础的methods，`EpochBasedRunner`和`IterBasedRunner`还提供了很多methods，包括：`resume`，`save_checkpoint`，`register_hook`等。其中`register_hook`为用户提供了增加或者改变runner训练过程的hooks函数的功能。


