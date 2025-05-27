---
title: "Python + Pytorch 指北"
layout: single
date: 2025-05-09
toc: true
toc_sticky: true  # 可选：目录是否固定在侧边
categories: [ "Python"]
tags: ["Programming"]
---
这篇博客主要记录自己在学习 python 过程中遇到的一些细节或坑。

## 类属性 vs 实例属性

起因是在看李沐的《动手学深度学习》，发现其中有这样的对于神经网络顺序模块的实现。

```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module

    def forward(self, x):
        for block in self._modules.values():
            x = block(x)
        return x
```

说明中有提到 `self._modules` 是一个 OrderedDict， 当时感觉疑惑，因为正常的python语法是不支持直接这样定义有序字典的，查阅相关资料后发现原来 `_modules` 是在实例化一个对象时自动定义好的属性，是在执行父类的初始化的时候定义的。

我试图通过 `print(nn.Module._modules)` 来查看其内容，但程序报错，显示 `nn.Module` 不存在这样的属性，但是如果我们事先实例化一个 `nn.Module` 的对象 `Model`，然后再去打印 `Model._modules` 就是可以的。

源码是这么写的：

```python
class Module:
    ......
    _modules: Dict[str, Optional['Module']]
    call_super_init: bool = False
    _compiled_call_impl: Optional[Callable] = None

    def __init__(self, *args, **kwargs) -> None:
        ......
        super().__setattr__("_modules", {})
        ......
```

这里定义在 `__init__()` 函数之前的是类的属性的声明，可以区分为两类：一类是注释类型 + 赋值；另一类只有注释类型而没有赋值，在 `__init__()` 函数中使用 `self.xxx = xxx` 或者 `super().__setattr__()` 进行赋值。

> 使用类属性声明的作用：
>
> 只是提前声明了「以后会有这些属性」
> 不赋初值（初值一般在 __init__() 里动态生成）
> 用于代码提示、类型检查（像用 MyPy 静态分析工具时）
>
> 换句话说，它们是为了：
>
> ✅ 让 IDE（如 VS Code、PyCharm）有智能提示
>
> ✅ 让开发者知道这些属性存在及其预期类型
>
> ✅ 对实际运行没影响（Python 解释器并不强制要求）
>
> 区别在于，声明的同时直接进行赋值的属性是类属性，在初始化时进行赋值的是实例属性。

类属性是能够直接通过类的属性查找到的（比如可以直接 `print(nn.Module.call_super_init`），且是不同的实例共享的，而实例属性是在定义实例的时候赋予该实例的属性，仅属于该实例，不同实例之间不共享。

关于是否共享，下面有个例子可以说明：

```python
class MyModule:
    _modules = {}  # 类属性

A = MyModule()
B = MyModule()

A._modules['1'] = 1
print(B._modules)  # 会输出 {'1': 1}
A._modules = {'2'}
print(A._modules) # 会输出 {'2'}
print(B._modules) # 会输出 {'1': 1}
```

注意，当 Python 编译器看到 `A._modules` 时会优先查找实例中是否有同名属性，如果没有才会去类中寻找，代码 `A._modules['1'] = 1` 是对内容进行了修改，并不会创建实例属性。而 `A._modules = {'2'}` 则是修改了引用，此时会创建实例的属性。下面还有个例子可以帮助理解：

```python
class cls:
    clsattr = 1
    def __init__(self):
        self.objattr = 2

A = cls()
B = cls()

print(A.clsattr) # 1
A.clsattr = 3
print(A.clsattr) # 3
print(B.clsattr) # 1

C = cls()
print(C.clsattr) # 1

A.__class__.clsattr = 3
print(B.clsattr) # 3
```

也就是说如果不能通过引用的方式利用实例去修改类属性时，必须先拿到该实例的类才行 `A.__class__.xxx = xxx`，如果直接赋值（如 `A.clsattr = 3`）实际上只是在该实例上新建了一个属性（效果相当于 `A.__dict__[clsattr] = 3`）。

pytorch的这个设计的好处在于：

- 模块的状态（如子模块、参数、缓冲区、hooks 等）都保存在实例上，而不是类上。
- 因为每个模型实例都有自己的结构，不能共享 `_modules` 字典。
- 类只是定义了行为（方法、接口），具体的数据和配置是在每个实例中分开存储的。

如果类直接挂 `_modules`，那所有实例就会共享一份，很容易出错。这是pytorch的核心设计哦！

---

## named_parameters vs state_dict

当我们需要查看模型的内部参数的时候，经常会用到这两个属性，其父类是 `nn.Module`，然而这两个属性是有区别的：

- named_parameters

返回模型中所有可学习参数 `nn.Parameter`，以 `(name, parameter)` 的形式迭代（通常是权重和偏置）。

常用场景：当需要把模型的参数传入到优化器中时常使用，如 `optimizer = torch.optim.Adam(model.parameters())`

> 这里传入的是 `model.parameters()` 是因为其不含有参数的名称 `name`

- state_dict

返回模型中所有 参数 + 缓冲区（包括 `nn.Parameter` 和注册的 buffer，比如 BatchNorm 中的 running_mean、running_var），以 `{name: tensor}` 的形式组成的 `OrderedDict`。

它是模型保存和加载的核心接口：

- 保存模型：`torch.save(model.state_dict(), 'model.pth')`
- 加载模型：`model.load_state_dict(torch.load('model.pth'))`

---

## 模型保存方法

当我们想要把训练好模型想要将其保存到本地，或是在训练过程中临时保存防止故障导致数据丢失时，通常有两种方式保存模型的参数。

- `torch.save(model, "xxx.pth")`

这种方法是保存了模型的实例，其使用 Python 的 **pickle 序列化机制** 来保存整个对象，包括**模型的类名、模块路径、方法、属性等**，而不仅仅是模型的权重。

这意味着当我们想要加载模型时，必须提供**完全一致的环境**，包括：类名一致、模块路径一致、属性和结构一致，否则 pickle 在反序列化时找不到对应类，或者加载出错。

典型的问题如下：

1. 改了类名：原来叫 `class MyNet`，后来改成 `class MyNetwork` → 加载失败。
2. 改了文件位置：原来在 `model.py`，后来移到 `models/model.py` → 加载失败。
3. 改了属性：原来 `self.hidden = nn.Linear(...)`，后来换成 `self.hidden_layer = nn.Linear(...)` → 加载成功但属性不对应，可能会出 bug。

- `torch.save(model.state_dict, "xxx.pth")`

这种方法只保存模型的**参数字典（OrderedDict）**，并不包含模型的架构。

**优点：**

1. 文件更小、更简单
2. 加载更灵活（只要你有同样架构的实例就能加载）
3. 在不同代码版本/环境中更稳健（尤其跨机器、跨版本）

加载方式如下：

```python
model = MyModel(*args)
model.load_state_dict(torch.load("xxx.pth"))
```

**两种保存方法的异同点总结：**

| 对比项       | 保存整个模型 (`torch.save(model)`) | 保存 state\_dict (`torch.save(model.state_dict())`) |
| ------------ | ------------------------------------ | ----------------------------------------------------- |
| 保存内容     | 模型架构 + 参数                      | 仅参数（权重字典）                                    |
| 文件大小     | 更大                                 | 更小                                                  |
| 加载依赖     | 需要完全相同的类定义和代码环境       | 只需提供同样架构的实例                                |
| 跨版本兼容性 | 差（代码变化可能导致无法加载）       | 好（只要匹配架构就行）                                |
| 推荐使用场景 | 快速实验、临时保存                   | 生产环境、发布模型、长期保存                          |


## .contiguous()

这玩意讲起来还挺复杂，所以我直接引一篇[知乎博客](https://zhuanlan.zhihu.com/p/64551412)，感觉写得很详细了。

简单来讲就是让一个张量的元信息（行、列读取规则）和数据在内存中的实际存储位置相一致。