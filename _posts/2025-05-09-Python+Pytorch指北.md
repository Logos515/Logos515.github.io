---
title: "Python指北"
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
不赋初值（初值一般在 __init__() 里动态生成）
用于代码提示、类型检查（像用 MyPy 静态分析工具时）
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
> 
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

