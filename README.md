# IoT-and-Edge-Intelligence
PyTorch Deep Learning training tutorial on Edge Computing devices.

- [IoT-and-Edge-Intelligence](#iot-and-edge-intelligence)
  - [设置PyTorch运行环境](#设置pytorch运行环境)
    - [Anaconda](#anaconda)
    - [PyTorch](#pytorch)
  - [使用PyTorch上训练CNN](#使用pytorch上训练cnn)
  - [配置CNN训练参数](#配置cnn训练参数)

## 设置PyTorch运行环境
### Anaconda
[Anaconda](https://www.anaconda.com/)是一个可以便捷获取包且对包进行管理，同时对环境进行统一管理的发行版本。Anaconda包含了conda、Python等在内的超过180个科学包及其依赖项。

下面我们将在Windows 10系统下先安装Anaconda，然后使用Anaconda中的`conda`命令安装Pytorch。

**下载安装程序** 首先从Anaconda官方网站的下载页面`https://www.anaconda.com/products/individual`下载其安装程序。

![Anaconda Individual Edition](images/anconda.png)

**安装** 双击打开下载好的 `Anaconda3-2020.11-Windows-x86_64.exe.exe` ，然后按照如下图所示的步骤安装好Anaconda。在安装过程中可以自定义安装路径，其余设置保持默认即可。安装时间大概10+分钟。

![Install Anaconda](images/win_0.png)

### PyTorch
> [PyTorch](https://pytorch.org/). An open source machine learning framework that accelerates the path from research prototyping to production deployment.

前面我们已经安装好了Anaconda，因此只需要使用 `conda` 运行简单的命令就可以安装PyTorch。

**PyTorch安装命令** 首先根据平台及需求到PyTorch官方网站获得安装命令，这里我们的安装命令是 `conda install pytorch torchvision torchaudio cpuonly -c pytorch` 。

![PyTorch Install Command](images/potorch.png)

**安装** 然后，打开Windows 10的开始菜单，选择 `Anaconda Prompt(anaconda3)` 打开Anaconda终端，执行上一步获得的安装命令。

![Install PyTorch](images/win_1.png)

**测试PyTorch** 等待安装成功之后，可以在打开的终端上运行下面的程序测试PyTorch是否安装成功。如果执行过程中程序没有报错，则安装成功。
``` Python
import torch
import torchvision

torch.cuda.is_available()
```

![Test Pytorch](images/win_2.png)

## 使用PyTorch上训练CNN

## 配置CNN训练参数