# 快速风格迁移

<!-- Author：Evan Zone 2022.6.20 -->
> Note：本项目基于pytorch实现，可使用该项目进行diy

## 项目功能

生成带有《神奈川冲浪里》风格滤镜的图片

## 项目文件说明

fast_style_transfer.py：定义模型并训练
fst.pth：保存了图片生成神经网络的所有模型参数
demo.py：无gui，测试用
webio.py：生成《神奈川冲浪里》风格图片的简单的web应用
requirements.txt：项目环境依赖

## 项目使用步骤

1. 在该项目文件夹下打开cmd或者terminal，安装包，使用如下命令（需要python>=3.6环境）

```shell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

1. 运行webio.py文件即可，在cmd或者terminal中使用如下命令（如果双击能打开就再好不过了）

``` shell
python webio.py
```

## 训练改造指南

- 本项目只训练了《神奈川冲浪里》风格，如果想diy其他风格的图片，可以修改fast_style_transfer.py中135行的风格图片，换成自己喜欢的图片的本地文件路径
- diy训练时，需要自行下载coco数据集train2014.zip，并将fast_style_transfer.py中15行的文件路径换成数据集的本地路径
- 本项目基于wsl2-ubuntu20.04系统，其它系统未测试过环境问题
本项目使用了cuda，若自己主机没有nvidia的gpu（或者没有配置cuda环境）,自行训练时需要删除fast_style_transfer.py代码中所有的".cuda()"
