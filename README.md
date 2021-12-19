# ArbSR-cuda11.x
学习、重构ArbSR的代码，使用cuda11.x版本，与cuda10.x不兼容
# 安装包
- python=3.6
- cuda=11.x
- pytorch=1.10.0
- cv2
- scikit-image

# 日志
2021-12-15 阅读了loss包中的内容，对Loss的实现有了比较清楚的认识，可以实现多种loss方法的混合使用
也可以自定义loss方法添加进来。

# 环境配置
Windows 10
- 安装cuda11.x相关驱动 nvcc -v检查是否安装成功
- 安装anaconda，配置清华源
- anaconda 环境安装
  - python=3.6
  - scikit-image
  - opencv
- cuda官网安装
  - pytorch=1.10.0