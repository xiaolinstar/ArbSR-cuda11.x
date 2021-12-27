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

# Version 0.5
- 复现了代码，可以正常执行train，运行150次后生成的模型，可以进行超分辨
- 修改了部分代码，设置参数resume可以从保存的模型处继续执行
- 增加参数 model_save, optimizer_save, scheduler_save 保存模型参数的文件夹
- 删除了checkpoint中不必要的内容，包括log

# Version 0.6
- quick_start：已经设置好了参数，可以直接开始run
  - 默认设置了cpu=True, sr_size=\[202, 311\], resume=150, dir_img=...
  - 若你想使用命令行参数灵活调用：务必需要将quick_start设置为False，命令示例如下
    - `python quick_sr.py --quick_start=False --cpu=False, sr_size=500+500`
    - 可以在option/args.json中修改相应的参数信息
- train, test, quick_start均已经可以成功运行
- 文件夹experiment中下列文件夹保存训练过程中得到的参数state_dict
  - model
  - optimizer
  - scheduler

# Version 0.7
- 删除了model_150.pt，文件太大，上传到GitHub不方便。
- 已经将该文将放置到夸克网盘，可永久保存**https://pan.quark.cn/s/3a8f7c884df2**
- 也可以发email，xlxing@bupt.edu.cn，快速启动quick_start需要这个model_150.pt