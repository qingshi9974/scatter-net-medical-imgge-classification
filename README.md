##introduction

基于3D散射变换和支持向量机的肺部结节CT序列的良恶性分类

1、打开test_python.py  设定相应的训练集及测试集路径。运行py文件，  进行图像的预处理，因为后续需要将测试集和训练集进行统一的归一化处理，所以仍然需要对训练集进行处理

2、打开test_matlab.m 将libsvm-3.17 和 scatnet-0.2的所有子文件夹添加到路径中。运行test_matlab.m即可生成可用于评分的Submission.csv文件。 整个特征提取过程大概需要5分钟左右。