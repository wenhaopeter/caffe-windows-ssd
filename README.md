# caffe-ssd-windows

##编译：
    推荐使用的基本软件：vs2015 cuda8.0 cudnn5.0 python3.5(推荐使用anaconda) 
    cmd进入根目录，执行scripts文件下的build_win.cmd（.\scripts\build_win.cmd）
    根据不同的安装环境以及是都有GPU情况需要修改的地方，主要在build_win.cmd文件中
    1、73 75 77行关于使用是否使用GPU以及使用哪种编译器的问题
    2、88 行python版本问题
    3、174行 cuda的安装路径，这里需要根据自己的实际路径设置。
    详细的配置过程可以参考[官方的配置教程](https://github.com/BVLC/caffe/tree/windows)。
    完成上面的配置之后，会在根目录下生成一个build 的文件夹，进入该文件夹点击.sln就会打开caffe windows工程。
##模型训练

####数据转换
    使用编译生成的convert_annoset.exe程序进行数据转换，详细的使用方法可以查看convert_annoset.cpp文件内容
    
####开始训练
    在命令行下执行 .\examples\train_SSD.cmd命令，需要根据自己生成的数据集位置更改models\SSD文件下train和test的相关参数
    具体需要修改的有 source:  label_map_file: name_size_file: 这三个参数
该版本库完全实现了SSD论文中的算法，主要参考SSD的源码，做到了无错误迁移

