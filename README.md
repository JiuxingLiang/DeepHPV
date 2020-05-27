本模型使用的平台：
python==3.7
tensorflow==1.13.1
keras==2.2.4

本研究提供两种测试数据，分成俩个文件夹进行测试：
1，采用TCGS-Pan-Cancer TCAG进行特征筛选：
              DeepHPV_with_HPV_integration_sequences+Cancer
2，采用repeat进行特征筛选：
              DeepHPV_with_HPV_integration_sequences+reprat

在以上俩个文件夹中，每个文件夹内均包含HPVdata，Model，test_result三个子文件夹，以及Data_process.py和DeepHPV_Test.py俩个子文件
1，HPVdata：存放着测试数据，数据类型为mat格式
2，Model：存放着已训练完毕的神经网络模型
3，test_result：存放着神经网络数据测试的结果
4，Data_process.py：该文件将对HPVdata文件夹中的数据进行one_hot编码操作，该文件需先运行
5，DeepHPV_Test.py：模型测试程序，将利用已训练好的神经网络模型对数据进行测试，进而将测试结果存放进test_result文件夹中



The platform used by this model:
python == 3.7
tensorflow == 1.13.1
keras == 2.2.4

This research provides two kinds of test data, which are divided into two folders for testing:
1.  Us TCGS-Pan-Cancer TCAG for feature screening:
            DeepHPV_with_HPV_integration_sequences + Cancer
2. Use repeat for feature screening:
            DeepHPV_with_HPV_integration_sequences + reprat

In the above two folders, each folder contains three subfolders HPVdata, Model, test_result, and two subfiles Data_process.py and DeepHPV_Test.py
1, HPVdata: store test data, the data type is mat format
2. Model: stores the trained neural network model
3. test_result: stores the results of neural network data testing
4. Data_process.py: This file will perform one_hot encoding operation on the data in the HPVdata folder, this file needs to be run first
5. DeepHPV_Test.py: model test program, will use the trained neural network model to test the data, and then store the test results in the test_result folder



