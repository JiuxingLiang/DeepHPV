��ģ��ʹ�õ�ƽ̨��
python==3.7
tensorflow==1.13.1
keras==2.2.4

���о��ṩ���ֲ������ݣ��ֳ������ļ��н��в��ԣ�
1������TCGS-Pan-Cancer TCAG��������ɸѡ��
              DeepHPV_with_HPV_integration_sequences+Cancer
2������repeat��������ɸѡ��
              DeepHPV_with_HPV_integration_sequences+reprat

�����������ļ����У�ÿ���ļ����ھ�����HPVdata��Model��test_result�������ļ��У��Լ�Data_process.py��DeepHPV_Test.py�������ļ�
1��HPVdata������Ų������ݣ���������Ϊmat��ʽ
2��Model���������ѵ����ϵ�������ģ��
3��test_result����������������ݲ��ԵĽ��
4��Data_process.py�����ļ�����HPVdata�ļ����е����ݽ���one_hot������������ļ���������
5��DeepHPV_Test.py��ģ�Ͳ��Գ��򣬽�������ѵ���õ�������ģ�Ͷ����ݽ��в��ԣ����������Խ����Ž�test_result�ļ�����



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



