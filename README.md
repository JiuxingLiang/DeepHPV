DeepHPV was designed for detection of HPV virus in DNA integration. The data provided here is for the model of DeepHPV with HPV integration sequences + TCGA Pan Cancer and the model of DeepHPV with integration sequences + RepeatMasker.

The platform used by this model: python == 3.7 tensorflow == 1.13.1 keras == 2.2.4

This github page provides two kinds of training strategies for DeepHPV:

1. using HPV integration sequences (dsVIS database) +TCGA Pan-Cancer peaks (UCSC Genome Browser)
2. using HPV integration sequences (dsVIS database) + RepeatMasker peaks (UCSC Genome Browser). 

The model and data for each training strategies were contained in different directories.

Both directories contain following elements:
1.	Data: stores test data, the data type is mat format. We provided two sets of testing data, one is testing data enter the model when training, the other is the independent testing data achieved from VISDB
2.	Model: stores the trained neural network model.
3.	Pred_Result: stores the results of neural network data testing.
4.	Data_process.py: This file will perform one_hot encoding operation on the data in the Data folder, this file needs to be run first.
5.	DeepHPV_Test.py: model test program, will use the trained neural network model to test the data, and then store the test results in the Pred_Result folder.

If you have any questions, please contact me.

Email: liangjiuxing@m.scnu.edu.cn
