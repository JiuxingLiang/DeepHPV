DeepHPV was designed for detection of HPV virus in DNA integration by deep learning. The data provided here is for the model of DeepHPV with HPV integration sequences + TCGA Pan Cancer.

Run ‘Data_Process.py’ to get one-hot encoding test data and labels.
Run ‘DeepHPV-Test.py’ to detect DNA sequence of HPV virus.

Framework:
DeepHPV framework model contains input layer, 1st convolution1D layer, 2nd convolution1D layer, max pooling layer, 1st dropout layer, 2nd dropout layer, attention layer, 1st dense layer (fully connected layer), 2nd dense layer, concatenate layer, classifier layer.

Dependency:
Keras library 2.2.4. 
scikit-learn 0.22. 

If you have any questions, please contact me.

Email: liangjiuxing@m.scnu.edu.cn
