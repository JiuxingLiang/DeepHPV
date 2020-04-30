import numpy as np
from scipy.io import loadmat

def seq_matrix(seq_list, dim):  # One_Hot Encoding

    tensor = np.zeros((len(seq_list), dim, 4))

    for i in range(len(seq_list)):
        seq = seq_list[i]
        j = 0
        for s in seq:
            if s == 'A' or s == 'a':
                tensor[i][j] = [1, 0, 0, 0]
            if s == 'G' or s == 'g':
                tensor[i][j] = [0, 0, 0, 1]
            if s == 'C' or s == 'c':
                tensor[i][j] = [0, 0, 1, 0]
            if s == 'T' or s == 't':
                tensor[i][j] = [0, 1, 0, 0]

            if s == 'N':
                tensor[i][j] = [0, 0, 0, 0]
            j = j + 1

    return tensor

def fasta_to_matrix():
    seq_name = ['HPVdata/DNA_neg_Test.mat',
                'HPVdata/DNA_pos_Test.mat']

    print(seq_name)
    dim = 2000
    ### Seq ###
    for name in seq_name:
        if 'neg_Test' in name:
            print(name)
            #y = []
            seq = []

            Liness_neg = loadmat(name)
            lines = Liness_neg['DNA_neg_Test']
            for line in lines:

                seq.append(line)
            print('neg_Test_starting!')
            global Data_neg_Test
            Data_neg_Test = seq_matrix(seq, dim)
            # global Data_neg_Test
            print('neg_Test_ending!')

        if 'pos_Test' in name:
            print(name)
            #y = []
            seq = []

            Liness_neg = loadmat(name)
            lines = Liness_neg['DNA_pos_Test']
            for line in lines:

                seq.append(line)
            print('pos_Test_starting!')
            global Data_pos_Test
            Data_pos_Test = seq_matrix(seq, dim)
            # global Data_pos_Test
            print('pos_Test_ending!')

    # global Data_pos_Test
    # global Data_neg_Train
    # global Data_neg_Test
    # global Data_pos_Train

    Data_Test = np.concatenate([Data_pos_Test, Data_neg_Test])
    Label_Test = np.concatenate([np.ones(len(Data_pos_Test)), np.zeros(len(Data_neg_Test))])

    np.save('HPVdata/Data_Test', Data_Test)  # Test_data
    np.save('HPVdata/Label_Test', Label_Test)  # Test_label

if __name__ == '__main__':

    fasta_to_matrix()
