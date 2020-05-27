import numpy as np
import os
from scipy.io import loadmat

# from array import array
# from util import seq_matrix

def seq_matrix(seq_list, dim):  # One Hot Encoding

    tensor = np.zeros((len(seq_list), dim, 4))

    for i in range(len(seq_list)):
        seq = seq_list[i]
        j = 0
        for s in seq:
            # if s == 'A' or s == 'a':
            #    tensor[i][j] = [1, 0, 0, 1, 0, 1, 0, 0]
            # if s == 'T' or s == 't':
            #    tensor[i][j] = [0, 1, 0, 0, 1, 0, 0, 0]
            # if s == 'C' or s == 'c':
            #    tensor[i][j] = [0, 0, 1, 0, 0, 1, 0, 1]
            # if s == 'G' or s == 'g':
            #    tensor[i][j] = [0, 0, 0, 1, 0, 0, 1, 1]
            # if s == 'N':
            #    tensor[i][j] = [0, 0, 0, 0, 0, 0, 0, 0]
            if s == 'A' or s == 'a':
                tensor[i][j] = [1, 0, 0, 0]
            if s == 'T' or s == 't':
                tensor[i][j] = [0, 1, 0, 0]
            if s == 'C' or s == 'c':
                tensor[i][j] = [0, 0, 1, 0]
            if s == 'G' or s == 'g':
                tensor[i][j] = [0, 0, 0, 1]
            if s == 'N':
                tensor[i][j] = [0, 0, 0, 0]
            j += 1
    return tensor


def fasta_to_matrix():
    seq_name_file = ['Data/dsVIS_neg_Test.mat',
                     'Data/dsVIS_pos_Test.mat',
                     'Data/Independent_VISDB_neg_Test.mat',
                     'Data/Independent_VISDB_pos_Test.mat']  # data file

    print(seq_name_file)
    dim = 2000
    # print('starting')

    for name in seq_name_file:
        if 'dsVIS_pos_Test' in name:
            print(name)
            y = []
            seq = []

            liness_pos = loadmat(name)
            lines = liness_pos['dsVIS_pos_Test']
            for line in lines:
                seq.append(line)  #

            print('dsVIS_pos_Tes_starting!')
            global dsVIS_pos_Test
            dsVIS_pos_Test = seq_matrix(seq, dim)
            # global Data_pos_Train
            print('dsVIS_pos_Tes_end!')

        if 'dsVIS_neg_Test' in name:
            print(name)
            y = []
            seq = []

            Liness_neg = loadmat(name)
            lines = Liness_neg['dsVIS_neg_Test']
            for line in lines:
                seq.append(line)
            print('dsVIS_neg_Test_processing!')
            global dsVIS_neg_Test
            dsVIS_neg_Test = seq_matrix(seq, dim)
            # global Data_neg_Train
            print('dsVIS_neg_Test_end!')

        if 'Independent_VISDB_neg_Test' in name:
            print(name)
            y = []
            seq = []
            Liness_neg = loadmat(name)
            lines = Liness_neg['Independent_VISDB_neg_Test']
            for line in lines:
                seq.append(line)
            print('Independent_VISDB_neg_Test_processing!')
            global Independent_VISDB_neg_Test
            Independent_VISDB_neg_Test = seq_matrix(seq, dim)
            # global Data_neg_Test
            print('Independent_VISDB_neg_Test_end!')

        if 'Independent_VISDB_pos_Test' in name:
            print(name)
            y = []
            seq = []
            Liness_neg = loadmat(name)
            lines = Liness_neg['Independent_VISDB_pos_Test']
            for line in lines:
                seq.append(line)
            print('Independent_VISDB_pos_Test_processing!')
            global Independent_VISDB_pos_Test
            Independent_VISDB_pos_Test = seq_matrix(seq, dim)
            # global Data_pos_Test
            print('Independent_VISDB_pos_Test_end!')

    # global Data_pos_Test
    # global Data_neg_Train
    # global Data_neg_Test
    # global Data_pos_Train

    # Data_Train = np.concatenate([Data_pos_Train, Data_neg_Train])  # Train data
    # Label_Train = np.concatenate([np.ones(len(Data_pos_Train)), np.zeros(len(Data_neg_Train))])  # Trian_label
    dsVIS_Test_Data = np.concatenate([dsVIS_pos_Test, dsVIS_neg_Test])  # Test data
    dsVIS_Test_Label = np.concatenate([np.ones(len(dsVIS_pos_Test)), np.zeros(len(dsVIS_neg_Test))])  # Test label
    VISDB_Test_Data = np.concatenate([Independent_VISDB_pos_Test, Independent_VISDB_neg_Test])  # Test data
    VISDB_Test_Label = np.concatenate(
        [np.ones(len(Independent_VISDB_pos_Test)), np.zeros(len(Independent_VISDB_neg_Test))])

    # np.save('Data/Data_Train', Data_Train)
    # np.save('Data/Label_Train', Label_Train)
    np.save('Data/dsVIS_Test_Data', dsVIS_Test_Data)
    np.save('Data/dsVIS_Test_Label', dsVIS_Test_Label)
    np.save('Data/VISDB_Test_Data', VISDB_Test_Data)
    np.save('Data/VISDB_Test_Label', VISDB_Test_Label)


if __name__ == '__main__':
    fasta_to_matrix()
