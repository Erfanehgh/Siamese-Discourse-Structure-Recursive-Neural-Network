import numpy as np

def Write_Weights(W1, W21, W22, W1_File, W21_File, W22_File):
    np.savetxt(W1_File, W1)
    np.savetxt(W21_File, W21)
    np.savetxt(W22_File, W22)
    #np.savetxt(W1_File_query, W1_query)
