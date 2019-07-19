import sys
import contact_counts
import numpy as np
import logging

def symmetrize(a):
    return a + np.triu(a,k=1).T


def main():
    logging.basicConfig(level=logging.DEBUG)
    binfile = sys.argv[1]
    precalculated_length_file = sys.argv[2]
    matfile_A = sys.argv[3] # normalized counts
    matfile_B = sys.argv[4] # normalized counts
    outfile = sys.argv[5]

    # set up bins   
    logging.info("setting up bins")
    (allBins,allBins_reversed,badBins) = contact_counts.generate_bins(binfile,0.5)

    # read input contact count matrices
    logging.info("creating matrices")
    matrix_A = contact_counts.contactCountMatrix(allBins,allBins_reversed,badBins)
    matrix_B = contact_counts.contactCountMatrix(allBins,allBins_reversed,badBins)
    logging.info("loading matrix data")
    matrix_A.load_from_file(matfile_A)
    matrix_B.load_from_file(matfile_B)

    data_A = matrix_A.data.todense()
    data_B = matrix_B.data.todense()
   
    logging.debug(data_A.shape) 
    logging.debug(data_B.shape)
    
    #data_A = symmetrize(data_A)
    #data_B = symmetrize(data_B)

    logging.debug(data_A.shape) 
    logging.debug(data_B.shape)
    
    logging.info("Calculating ratios")
    ratios = (data_A + 1)/(data_B + 1)
    
    logging.debug(matrix_A.data.todense()[0:5,0:5])
    logging.debug(data_A[0:5,0:5])
    logging.debug(matrix_B.data.todense()[0:5,0:5])
    logging.debug(data_B[0:5,0:5])
    logging.debug(ratios[0:5,0:5])
    
    logging.info("saving output")
    np.savetxt(outfile,ratios,delimiter='\t')

if __name__ == "__main__":
    main()
