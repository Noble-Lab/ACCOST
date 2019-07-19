import logging
import contact_counts
import numpy as np
import sys

def main():
    logging.basicConfig(format='%(levelname)s from %(filename)s %(funcName)s :: %(message)s', level=logging.INFO)
    binfile = sys.argv[1]
    outprefix = sys.argv[2]
    #binfile = '/net/noble/vol2/home/katecook/proj/2015HiC-differential/data/test_data/cc_bins.txt' 
    #binfile = '/net/noble/vol2/home/katecook/proj/2015HiC-differential/data/test_data/pfal3D7-pfal3D7.MboI.w20000' 
    #binfile = sys.argv[1]
    #binfile = '/net/noble/vol2/home/katecook/proj/2015HiC-differential/results/katecook/2016-01-07_lower_res_contact_counts/data/preprocessedForBias/pfal3D7-pfal3D7.MboI.w10000'
    mappability = 0.5 # sys.argv[2]
    allBins,allBins_reversed,badBins = contact_counts.generate_bins(binfile,mappability)
    lengths,lengths_reversed = contact_counts.get_lengths(allBins_reversed)
    logging.info("saving length matrix")
    np.savetxt(outprefix + '_lengths.tab',lengths,delimiter='\t',newline='\n')
    logging.info("saving reversed length matrix")
    filename = outprefix+'_lengths_reversed.tab'
    fh = open(filename,'w')
    for l in lengths_reversed:
        pairs = lengths_reversed[l]
        for (i,j) in pairs:
            fh.write("%d\t%d\t%d\n" % (l,i,j))
    fh.close()



if __name__ == "__main__":
    main()

