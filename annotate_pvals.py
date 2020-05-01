import sys
import contact_counts
import csv
import numpy as np

def main():
    pval_file = sys.argv[1]
    binfile = sys.argv[2]
    outfile = sys.argv[3]
   
    print_counts = False
    
    if len(sys.argv) > 4:
        print_counts = True
        countfile_A = sys.argv[4]
        countfile_B = sys.argv[5]
     
    chrMid_to_index, index_to_chrMid, badBins = contact_counts.generate_bins(binfile,0)
    
    outfh = open(outfile,'w')

    pvalfh = open(pval_file,'r')
    reader = csv.reader(pvalfh,delimiter=',')
    
    if (print_counts):
        outfh.write("i\tj\tchr1\tmid1\tchr2\tmid2\tdist\tcount_A\tcount_B\tlogFC\tln_pval\tlog10_pval\tpval\n")
        for line in reader:
            i = int(line[0])
            j = int(line[1])
            ln_pval = float(line[2])
            chr1,mid1 = index_to_chrMid[i]
            chr2,mid2 = index_to_chrMid[j]
            dist = abs(i-j)
            outfh.write("%d\t%d\t%s\t%d\t%s\t%d\t%d\t%.3f\t%.3f\t%.3e\n" % (i,j,chr1,mid1,chr2,mid2,dist,ln_pval,np.log10(np.exp(ln_pval)),np.exp(ln_pval)))
        
    else:
        outfh.write("i\tj\tchr1\tmid1\tchr2\tmid2\tdist\tln_pval\tlog10_pval\tpval\n")
        for line in reader:
            i = int(line[0])
            j = int(line[1])
            ln_pval = float(line[2])
            chr1,mid1 = index_to_chrMid[i]
            chr2,mid2 = index_to_chrMid[j]
            dist = abs(i-j)
            outfh.write("%d\t%d\t%s\t%d\t%s\t%d\t%d\t%.3f\t%.3f\t%.3e\n" % (i,j,chr1,mid1,chr2,mid2,dist,ln_pval,np.log10(np.exp(ln_pval)),np.exp(ln_pval)))
    
    outfh.close()
    pvalfh.close()

if __name__ == "__main__":
    main()

