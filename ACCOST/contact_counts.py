import numpy as np
import sys,csv
import gzip
import logging
import subprocess
import scipy
from itertools import izip_longest
from running_stats import running_average_simple,running_variance_simple,running_z_simple

def load_variances(infile,nBins):
    varArray = np.zeros((nBins,nBins))
    varArray[:] = np.nan
    fh = open(infile,'r')
    reader = csv.reader(fh,delimiter='\t')
    next(reader,None)
    for line in reader:
        i = int(line[0])
        j = int(line[1])
        varArray[i,j] = float(line[2])
    return varArray

def generate_bins(midsFile, lowMappThresh):
    """
    Generate data structures to hold binned data and keep track of "bad" (low mappability) bins.

    Args:
       midsFile: name of the file with bin midpoints and mappability data. Tab delimited file with format:
           <chr ID>        <mid>   <anythingElse> <mappabilityValue> <anythingElse>+
               chr10   50000   NA      0.65    ...
       lowMappThresh: threshold at which we want to discard those bins

    Returns:
       chrMid_to_index: dictionary of chromosome + midpoints to bin index
       index_to_chrMid: list of indices, each pointing to a (chr,mid) tuple
       badBins: list of indices of bad (low mappability) bins

    """
    badBins = set() # set of indices with mappabilties < lowMappThresh
    chrMid_to_index = {}  # chr and mid to index
    index_to_chrMid = []  # index to chr and mid

    fh = open(midsFile)
    reader = csv.reader(fh, delimiter='\t')
    i = 0
    logging.info('Reading bin midpoints and mappability from %s', midsFile)
    for line in reader:  # might need a check here to avoid header line
        chrom = line[0]
        mid = int(line[1])
        mappability = float(line[3])
        if chrom not in chrMid_to_index:
            chrMid_to_index[chrom] = {}
        chrMid_to_index[chrom][mid] = i
        index_to_chrMid.append((chrom, mid))
        if mappability <= lowMappThresh:
            badBins.add(i)
        i += 1
    fh.close()
    logging.info("Found %d bins, including %d low mappability bins."
                 % (i, len(badBins)))
    return (chrMid_to_index, index_to_chrMid, badBins)



def length_index_contact_counts(infile,outfh,binSize,index_to_chrMid,chrMid_to_index=None):
    """
    Read a csv-like contact count file and output comma-delimited csv file of bin indices, counts, and genomic distances.
    
    Args:
        infile: input filename. See below for format
        outfh: output filehandle. See below for format
        index_to_chrMid: array of bin indices to (chromosome,midpoint) tuples
        chrMid_to_index: only required if input file has chromosomes and midpoints, dictionary of [chrom][mid] to bin index
    
    Input format:
        Input is assumed to be csv-like. Delimiter/dialect is guessed using csv.Sniffer
        Input can either be 3 column, or 5 column. 3 column is of the form:
            <bin index 1> <bin index 2> <count>

        5 column is of the form:
            <chrom 1> <mid 1> <chrom 2> <mid 2> <count>

        If 5 column format is used, a dictionary mapping chromosomes and midpoints to bin indices
        must also be provided. An exception will be raised if it is not.
        
        All counts are assumed to be integers, and an exception is raised if they aren't.
    
    Output format:
        Output is in csv format:
            <bin index 1>,<bin index 2>,<count>,<genomic distance>
        
        Note that no output is returned, it is just written to the filename provided.
    
    """
    #TODO: implement converting from dense contact count matrices
    # open input file
    if infile.endswith('gz'):
        fh = gzip.open(infile,'r')
    else:
        fh = open(infile,'r')
    dialect = csv.Sniffer().sniff(fh.read(1024))
    fh.seek(0)
    reader = csv.reader(fh, dialect)
    # open output file
    #outfh = open(outfile,'w')
    total_count = 0
    # read data
    for line in reader:
        # first figure out the bin indices
        if len(line) == 3:
            binIndex1 = int(line[0])
            binIndex2 = int(line[1])
            countstr = line[2]
        elif len(line) == 5:
            if chrMid_to_index is None:
                raise Exception("if input file has chromosome and midpoint, need to supply chrMid_to_index")
            chr1 = line[0]
            mid1 = int(line[1])
            chr2 = line[2]
            mid2 = int(line[3])
            countstr = line[4]
            binIndex1 = chrMid_to_index[chr1][mid1]
            binIndex2 = chrMid_to_index[chr2][mid2]
        else:
            raise Exception("Don't know how to load data of format: %s" % " ".join(line))
        if binIndex1 > binIndex2:
            raise Exception("Data should be upper triangular: %s" % " ".join(line))
        # now convert the count string to a count
        count = int(float(countstr))
        total_count += count
        #try:
        #    count = int(countstr)
        #except ValueError:
        #    raise Exception("Attempted to load non-integer data. Are you trying to load normalized contact counts?")
        length = get_length(binIndex1,binIndex2,index_to_chrMid,binSize)
        outLine = (binIndex1,binIndex2,count,length)
        outfh.write("%d,%d,%d,%d\n" % outLine)
    fh.close()
    outfh.close()
    return(total_count)

def get_length(binIndex1,binIndex2,index_to_chrMid,binSize):
    (chr1,mid1) = index_to_chrMid[binIndex1]
    (chr2,mid2) = index_to_chrMid[binIndex2]
    if chr1!=chr2:
        return -1
    else:
        return abs(mid2-mid1)/binSize

def sort_contactfile_by_length(unsorted_file,sorted_file):
    """
    Wrapper for unix sort for the processed counts csv. The counts
    are sorted by genomic distance (column 4).
    
    Args:
        unsorted_file: filename pointing to unsorted csv
            of i,j,c,d (i,j = indices, c = counts, d = distance)
        sorted_file: sorted output file objects
    """
    subprocess.check_call(["sort",unsorted_file,"-nk4,4","-nk1,1","-nk2,2","-t,"], stdout=sorted_file)

def sort_contactfile_by_index(unsorted_file,sorted_file):
    subprocess.check_call(["sort",unsorted_file,"-nk1,1","-nk2,2","-t,"], stdout=sorted_file)

def sort_contactfile_by_count(unsorted_file,sorted_file):
    subprocess.check_call(["sort",unsorted_file,"-nk3,3","-nk1,1","-t,"], stdout=sorted_file)


def open_files(filenames,delim=','):
    """
    Open a list of csv files with specified delimiter and return a list of csv readers.
    
    Args:
        filenames: list of filenames
        delim: delimiter of the input files (defaults to comma, i.e. csv files)
    
    Returns:
        readers: list of csv.readers
    """
    readers = []
    for infile in filenames:
        fh = open(infile,'r')
        reader = csv.reader(fh,delimiter=delim)
        readers.append(reader)
    return readers

#def calculate_tau_phi(mats,taufh):
#    nReplicates = len(mats)
#    nBins = mats[0].nBins
#    for i in range(nBins):
#        for j in range(nBins):
#            t = 0
#            f = 0
#            d = abs(i-j)
#            for m in mats:
#                t = t + m.size_factors(d)*m.biases[i]*m.biases[j]
#                f = f + (m.size_factors(d)*m.biases[i]*m.biases[j])**2
#            taufh.write("%d,%d,%.5f,%.5f\n" % (i,j,t,f))
#    taufh.close()

def sum_mats(mats,outfh):
    nReplicates = len(mats)
    nBins = mats[0].nBins
    filenames = []
    for m in mats:
        filenames.append(m.infile_sorted_by_index)
    logging.debug(filenames)
    readers = open_files(filenames)
    next_counts = {}
    done = {}
    current_row = 0
    done[current_row] = [False] * nReplicates
    next_counts[current_row] = np.zeros(nBins)
    for lines in izip_longest(*readers, fillvalue=None):
        for m,line in enumerate(lines):
            #logging.debug("%d line: %s" % (m,str(line)))
            if line is None:
                continue
            i = int(line[0])
            if i > current_row:
                #logging.debug("row %d is not cur_row %d" % (i,current_row))
                if i > current_row and all(done[current_row]):
                    #logging.debug("all done %d" % current_row)
                    for j in range(nBins):
                        if next_counts[current_row][j] > 0:
                            #logging.debug("writing output %d,%d,%d"  % (current_row,j,next_counts[current_row][j]))
                            t = 0
                            f = 0
                            d = abs(current_row-j)
                            for m in mats:
                                if m.size_factors is not None:
                                    t = t + m.size_factors[d]*m.biases[current_row]*m.biases[j]
                                    f = f + (m.size_factors[d]*m.biases[current_row]*m.biases[j])**2
                                else:
                                    t = t + m.biases[current_row]*m.biases[j]
                                    f = f + (m.biases[current_row]*m.biases[j])**2
                            outfh.write("%d,%d,%d,%.5f,%.5f\n" % (current_row,j,next_counts[current_row][j],t,f))
                    #logging.debug("deleting next counts and incremending cur row %d" % current_row)
                    del next_counts[current_row]
                    current_row += 1
                    if current_row not in done:
                        done[current_row] = [False] * nReplicates
                        next_counts[current_row] = np.zeros(nBins)
                else:
                    if i not in done:
                        done[i] = [False] * nReplicates
                        next_counts[i] = np.zeros(nBins)
                    #logging.debug('setting %d %d to true' % (m,current_row))
                    done[current_row][m] = True
            #logging.debug("save the data")
            # save the data
            j = int(line[1])
            c = int(line[2])
            d = int(line[3])
            #logging.debug('rep%d cur%d: %d %d %d %d %s %s' % (m,current_row,i,j,c,d,str(next_counts.keys()),str(done.keys())))
            #logging.debug(done)
            if i not in done:
                done[i] = [False]*nReplicates
                next_counts[i] = np.zeros(nBins)
            next_counts[i][j] += c
    for i in sorted(next_counts.keys()):
        for j in range(nBins):
            if next_counts[i][j] > 0:
                t = 0
                f = 0
                d = abs(i-j)
                for m in mats:
                    if m.size_factors is not None:
                        t = t + m.size_factors[d]*m.biases[i]*m.biases[j]
                        f = f + (m.size_factors[d]*m.biases[i]*m.biases[j])**2
                    else:
                        t = t + m.biases[i]*m.biases[j]
                        f = f + (m.biases[i]*m.biases[j])**2
                outfh.write("%d,%d,%d,%.5f,%.5f\n" % (i,j,next_counts[i][j],t,f))
    outfh.close()

def normalize_contact_counts(rawfile,normfh,biases):
    logging.info("normalizing contact counts")
    fh = open(rawfile,'r')
    reader = csv.reader(fh,delimiter=',')
    total_normalized = 0
    for (i,j,c,d) in reader:
        i = int(i)
        j = int(j)
        c = float(c)
        if np.isfinite(biases[i]) and np.isfinite(biases[j]) and np.isfinite(c) and biases[i]>0 and biases[j]>0:
            n = c / (biases[i]*biases[j])
            total_normalized += n
            normfh.write('%d,%d,%f' % (i,j,n))
    fh.close()
    normfh.close()
    return(total_normalized)
    

def get_q0(mats,q0_fh,no_dist_norm):
    logging.info("calculating q0")
    nReplicates = len(mats)
    nBins = mats[0].nBins
    filenames = []
    all_biases = []
    all_size_factors = []
    for m in mats:
        filenames.append(m.infile_sorted_by_index)
        all_biases.append(m.biases)
        all_size_factors.append(m.size_factors)
    logging.debug(filenames)
    # open input csv files
    readers = open_files(filenames)
    # ok now read the files one genomic distance at a time
    next_counts = {}
    done = {} # for each row (i) we will save them
    current_row = 0
    done[current_row] = [False] * nReplicates
    next_counts[current_row] = np.zeros(nBins)
    for lines in izip_longest(*readers, fillvalue=None):
        for m,line in enumerate(lines):
            if line is None:
                continue
            biases = all_biases[m]
            size_factors = all_size_factors[m]
            i = int(line[0])
            logging.debug("cur: %d m: %d line: %s done: %s keys done: %s keys next: %s" % (current_row,m,str(line),str(done[current_row]),str(done.keys()),str(next_counts.keys())))
            if i != current_row:
                if i not in done:
                    done[i] = [False]*nReplicates
                    next_counts[i] = np.zeros(nBins)
                done[current_row][m] = True
                if all(done[current_row]):
                    # write to output file
                    for j in range(current_row,nBins):
                        q0_fh.write("%d,%d,%.9f\n" % (current_row,j,next_counts[current_row][j]/nReplicates))
                    del next_counts[current_row]
                    current_row = current_row + 1
                    if current_row not in done:
                        done[current_row] = [False] * nReplicates
                        next_counts[current_row] = np.zeros(nBins)
            # save the data
            j = int(line[1])
            c = int(line[2])
            d = int(line[3])
            if i not in done:
                done[i] = [False]*nReplicates
                next_counts[i] = np.zeros(nBins)
            if no_dist_norm:
                normalized = float(c) / (biases[i]*biases[j])
            else:
                normalized = float(c) / (biases[i]*biases[j]*size_factors[d])
            next_counts[i][j] += normalized
    for i in sorted(next_counts.keys()):
        for j in range(i,nBins):
            q0_fh.write("%d,%d,%.9f\n" % (i,j,next_counts[i][j]/nReplicates))
    q0_fh.close()

def get_percentile(sum_file, index_to_chrMid, perc):
    intrachromosomal_sums = []
    fh = open(sum_file,'r')
    reader = csv.reader(fh,delimiter=',')
    for line in reader:
        i = int(line[0])
        j = int(line[1])
        if index_to_chrMid[i][0] == index_to_chrMid[j][0]:
            intrachromosomal_sums.append(float(line[2]))
    val = mynanpercentile(np.array(intrachromosomal_sums), perc)
    return val
 
def mynanpercentile(a, perc):
    part = a.ravel()
    c = np.isnan(part)
    s = np.where(~c)[0]
    return np.percentile(part[s], perc)


def remove_zeroes_and_nans(my_matrix):
    "Removing NaNs and zeroes from a matrix."
    realMatrix = my_matrix[~np.isnan(my_matrix)]
    return(realMatrix[np.nonzero(realMatrix)])

def calculate_size_factors(mats,nBins,nDistances):
    """
    Calculate distance-specific size factors from the given filenames (of distance
    sorted contact counts).
    
    Args:
        filenames: list of filenames
        nBins: number of bins in the matrix
        nDistances: number of distances
    
    Returns:
        size_factors: numpy array (len(filenames),nDistances) of size factors
            for each replicate and each distance
    """
    nReplicates = len(mats)
    logging.info("calculating size factors: nBins=%d nDistances=%d nReplicates=%d" % (nBins, nDistances, nReplicates))
    filenames = [m.infile_sorted_by_length for m in mats]
    logging.debug(filenames)
    size_factors = np.ones((nReplicates,nDistances))
    # open input csv files
    readers = open_files(filenames)
    # ok now read the files one genomic distance at a time
    current_distance = 0
    counts = np.zeros((nReplicates,(nBins)))
    counts[:] = np.nan
    next_counts = {}
    done = {}
    done[0] = [False]*nReplicates
    meanfh = open('means_by_distance_and_replicate.tab','w')
    meanfh.write("replicate\tdist_nbins\tmean\n")
    gmeansfh = open('gmeans_by_distance.tab','w')
    gmeansfh.write("dist_nbins\tgmean\n")
    for lines in izip_longest(*readers, fillvalue=None):
        # lines is a tuple of arrays
        logging.debug(lines)
        for m,line in enumerate(lines):
            biases = mats[m].biases
            if line is None:
                continue
            d = int(line[3])
            logging.debug("LINE: %d %d %d %s" % (m,current_distance,d,str(done)))
            if d != current_distance:
                logging.debug("distances don't match")
                if d not in done:
                    logging.debug("d not in done, saving")
                    done[d] = [False]*nReplicates
                if all(done[current_distance]):
                    logging.debug("all done for d=%d" % current_distance)
                    # calculate the geometric mean across replicates
                    logging.debug("calculating gmean: %s" % str(counts.shape))
                    logging.debug(counts)
                    gmean = scipy.stats.mstats.gmean(remove_zeroes_and_nans(counts), axis=None)
                    logging.debug("gmean is: %e" % gmean)
                    gmeansfh.write("%d\t%e\n" % (current_distance,gmean))
                    mean = np.nanmean(counts,axis=1)
                    for r in range(nReplicates):
                        meanfh.write("%d\t%d\t%e\n" % (r,current_distance,mean[r]))
                    # calculate the actual size factor for this replicate at this genomic distance
                    # by taking the median of the distribution of this replicate's counts divided by the
                    # geometric mean over replicates
                    for r in range(nReplicates):
                        logging.debug("calculating size_factor: %d %s" % (r,str(counts.shape)))
                        rCounts = counts[r,:]
                        logging.debug(rCounts)
                        size_factor = np.nanmedian( np.divide( rCounts, gmean ) )
                        if np.isnan(size_factor):
                            size_factor = np.nan
                        logging.debug("size factor is: %f" % size_factor)
                        size_factors[r,current_distance] = size_factor
                    # now reset the current distance + counts + done
                    del done[current_distance]
                    current_distance = current_distance + 1
                    if current_distance not in done:
                        logging.debug("current_distance not in done, saving")
                        done[current_distance] = [False]*nReplicates
                    if current_distance in next_counts:
                        logging.debug("counts is now: (%d,%d) for current_distance %d" % (next_counts[current_distance].shape[0],next_counts[current_distance].shape[1],current_distance))
                        counts = next_counts[current_distance]
                        del next_counts[current_distance]
                    else:
                        logging.debug("next counts shape: (%d,%d)" % (nReplicates,nBins+1-current_distance))
                        counts = np.zeros((nReplicates,nBins-current_distance))
                        counts[:] = np.nan
                    i = int(line[0])
                    j = int(line[1])
                    c = float(line[2])
                    normalized = c / (biases[i]*biases[j])
                    counts[m,i] = normalized
                else:
                    logging.debug("not all done for d=%d so saving d=%d to next counts" % (current_distance,d))
                    for dd in range(current_distance+1):
                        if dd in done:
                            done[current_distance][m] = True
                    i = int(line[0]) # the index is just the i part of the (i,j)
                    j = int(line[1])
                    c = float(line[2])
                    normalized = c / (biases[i]*biases[j])
                    if d not in next_counts:
                        next_counts[d] = np.zeros((nReplicates,nBins-d))
                        next_counts[d][:] = np.nan
                        logging.debug("next_counts size: (%d,%d) d: %d" % (nReplicates,nBins+1-d,d))
                    next_counts[d][m,i] = normalized
            else:
                if d not in done:
                    done[d] = [False]*nReplicates
                i = int(line[0])
                j = int(line[1])
                c = float(line[2])
                logging.debug("distances match so saving to counts: %d %d  %s" % (i,j,str(counts.shape)))
                normalized = c / (biases[i]*biases[j])
                counts[m,i] = normalized
    # doing the final one
    gmean = scipy.stats.mstats.gmean(remove_zeroes_and_nans(counts), axis=None)
    gmeansfh.write("%d\t%.3f\n" % (current_distance,gmean))
    np.seterr(all='raise')  # From here on out, die if you hit a NaN.
    mean = np.nanmean(counts,axis=1)
    for r in range(nReplicates):
        meanfh.write("%d\t%d\t%f\n" % (r,current_distance,mean[r]))
    for r in range(nReplicates):
        logging.debug("calculating size_factor: %d %s" % (r,str(counts.shape)))
        rCounts = counts[r,:]
        size_factor = np.nanmedian( np.divide( rCounts, gmean ) )
        if np.isnan(size_factor):
            size_factor = np.nan
        size_factors[r,current_distance] = size_factor
    # do the rest of the leftovers
    #logging.debug("calculating the rest of the leftovers: %s" % str(next_counts))
    for d in next_counts.keys():
        logging.debug("doing d=%d from next counts" % d)
        c = next_counts[d]
        logging.debug(remove_zeroes_and_nans(c))
        gmean = scipy.stats.mstats.gmean(remove_zeroes_and_nans(c), axis=None)
        gmeansfh.write("%d\t%.3f\n" % (d,gmean))

        # N.B. This block is useless because it just goes to an unused output file.
        mean = np.nanmean(c,axis=1)
        if (len(c[1]) == 0):
            logging.info("doing d=%d from next counts" % d)
            logging.info(remove_zeroes_and_nans(c))
            logging.info("Cannot take the mean of zero values.")
            gmeansfh.close()
            meanfh.close()
            sys.exit(1)
        for r in range(nReplicates):
             meanfh.write("%d\t%d\t%f\n" % (r,d,mean[r]))

        for r in range(nReplicates):
            rCounts = next_counts[d][r,:]
            #logging.debug("calculating size_factor: %d %s %s %s %.3f" % (r,str(next_counts[d].shape),str(next_counts[d][r,:]),str(rCounts),gmean))
            size_factor = np.nanmedian( np.divide( rCounts, gmean ) )
            if np.isnan(size_factor):
                size_factor = np.nan
            size_factors[r,d] = size_factor
    logging.debug("size factors:\n%s" % str(size_factors))
    for i,m in enumerate(mats):
        m.size_factors = size_factors[i,:]
    meanfh.close()
    gmeansfh.close()
    return mats


def calc_mean_var(readers,all_biases,all_size_factors,nBins,nDistances,badBins,smooth_dist):
    """
    Calculate means, variances, z factors, and corrected variances for each genomic distance.
    
    readers, all_biases, and all_size_factors are assumed to be lists corresponding to replicates.
    
    Called by combine_replicates_and_calculate_mean_variance.
    
    Args:
        readers: list of csv readers of contact counts sorted by genomic distance
        all_biases: list of bias vectors (e.g., ICE biases)
        all_size_factors: list of size factor vectors (as calculated by calculate_size_factors)
        nnfh = open('means_by_distance_and_replicate.tab','w')
    meanfh.write("replicate\tdist_nbins\tmean\n")
Bins: number of 1D bins
        nDistances: number of genomic distances
        badBins: Python set of bin indices that are "bad" (e.g., unmappable regions)
    
    Returns:
        All returned data is in numpy arrays of size (nBins+1,). The +1 is for interchromosomal counts
        means: mean count per genomic distance
        variances: variance per genomic distance
        z_factors: z_factor (see paper) per genomic distance
        corrected_variances: variances corrected by z_factors
    
    """
    outfh = open("means_vars_A.txt",'w')
    nReplicates = len(readers)
    means = {}
    variances = {}
    z_factors = {}
    corrected_variances = {}
    # ok now read the files one genomic distance at a time
    n = 0
    current_distance = 0
    current_k = 0 # no longer just distance
    done = {}
    normalized_counts = {}
    i_values = {}
    j_values = {}
    z = {}
    meanfh = open('means_by_distance_calcmeanvar.tab','w')
    meanfh.write("dist_nbins\tk\tmean\n")
    varfh = open('vars_by_distance_calcmeanvar.tab','w')
    varfh.write("dist_nbins\tk\tvar\n") 
    for lines in izip_longest(*readers, fillvalue=None):
        # lines is a tuple of arrays
        # m is sample index
        for m,line in enumerate(lines):
            if line is None:
                continue
            logging.debug(line)
            logging.debug(done)
            d = int(line[3])
            if d<0:
                continue # TODO: skip interchromosomal
            biases = all_biases[m]
            size_factors = all_size_factors[m]
            i = int(line[0])
            j = int(line[1])
            if d != current_distance:
                # check if the last distance ("current_distance") has been completed for all replicates
                # if it's completed so calculate the mean and variance for each k
                if current_distance not in done:
                    done[current_distance] = [False]*nReplicates
                    done[current_distance][m] = True
                elif all(done[current_distance]):
                    # calculate mean and variance from the saved stuff
                    if current_distance not in normalized_counts:
                        # we don't have anything for this one, so skip it
                        logging.debug("no normalized counts for distance %d" % current_distance)
                        means[current_distance] = np.nan
                        variances[current_distance] = np.nan
                        z_factors[current_distance] = np.nan
                        corrected_variances[current_distance] = np.nan
                    else:
                        # calculate the running means
                        logging.debug("normalized_counts: %s" % str(normalized_counts[current_distance]))
                        #norm_counts_series = pd.Series(normalized_counts[current_distance])
                        #z_series = pd.Series(z[current_distance])
                        means[current_distance] = running_average_simple(normalized_counts[current_distance], window=smooth_dist)
                        # norm_counts_series.rolling(window=smooth_dist, min_periods=smooth_dist).mean()
                        # np.nanmean(np.array(normalized_counts[current_distance]))
                        variances[current_distance] = running_variance_simple(normalized_counts[current_distance], window=smooth_dist)
                        for smoothstart,mm in enumerate(means[current_distance]):
                            ii = smoothstart + (smooth_dist-1)/2
                            if i_values[current_distance].shape[0]<= ii or np.isnan(i_values[current_distance][ii][0]):
                                continue
                            assert ii == i_values[current_distance][ii][0]
                            jj = j_values[current_distance][ii][0]
                            vv = variances[current_distance][smoothstart]
                            outfh.write("%d\t%d\t%d\t%f\t%f\n" % (ii,jj,current_distance,mm,vv))
                        # norm_counts_series.rolling(window=smooth_dist, min_periods=smooth_dist).var()
                        # np.nanvar(np.array(normalized_counts[current_distance]), ddof=1)
                        logging.debug("means: %s vars: %s" % (str(means[current_distance]), str(variances[current_distance])))
                        z_factors[current_distance] = running_z_simple(z[current_distance], means[current_distance], window=smooth_dist)
                        # (z_series.rolling(window=smooth_dist, min_periods=smooth_dist).sum() * means[current_distance] ) / smooth_dist
                        # np.nansum(np.array(z[current_distance]))*(means[current_distance])/len(z[current_distance])
                        corrected_variances[current_distance] = variances[current_distance] - z_factors[current_distance]
                        # get rid of the stuff we don't need any more
                        del normalized_counts[current_distance]
                        del z[current_distance]
                        del done[current_distance]
                    logging.debug("writing for %d, internal loop" % d)
                    #for (u,uu) in enumerate(means[current_distance]):
                    #    meanfh.write("%d\t%d\t%f\n" % (current_distance,u,uu))
                    #for (v,vv) in enumerate(variances[current_distance]):
                    #    varfh.write("%d\t%d\t%f\n" % (current_distance,v,vv))
                    # update the current distance
                    current_distance = d
                else:
                    done[current_distance][m] = True
                    logging.debug(d)
            # whether or not there is a mean/var to calculate, we have to store the counts
            i = int(line[0])
            j = int(line[1])
            c = float(line[2])
            if i not in badBins and j not in badBins:
                if d not in normalized_counts:
                    normalized_counts[d] = np.empty((nBins-d,nReplicates))
                    normalized_counts[d][:] = np.nan
                    
                    i_values[d] = np.empty((nBins-d,nReplicates))
                    i_values[d][:] = np.nan
                    j_values[d] = np.empty((nBins-d,nReplicates))
                    j_values[d][:] = np.nan
                    
                    z[d] = np.empty((nBins-d,nReplicates)) 
                    z[d][:] = np.nan
                if size_factors is not None:
                    normalized = c/(biases[i]*biases[j]*size_factors[d])
                    normalized_counts[d][i,m] = float(normalized)
                    i_values[d][i] = i
                    j_values[d][i] = j
                    z[d][i,m] = 1./(biases[i]*biases[j]*size_factors[d])
                else:
                    normalized = c/(biases[i]*biases[j])
                    normalized_counts[d][i,m] = float(normalized)
                    z[d][i,m] = 1./(biases[i]*biases[j])
        #logging.debug("%d %d" % (current_distance,(nDistances-1)))
        n = n + 1
    # any leftovers?
    for current_distance in normalized_counts:
        # calculate the running means
        logging.debug("normalized_counts: %s" % str(normalized_counts[current_distance]))
        means[current_distance] = running_average_simple(normalized_counts[current_distance], window=smooth_dist)
        variances[current_distance] = running_variance_simple(normalized_counts[current_distance], window=smooth_dist)
        for smoothstart,mm in enumerate(means[current_distance]):
            ii = smoothstart + (smooth_dist-1)/2
            if i_values[current_distance].shape[0]<= ii or np.isnan(i_values[current_distance][ii][0]):
                continue
            assert ii == i_values[current_distance][ii][0]
            jj = j_values[current_distance][ii][0]
            vv = variances[current_distance][smoothstart]
            outfh.write("%d\t%d\t%d\t%f\t%f\n" % (ii,jj,current_distance,mm,vv))
        logging.debug("means: %s vars: %s" % (str(means[current_distance]), str(variances[current_distance])))
        z_factors[current_distance] = running_z_simple(z[current_distance], means[current_distance], window=smooth_dist)
        corrected_variances[current_distance] = variances[current_distance] - z_factors[current_distance]
        logging.debug("writing for %d, final" % d)
        for (u,uu) in enumerate(means[current_distance]):
            meanfh.write("%d\t%d\t%f\n" % (current_distance,u,uu))
        for (v,vv) in enumerate(variances[current_distance]):
            varfh.write("%d\t%d\t%f\n" % (current_distance,v,vv))
    meanfh.close()
    varfh.close()
    # now collapse the values into 1d arrays
    means_list = []
    variances_list = []
    z_list = []
    corrected_list = []
    for d in means.keys():
        if not np.all((np.isnan(means[d]))):
            means_list.extend(list(means[d]))
            variances_list.extend(list(variances[d]))
            z_list.extend(list(z_factors[d]))
            corrected_list.extend(list(corrected_variances[d]))
    outfh.close()
    return np.array(means_list),np.array(variances_list),np.array(z_list),np.array(corrected_list)



def check_replicates(replicates):
    """
    Make sure that all the replicates have the same number of bins, etc.
    """
    # check nBins
    cur_nBins = replicates[0].nBins
    for r in replicates:
        assert r.nBins == cur_nBins, "Different nBins between replicates"

    # check nDistances
    cur_nDistances = replicates[0].nDistances
    for r in replicates:
        assert r.nDistances == cur_nDistances, "Different nDistances between replicates"

    # check badBins
    cur_badBins = replicates[0].badBins
    for r in replicates:
        assert r.badBins == cur_badBins, "Different badBins sets"
    
    return True

 
def combine_replicates_and_calculate_mean_variance(replicates,smooth_dist):
    """
    Calculate mean, variance, etc for each genomic distance from a list of replicates.
    
    Wrapper for calc_mean_var.
    
    Args:
        replicates: list of contactCountMatrix objects
    
    Returns:
        output: contactCountMatrix object with mean, vars, z, and w set
    """
    assert check_replicates(replicates) #make sure that they have the same number of bins etc
    nReps = len(replicates)
    nBins = replicates[0].nBins
    nDistances = replicates[0].nDistances
    badBins = replicates[0].badBins
    # open input csv files
    filenames = []
    all_biases = []
    all_size_factors = []
    for m in replicates:
        filenames.append(m.infile_sorted_by_length)
        all_biases.append(m.biases)
        if m.size_factors is not None:
            all_size_factors.append(m.size_factors)
        else:
            all_size_factors.append([1.]*nBins)
    readers = open_files(filenames)
    
    means,variances,z_factors,corrected_variances = calc_mean_var(readers,all_biases,all_size_factors,nBins,nDistances,badBins,smooth_dist)
    
    
    # create output object and set statistics
    output = contactCountMatrix(replicates[0].chrMid_to_index, replicates[0].index_to_chrMid, replicates[0].badBins)
    output.binSize = replicates[0].binSize
    output.mean = means
    output.vars = variances
    output.z = z_factors
    output.w = corrected_variances
    output.nReplicates = nReps
    #if replicates[0].size_factors is None:
    #    output.size_factors = None
    #else:
    #    output.size_factors = average_size_factors(replicates)
    #output.biases = average_biases(replicates)
    return output

def average_size_factors(replicates):
    all_size_factors = np.zeros((len(replicates),len(replicates[0].size_factors)))
    for i,mat in enumerate(replicates):
        all_size_factors[i,:] = mat.size_factors
    return np.mean(all_size_factors,axis=0)

def average_biases(replicates):
    all_biases = np.zeros((len(replicates),len(replicates[0].biases)))
    for i,mat in enumerate(replicates):
        all_biases[i,:] = mat.biases.reshape((len(replicates[0].biases),))
    return np.mean(all_biases,axis=0)

class contactCountMatrix:
    """
    Class to represent contact count matrices. Has functionality for modeling distribution of counts.

    Args:
       chrMid_to_index: dictionary of chromosome + midpoints to bin index
       index_to_chrMid: list of (chr,mid) tuple
       badBins: unsorted list of indices of bad (low mappability) bins
    """
    
    def __init__(self, chrMid_to_index=None, index_to_chrMid=None, badBins=None, nBins=None):
        self.chrMid_to_index = chrMid_to_index
        self.index_to_chrMid = index_to_chrMid
        self.badBins = badBins
        if nBins is not None:
            self.nBins = nBins
        else:
            self.nBins = len(index_to_chrMid)
        self.has_biases = False
    
    def load_biases(self, bias_filename):
        logging.info('Loading biases from %s' % bias_filename)
        # set up array to hold the biases
        bias_array = np.empty((self.nBins,1))
        bias_array[:] = np.nan
        i = 0
        # check for gzip
        if bias_filename.endswith('.gz'):
            bias_file = gzip.open(bias_filename)
        else:
            bias_file = open(bias_filename, 'r')
        # load the biases
        for line in bias_file:
            line = line.rstrip()
            # check for tab- or comma-delimited data
            # if delimited then us the first two columns as chromosome and bin midpoint
            if '\t' in line or ',' in line:
                if '\t' in line:
                    cells = line.split('\t')
                else:
                    cells = line.split(',')
                chrom,mid,bias = cells
                mid = int(mid)
                bias = float(bias)
                binIndex = self.chrMid_to_index[chrom][mid]
                bias_array[binIndex] = bias
                # check for zeros and add to bad bins
                if bias==0 and binIndex not in self.badBins:
                    self.badBins.add(binIndex)
            # otherwise assume there's a single float per line that represents the bias
            else:
                try:
                    bias = float(line)
                except ValueError:
                    raise Exception("Couldn't understand bias %s" % line)
                bias_array[i] = bias
                # check for zeros and add to bad bins
                if bias==0 and i not in self.badBins:
                    self.badBins.add(i)
                i += 1
        bias_array[np.where(bias_array==0)] = 1
        bias_array[np.where(~np.isfinite(bias_array))] = 1
        self.has_biases = True
        self.biases = bias_array


