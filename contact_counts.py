import numpy as np
import sys,csv
import gzip
import logging
import subprocess
import scipy
from itertools import izip_longest
from running_stats import running_average_simple,running_variance_simple,running_z_simple

import os
import tempfile
import pdb

def load_variances(infile,nBins):
    assert False, "not used"
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


# --------------------------------------------------------------------------- #

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
    
    # Open the input file and sniff it
    if infile.endswith('gz'):
        fh = gzip.open(infile,'r')
    else:
        fh = open(infile,'r')
    dialect = csv.Sniffer().sniff(fh.read(1024))
    fh.seek(0)
    reader = csv.reader(fh, dialect)

    total_count = 0
    # Read the data
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
        
        # calculated the genomic distance and write it to the temp file
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

# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #



def sum_mats(mats,outfh):
    nReplicates = len(mats)
    nBins = mats[0].nBins
    filenames = []
    for m in mats:
        filenames.append(m.infile_sorted_by_index)
    readers = open_files(filenames)
    next_counts = {}
    done = {}
    current_row = 0
    done[current_row] = [False] * nReplicates
    next_counts[current_row] = np.zeros(nBins)

    for lines in izip_longest(*readers, fillvalue=None):
        for m,line in enumerate(lines):           
            if line is None:
                continue
            
            i = int(line[0])

            if i > current_row:               
                if i > current_row and all(done[current_row]):
                    for j in range(nBins):
                        if next_counts[current_row][j] > 0:
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
                    del next_counts[current_row]
                    current_row += 1
                    if current_row not in done:
                        done[current_row] = [False] * nReplicates
                        next_counts[current_row] = np.zeros(nBins)
                else:
                    if i not in done:
                        done[i] = [False] * nReplicates
                        next_counts[i] = np.zeros(nBins)
                    
                    done[current_row][m] = True
           
            j = int(line[1])
            c = int(line[2])
            d = int(line[3])
            if i not in done:
                done[i] = [False]*nReplicates
                next_counts[i] = np.zeros(nBins)
            next_counts[i][j] += c
   
    # handle leftovers
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
    
    # initialize size_factors and counts matrices with 1 and nans respectively
    size_factors = np.ones((nReplicates,nDistances))
    
    current_distance = 0
    counts = np.zeros((nReplicates,(nBins)))
    counts[:] = np.nan
    
    next_counts = {}
    done = {}
    done[0] = [False]*nReplicates
    
    # ready file handlers
    meanfh = tempfile.NamedTemporaryFile(delete=False,prefix="means_by_distance_and_replicate_")
    #meanfh = open('means_by_distance_and_replicate.tab','w')
    meanfh.write("replicate\tdist_nbins\tmean\n")
    gmeansfh = tempfile.NamedTemporaryFile(delete=False,prefix="gmeans_by_distance_")
    #gmeansfh = open('gmeans_by_distance.tab','w')
    gmeansfh.write("dist_nbins\tgmean\n")

    # open input csv files
    readers = open_files(filenames)

    # let's read each file concurretly, one line at a time
    for lines in izip_longest(*readers, fillvalue=None):
        # process file m with line line
        for m,line in enumerate(lines):
            biases = mats[m].biases
            # if the current file has been exhausted keep going with the rest of the files
            if line is None:
                continue
            
            # get the distance    
            d = int(line[3])
            
            # if the current line has moved up further to a larger distance
            if d != current_distance:
                if d not in done:
                    done[d] = [False]*nReplicates
                
                # if we have processed all files for the current distance
                if all(done[current_distance]):                   
                    # 1. calculate the geometric mean across all replicates and write it down
                    gmean = scipy.stats.mstats.gmean(remove_zeroes_and_nans(counts), axis=None)
                    gmeansfh.write("%d\t%e\n" % (current_distance,gmean))
                    
                    # 2. calculate the mean for this disatnce d for each replicate and write it down
                    mean = np.nanmean(counts,axis=1)
                    for r in range(nReplicates):
                        meanfh.write("%d\t%d\t%e\n" % (r,current_distance,mean[r]))
                    
                    # 3. calculate the actual size factor for this replicate at this genomic distance
                    # by taking the median of the distribution of this replicate's counts divided by the
                    # geometric mean over replicates
                    for r in range(nReplicates):
                        rCounts = counts[r,:]
                        size_factor = np.nanmedian( np.divide( rCounts, gmean ) ) # is a scalar at this point
                        if np.isnan(size_factor):
                            size_factor = np.nan
                        size_factors[r,current_distance] = size_factor
                    
                    
                    # now reset the current distance + counts + done
                    del done[current_distance]
                    current_distance = current_distance + 1
                    if current_distance not in done:
                        done[current_distance] = [False]*nReplicates
                    if current_distance in next_counts:
                        counts = next_counts[current_distance]
                        del next_counts[current_distance]
                    else:
                        counts = np.zeros((nReplicates,nBins-current_distance))
                        counts[:] = np.nan
                    
                    # calculate the normalzied count and store it in the matrix count
                    i = int(line[0])
                    j = int(line[1])
                    c = float(line[2])
                    normalized = c / (biases[i]*biases[j])
                    counts[m,i] = normalized
                else:
                    # we are done with the previous distance; set it to true in done[]
                    for dd in range(current_distance+1):
                        # if any distance smaller or equal current_distance exists then set current_distance do true as it is the previous distance (we are now in the case d != current_distance)
                        
                        # still I don't understand this **** 
                        if dd in done:
                            done[current_distance][m] = True
                    

                    # initialize next_counts[d] to be a matrix of nan of size (nReplicates,nBins-d)
                    # we are working with the upper triangular so we know that i can't be > nBins-d
                    # so we can use a smaller matrix
                    if d not in next_counts:
                        next_counts[d] = np.zeros((nReplicates,nBins-d))
                        next_counts[d][:] = np.nan
                    
                    # calculate the normalzied count and store it in the matrix count
                    i = int(line[0])
                    j = int(line[1])
                    c = float(line[2])
                    normalized = c / (biases[i]*biases[j])
                    next_counts[d][m,i] = normalized
            else:
                if d not in done:
                    done[d] = [False]*nReplicates
                
                # calculate the normalzied count and store it in the matrix count
                i = int(line[0])
                j = int(line[1])
                c = float(line[2])
                normalized = c / (biases[i]*biases[j])
                counts[m,i] = normalized
    
    
    
    # Let's do the final one!
    # 1. calculate the geometric mean across replicates and write it down
    gmean = scipy.stats.mstats.gmean(remove_zeroes_and_nans(counts), axis=None)
    gmeansfh.write("%d\t%.3f\n" % (current_distance,gmean))
    
    np.seterr(all='raise')  # From here on out, die if you hit a NaN.
    
    # 2. calculate the mean for this disatnce d for each replicate and write it down
    mean = np.nanmean(counts,axis=1)
    for r in range(nReplicates):
        meanfh.write("%d\t%d\t%f\n" % (r,current_distance,mean[r]))
    
    # 3. calculate the actual size factor for this replicate at this genomic distance
    for r in range(nReplicates):
        rCounts = counts[r,:]
        size_factor = np.nanmedian( np.divide( rCounts, gmean ) )
        if np.isnan(size_factor):
            size_factor = np.nan
        size_factors[r,current_distance] = size_factor
    
    
    # Wait! There might be a few more matrices lingering in next_counts. Do them!
    for d in next_counts.keys():
        c = next_counts[d]
        
        # 1. calculate the geometric mean across replicates and write it down
        gmean = scipy.stats.mstats.gmean(remove_zeroes_and_nans(c), axis=None)
        gmeansfh.write("%d\t%.3f\n" % (d,gmean))

        # 2. calculate the mean for this disatnce d for each replicate and write it down
        mean = np.nanmean(c,axis=1)
        for r in range(nReplicates):
             meanfh.write("%d\t%d\t%f\n" % (r,d,mean[r]))

        # 3. calculate the actual size factor for this replicate at this genomic distance 
        for r in range(nReplicates):
            rCounts = next_counts[d][r,:]
            size_factor = np.nanmedian( np.divide( rCounts, gmean ) )
            if np.isnan(size_factor):
                size_factor = np.nan
            size_factors[r,d] = size_factor
    
    
    # Ok, ready to store the just computed size factors in the mats
    for i,m in enumerate(mats):
        m.size_factors = size_factors[i,:]
    
    # we are done here
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
    nReplicates = len(readers)
    means = {}
    variances = {}
    z_factors = {}
    corrected_variances = {}
    # ok now read the files one genomic distance at a time
    n = 0
    current_distance = 0
    done = {}
    normalized_counts = {}
    i_values = {}
    j_values = {}
    z = {}

    for lines in izip_longest(*readers, fillvalue=None):
        for m,line in enumerate(lines):
            if line is None:
                continue

            i = int(line[0])
            j = int(line[1])
            d = int(line[3])
            
            biases = all_biases[m]
            size_factors = all_size_factors[m]

            if d<0:
                logging.info(" !! !! NEGATIVE DISTANCE d=%d" % d)
                continue

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
                        means[current_distance] = np.nan
                        variances[current_distance] = np.nan
                        z_factors[current_distance] = np.nan
                        corrected_variances[current_distance] = np.nan
                    else:
                        # calculate the running means
                        means[current_distance] = running_average_simple(normalized_counts[current_distance], window=smooth_dist)
                        variances[current_distance] = running_variance_simple(normalized_counts[current_distance], window=smooth_dist)
                        
                        for smoothstart,mm in enumerate(means[current_distance]):
                            ii = smoothstart + (smooth_dist-1)/2
                            if i_values[current_distance].shape[0]<= ii or np.isnan(i_values[current_distance][ii][0]):
                                continue
                            assert ii == i_values[current_distance][ii][0]
                            jj = j_values[current_distance][ii][0]
                            vv = variances[current_distance][smoothstart]

                        z_factors[current_distance] = running_z_simple(z[current_distance], means[current_distance], window=smooth_dist)
                        corrected_variances[current_distance] = variances[current_distance] - z_factors[current_distance]
                        
                        # get rid of the stuff we don't need any more
                        del normalized_counts[current_distance]
                        del z[current_distance]
                        del done[current_distance]

                    # update the current distance
                    current_distance = d
                else:
                    done[current_distance][m] = True
            
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
        n = n + 1

    # any leftovers?
    for current_distance in normalized_counts:
        # calculate the running means
        means[current_distance] = running_average_simple(normalized_counts[current_distance], window=smooth_dist)
        variances[current_distance] = running_variance_simple(normalized_counts[current_distance], window=smooth_dist)
        for smoothstart,mm in enumerate(means[current_distance]):
            ii = smoothstart + (smooth_dist-1)/2
            if i_values[current_distance].shape[0]<= ii or np.isnan(i_values[current_distance][ii][0]):
                continue
            assert ii == i_values[current_distance][ii][0]
            jj = j_values[current_distance][ii][0]
            vv = variances[current_distance][smoothstart]
        z_factors[current_distance] = running_z_simple(z[current_distance], means[current_distance], window=smooth_dist)
        corrected_variances[current_distance] = variances[current_distance] - z_factors[current_distance]

    # now collapse the values into 1d arrays why???????
    means_list = []
    variances_list = []
    z_list = []
    corrected_list = []
    for d in means.keys():
        # is it possible to have a single nan somewhere?
        if not np.all((np.isnan(means[d]))):
            means_list.extend(list(means[d]))
            variances_list.extend(list(variances[d]))
            z_list.extend(list(z_factors[d]))
            corrected_list.extend(list(corrected_variances[d]))

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
    
    # calculate the statistics
    means,variances,z_factors,corrected_variances = calc_mean_var(readers,all_biases,all_size_factors,nBins,nDistances,badBins,smooth_dist)
    
    # create an output object containing the statistics
    output = contactCountMatrix(replicates[0].chrMid_to_index, replicates[0].index_to_chrMid, replicates[0].badBins)
    output.binSize = replicates[0].binSize
    output.mean = means
    output.vars = variances
    output.z = z_factors
    output.w = corrected_variances
    output.nReplicates = nReps

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
    
    # Reads an input file containing biases. There are two options:
    # 1) each line has three entries separated by tab or comma:
    #    chr mid bias
    # 2) each line has a single float entry corresponding to the bias in bin
    # the number of line- that is line i gives the bias in bin i
    #
    # If a bias is zero the bin will be marked as a bad bin (having low mappability
    # below the threshold).
    # Bins with bias 0 or not given any bias (nan) will be reasign bias of 1
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

