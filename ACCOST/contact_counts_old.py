import numpy as np
import csv
import gzip
import logging
import copy
import scipy
from copy import deepcopy
from scipy import sparse
from scipy.sparse import coo_matrix
import sklearn.cross_validation
import heapq


class contactCountMatrix:
    """
    Class to represent contact count matrices. Has functionality for modeling distribution of counts.

    Args:
       allBins: dictionary of chromosome + midpoints to bin index
       allBins_reversed: list of indices, each pointing to a (chr,mid) tuple
       badBins: list of indices of bad (low mappability) bins
    """

    def __init__(self, allBins, allBins_reversed, badBins):
        self.allBins = allBins
        self.allBins_reversed = allBins_reversed
        self.badBins = badBins
        self.data = None
        self.biases = None
        self.vars = []
        self.mean = []
        self.z = []
        self.w = []
        self.heap = []

    def load_from_file(self, counts_filename, bias_filename, bin_size, filter_diagonal=False, total_counts=1):
        """
        Load a contact count matrix from a file.

        Matrix should be in tab delimited text format:

        chr10    5000    chr10    5000    2
        chr10    5000    chr10    15000    6
        chr10    5000    chr10    105000    2

        Gzipped (detected by having '.gz' at the end of the filename) data is allowed.

        Args:
           counts_filename: the name of the file containing the matrix to load
           bias_filename: the name of the file containing the bias of the contact counts
           length: the maximum length of the chromosome
           bin_size: the bin size of the input data
        """
        if counts_filename.endswith('.gz'):
            fh = gzip.open(counts_filename)
        else:
            fh = open(counts_filename)
        reader = csv.reader(fh, delimiter='\t')
        
        nBins = len(self.allBins_reversed)
        
        # first load the biases
        logging.info('Loading biases from %s' % bias_filename)
        # set up array to hold the biases
        bias_array = np.empty((nBins,1))
        logging.debug(nBins)
        bias_array[:] = np.nan
        logging.debug(nBins)
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
                bias_array[self.allBins[chrom][mid]] = bias
                # check for zeros and add to bad bins
                if bias==0 and self.allBins[chrom][mid] not in self.badBins:
                    self.badBins.append(self.allBins[chrom][mid])
            # otherwise assume there's a single float per line that represents the bias
            else:
                bias = float(line)
                bias_array[i] = bias
                # check for zeros and add to bad bins
                if bias==0 and i not in self.badBins:
                    self.badBins.append(i)
            i += 1
        
        # now load the data
        logging.info('Loading matrix from %s', counts_filename)
        nBins = len(self.allBins_reversed)
        
        # first set up zeros for each genomic distance
        raw_counts_by_length = [[]]*(nBins+1)
        recip_biases_by_length = [[]]*(nBins+1)
        n_binpairs = [0]*(nBins+1)
        start_indices = {} # by chr then by length
        for chrom in self.allBins:
            start_indices[chrom] = [0] * len(self.allBins[chrom])
            for l in range(len(self.allBins[chrom])):
                if l in n_binpairs:
                    start_indices[chrom][l] = n_binpairs[l] # the values for this chromosome start here
                else:
                    start_indices[chrom][l] = 0 
                n_binpairs[l] += len(self.allBins[chrom]) - l
        for l in range(nBins):
            raw_counts_by_length[l] = [0] * n_binpairs[l]
            recip_biases_by_length[l] = [0] * n_binpairs[l]
        
        
        n = 0
        rows = []
        cols = []
        counts = []
        
        # Nx4 matrix, where N is the number of genomic distances
        cumulate_data = np.zeros((4, nBins + 1))
        # row 0: normalized count
        # row 1: sum
        # row 2: sum of squares
        # row 3: sum of reciprocal bias
        # column: genomic distance/resolution. last column = interchromosomal counts
       
        
        # now load the data
        length_counts = {}
        for line in reader:
            (chr1, mid1, chr2, mid2, count) = line
            mid1 = int(mid1)
            mid2 = int(mid2)
            count = float(count)

            # enforce lower triangular
            if chrgreater(chr1, chr2) or (chr1 == chr2 and int(mid1) > int(mid2)):
                chrtemp = chr1
                chr1 = chr2
                chr2 = chrtemp
                midtemp = mid1
                mid1 = mid2
                mid2 = midtemp
            # paranoid assertions here, these are probably unneccessary
            assert (chrgreater(chr2,chr1) or ( chr1 == chr2 and mid1 <= mid2) ), "Matrix should be lower triangular (chrs: %s %s)" % (chr1, chr2)
            assert self.allBins[chr1][mid1] <= self.allBins[chr2][mid2], "???? bin indices not lower triangular %d %d" % (self.allBins[chr1][mid1], self.allBins[chr2][mid2])
            
            # filter out bad bins
            if self.allBins[chr1][mid1] in self.badBins or self.allBins[chr2][mid2] in self.badBins:
                continue
            if filter_diagonal and chr1==chr2 and mid1==mid2:
                continue
            if np.isnan(bias_array[self.allBins[chr1][mid1]]) or np.isnan(bias_array[self.allBins[chr2][mid2]]):
                continue
            
            # TODO: make it optional whether to store all counts
            rows.append(self.allBins[chr1][mid1])
            cols.append(self.allBins[chr2][mid2])
            counts.append(count)
            n += 1 # TODO: decide whether this should consider things that are masked out
           
            # figure out where to put the data
            # length_index is the genomic distace / bin size
            # start_index is where this chromosome starts (for this genomic distance)
            # count_index is where this particular contact count goes 
            if chr1 != chr2:
                length_index = 0 # interchromosomal
                start_index = 0
            else:
                length_index = abs((int(mid2) - int(mid1)) / bin_size)
                start_index = start_indices[chr1][length_index]
            count_index = start_index + int((mid1-1)/bin_size)
            
            # calculate sum of normalized counts & sum of squares of normalized counts
            recip_bias = 1. / ( bias_array[self.allBins[chr1][mid1],0] * bias_array[self.allBins[chr2][mid2],0] )
            normalized_count = count * recip_bias
            if not np.isnan(count):
                cumulate_data[0,length_index] += 1
                cumulate_data[1,length_index] += normalized_count
                cumulate_data[2,length_index] += normalized_count ** 2
                cumulate_data[3,length_index] += recip_bias
                #logging.debug("%s %s %d %d length_index: %d start_index: %d count_index: %d" % (chr1,chr2,mid1,mid2,length_index,start_index,count_index))
                #logging.debug("%d %d" % (len(raw_counts_by_length[length_index]),len(recip_biases_by_length[length_index])))
                raw_counts_by_length[length_index][count_index] = count
                recip_biases_by_length[length_index][count_index] = recip_bias
                # update the heap
                #if index != length:
                #    if len(self.heap) < heap_size:
                #        heapq.heappush(self.heap, int(count))
                #    else:
                #        heapq.heappushpop(self.heap, int(count))

        for i in range(len(self.allBins_reversed)):
            rows.append(i)
            cols.append(0)
            counts.append(0.0)
            cols.append(i)
            rows.append(0)
            counts.append(0.0)
        if filter_diagonal:
            cumulate_data = cumulate_data[:,1:(nBins+1)]
        data = scipy.sparse.coo_matrix((counts, (rows, cols)))
        logging.debug("min and max of rows: %d %d" % (np.min(rows), np.max(cols)))
        logging.debug("min and max of cols: %d %d" % (np.min(cols), np.max(cols)))
        logging.debug("data shape: " + str(np.shape(data)))
        logging.debug("dense data shape: " + str(np.shape(data.todense())))
        self.data = scipy.sparse.csc_matrix(data).todense()
        #np.savetxt("data.txt",self.data,delimiter="\t",fmt="%d")
        #np.savetxt("norm.txt",np.divide(self.data,np.outer(bias_array,bias_array)),delimiter="\t")
        #np.savetxt("biasmat.txt",np.outer(bias_array,bias_array),delimiter="\t")
        #np.savetxt("cumulate_data.txt",np.transpose(cumulate_data),fmt=["%d","%.4e","%.4e","%.4e"],delimiter="\t")
        self.cumulate_data = cumulate_data
        self.N = n
        self.N_perlength = cumulate_data[0,:]
        self.biases = bias_array
        self.raw_counts_by_length = raw_counts_by_length
        self.recip_biases_by_length = recip_biases_by_length
        self.nBins = nBins
        return(self)

    def calculate_mean_variance_old(self):
        cumulate_data = self.cumulate_data
        nBins = len(self.allBins_reversed) 
        # calculate the mean and adjusted variance
        result = np.zeros((4, cumulate_data.shape[1])) # rows are mean, variance, z, w
        for x in range(0, cumulate_data.shape[1]):
            if cumulate_data[0][x] > 1:
                mean = 1.0 * cumulate_data[1][x] / cumulate_data[0][x]
                variance = 1.0 * (cumulate_data[2][x] - (( cumulate_data[1][x]**2 ) / cumulate_data[0][x] / self.N)) / (cumulate_data[0][x] - 1)
                z = (mean / cumulate_data[0][x]) * cumulate_data[3][x]
            else:
                mean = np.nan
                variance = np.nan
                z = np.nan
            result[0][x] = mean
            result[1][x] = variance
            result[2][x] = z
            result[3][x] = variance - z
        self.mean = result[0]
        self.vars = result[1]
        self.z = result[2]
        self.w = result[3]
        return self


    
    def load_ICE_biases_vec(self, filename):
        """
        Load an ICE bias vector from a raw file, one bias per line.

        Gzipped (detected by having '.gz' at the end of the filename) data is allowed.

        This function sets the biases and biasBins attributes of the contact count matrix.
        biasBins is a list of boolens corresponding to indices where the biases are defined.

        Args:
            filename: the name of the file containing the bias vector

        """
        logging.info("loading biases from %s" % filename)
        biases = np.loadtxt(filename)
        self.biases = biases
        logging.debug("biases shape: " + str(np.shape(biases)))
        logging.debug("data shape: " + str(np.shape(self.data)))
        biasBins = np.array(range(len(biases)))
        biasMask = np.array([False] * len(biases))
        self.biasBins = biasBins
        self.biasMask = biasMask

    def load_ICE_biases(self, filename, trim_counts=False):
        """
        Load an ICE bias vector from a file.

        Biases should be in tab delimited text format:

        chr1    5000    1.0
        chr1    15000    1.0
        chr1    25000    1.0
        chr1    35000    0.710793549712
        chr1    45000    0.313981161587

        Gzipped (detected by having '.gz' at the end of the filename) data is allowed.

        This function sets the biases and biasBins attributes of the contact count matrix.
        biasBins is a list of boolens corresponding to indices where the biases are defined.

        Args:
            filename: the name of the file containing the bias vector
        """
        logging.info("loading biases from %s" % filename)
        if filename.endswith('.gz'):
            fh = gzip.open(filename)
        else:
            fh = open(filename)
        reader = csv.reader(fh, delimiter='\t')
        n = 0
        biases = np.zeros(len(self.allBins_reversed))
        biasBins = []

        for line in reader:
            (chr, mid, bias) = line
            mid = int(mid)
            if mid in self.allBins[chr]:
                index = self.allBins[chr][mid]
                biases[index] = bias
                biasBins.append(index)
                n += 1
        biasMask = []
        for i in range(np.shape(self.data)[0]):
            if i in biasBins:
                biasMask.append(False)
            else:
                biasMask.append(True)
        self.biases = np.array(biases)
        logging.debug("biases shape: " + str(np.shape(biases)))
        logging.debug("data shape: " + str(np.shape(self.data)))
        self.biasBins = biasBins
        self.biasMask = biasMask

    def mask_matrix(self, mask_zeros=False, mask_no_bias=True, mask_low_mappability=False):
        """Mask bins without a bias or with low mappability bins"""
        # only use counts for which a bias is known
        biasMask = np.maximum.outer(self.biasMask, self.biasMask)
        # mask zeros
        if mask_zeros:
            zeroCounts = self.data != 0
            zeroCounts = ~(zeroCounts.todense())
            mask = np.logical_or(biasMask, zeroCounts)
        else:
            mask = biasMask

        masked_counts = np.ma.masked_array(self.data.todense(), mask)

        # mask rows and columns in bad bins (poor mappability)
        if mask_low_mappability:
            masked_counts[self.badBins, :] = np.ma.masked
            masked_counts[:, self.badBins] = np.ma.masked

        self.masked_counts = masked_counts
        self.masked = True

 


def generate_binpairs(allBins):
    """
    Generate all bin pairs from a dictionary of bins.

    Args:
       allBins: dictionary of chr to mid to index, generated by generate_bins

    Returns:
       binpairs: dictionary of chr1 to mid1 to chr2 to mid2 to 0
    """
    raise NotImplementedError("Shouldn't need this!")
    logging.info('Generating bin pairs')
    binpairs = {}
    binpairs_reversed = []
    for chr1 in allBins.keys():
        binpairs[chr1] = {}
        for mid1 in allBins[chr1].keys():
            binpairs[chr1][mid1] = {}
            for chr2 in allBins.keys():
                if chr1 <= chr2:
                    binpairs[chr1][mid1][chr2] = {}
                    for mid2 in allBins[chr2].keys():
                        if chr1 < chr2 or (chr1 == chr2 and mid1 <= mid2):
                            binpairs[chr1][mid1][chr2][mid2] = 0
                            binpairs_reversed.append((chr1, mid1, chr2, mid2))
    return binpairs, binpairs_reversed


def generate_bins(midsFile, lowMappThresh):
    """
    Generate data structures to hold binned data and keep track of "bad" (low mappability) bins.

    Args:
       midsFile: name of the file with bin midpoints and mappability data. Tab delimited file with format:
           <chr ID>        <mid>   <anythingElse> <mappabilityValue> <anythingElse>+
               chr10   50000   NA      0.65    ...
       lowMappThresh: threshold at which we want to discard those bins

    Returns:
       allBins: dictionary of chromosome + midpoints to bin index
       allBins_reversed: list of indices, each pointing to a (chr,mid) tuple
       badBins: list of indices of bad (low mappability) bins

    """
    badBins = []  # list of indices with mappabilties < lowMappThresh
    allBins = {}  # chr and mid to index
    allBins_reversed = []  # index to chr and mid

    fh = open(midsFile)
    reader = csv.reader(fh, delimiter='\t')
    i = 0
    logging.info('Reading bin midpoints and mappability from %s', midsFile)
    for line in reader:  # might need a check here to avoid header line
        chr = line[0]
        mid = int(line[1])
        mappability = float(line[3])
        if chr not in allBins:
            allBins[chr] = {}
        allBins[chr][mid] = i
        allBins_reversed.append((chr, mid))
        if mappability <= lowMappThresh:
            badBins.append(i)
        i += 1
    fh.close()
    return (allBins, allBins_reversed, badBins)


def dict_to_sparse(counts, allBins_reversed):
    """
    Convert count matrix in nested dictionary format to sparse scipy coordinate matrix format.

    Args:
       counts: nested dictionary of counts (ie data[chr1][mid1][chr2][mid2])
       allBins_reversed: index of ordered bins. list of (chr,mid) tuples

    Returns:
       sparse_counts: sparse format matrix in scipy COO (coordinate) format
    """
    raise NotImplementedError("Shouldn't need this!")
    rows = []
    cols = []
    data = []
    n = 0
    for i, (chr1, mid1) in enumerate(allBins_reversed):
        for j, (chr2, mid2) in enumerate(allBins_reversed):
            if mid1 in counts[chr1]:
                if chr1 <= chr2 and mid2 in counts[chr1][mid1][chr2]:
                    if counts[chr1][mid1][chr2][mid2] > 0:
                        rows.append(i)
                        cols.append(j)
                        data.append(counts[chr1][mid1][chr2][mid2])
                        n += 1
                        logging.info("%04d %04d %04.4f" % (i, j, counts[chr1][mid1][chr2][mid2]))
    logging.debug("n= %4d" % n)
    sparse_counts = sparse.coo_matrix(
        (np.array(data, dtype=np.float64), (np.array(rows, dtype=np.int16), np.array(cols, dtype=np.int16))),
        dtype=np.float64)
    (r, c, d) = sparse.find(sparse_counts)
    logging.info(r[1:10])
    logging.info(c[1:10])
    logging.info(d[1:10])
    return sparse_counts


def sparse_to_dict(counts_sparse, allBins, allBins_reversed):
    """
    Convert contact count matrix in scipy sparse format to nested dictionary format.

    Args:
        counts_sparse: scipy sparse matrix of contact counts

    Returns:
        counts: dictionary of contact counts
    """
    raise NotImplementedError("Shouldn't need this!")
    logging.info("counts_sparse shape: " + str(np.shape(counts_sparse)))
    logging.info("allBins length: %04d" % len(allBins))
    logging.info("allBins_reversed length: %04d" % len(allBins_reversed))
    counts, binpairs_reversed = generate_binpairs(allBins)
    counts_sparse_csc = counts_sparse.tocsc()
    for i, (chr1, mid1) in enumerate(allBins_reversed):
        for j, (chr2, mid2) in enumerate(allBins_reversed):
            if chr1 < chr2 or (chr1 == chr2 and mid1 <= mid2):
                if i < np.shape(counts_sparse_csc)[0] and j < np.shape(counts_sparse_csc)[1]:
                    counts[chr1][mid1][chr2][mid2] = counts_sparse_csc[i, j]
                    logging.debug("%04d, %04d, %04.4f" % (i, j, counts_sparse_csc[i, j]))
    return counts


def get_lengths(allBins_reversed):
    """
    Calculate distance between all bins (not necessarily with
    a count matrix attached)

    Distance is defined by: | mid2 - mid1 |

    Interchromosomal lengths are set to -1

    Args:
        TODO
    Returns:
        TODO
    """
    logging.info('Generating lengths')
    n = len(allBins_reversed)
    lengths = np.zeros((n, n))
    lengths_reversed = {}
    lengths_reversed[-1] = []  # we know this one exists!
    for (i, (chr1, mid1)) in enumerate(allBins_reversed):
        for (j, (chr2, mid2)) in enumerate(allBins_reversed):
            if chr1 != chr2:
                lengths[i, j] = -1
                lengths_reversed[-1].append((i, j))
            else:
                dist = abs(mid2 - mid1)
                # print(mid1,mid2,dist)
                lengths[i, j] = dist
                if dist in lengths_reversed:
                    lengths_reversed[dist].append((i, j))
                else:
                    lengths_reversed[dist] = [(i, j)]
    return lengths, lengths_reversed


def get_lengths_for_matrix(allBins_reversed, mat, check_bias_indices=False):
    """
    Calculate distance between all bins in the full count matrix.

    Distance is defined by: | mid2 - mid1 |

    Interchromosomal lengths are set to -1

    Args:
        TODO
    Returns:
        TODO
    """
    logging.info('Generating lengths')
    n = len(allBins_reversed)
    lengths = np.zeros((n, n))
    lengths_reversed = {}
    lengths_reversed[-1] = []  # we know this one exists!
    for (i, (chr1, mid1)) in enumerate(allBins_reversed):
        for (j, (chr2, mid2)) in enumerate(allBins_reversed):
            if i < np.shape(mat.data)[0] and j < np.shape(mat.data)[1]:
                if check_bias_indices and (i in mat.biasBins and j in mat.biasBins):
                    if chr1 != chr2:
                        lengths[i, j] = -1
                        lengths_reversed[-1].append((i, j))
                    else:
                        dist = abs(mid2 - mid1)
                        lengths[i, j] = dist
                        if dist in lengths_reversed:
                            lengths_reversed[dist].append((i, j))
                        else:
                            lengths_reversed[dist] = [(i, j)]
    return lengths, lengths_reversed

def checkEqual(lst):
    """
    Checks to make sure all items in a list are equal.
    """
    return not lst or lst.count(lst[0]) == len(lst)

def check_replicates(replicates):
    """
    Checks to make sure a set of replicates are equivalent (same bins)
    
    Args:
        replicates - a list of contactCountMatrix objects
    Returns:
        True if the 
        False otherwise
    
    """
    if not checkEqual ([x.nBins for x in replicates]):
        return False
    if not checkEqual([x.allBins for x in replicates]):
        return False
    if not checkEqual([x.badBins for x in replicates]):
        return False
    if not checkEqual([x.allBins_reversed for x in replicates]):
        return False
    return True



def combine_replicates_old(replicates):
    check_replicates(replicates)
    nReps = len(replicates)
    output = copy.deepcopy(replicates[0])
    for i in range(1,nReps):
        cur = replicates[i]
        output.cumulate_data += cur.cumulate_data
        output.N += cur.N
        output.data += cur.data
    output.biases = None # this no longer makes sense
    return output

def combine_replicates_and_calculate_mean_variance(replicates):
    assert check_replicates(replicates)
    nReps = len(replicates)
    output = copy.deepcopy(replicates[0])
    nBins = replicates[0].nBins
    means = np.zeros(nBins+1,)
    variances = np.zeros(nBins+1,)
    z_factors = np.zeros(nBins+1,)
    corrected_variances = np.zeros(nBins+1,)
    
    # first calculate means
    for l in range(0,nBins+1): # loop over lengths
        sum_normalized_counts = 0.
        total_counts = 0.
        for i in range(nReps): # loop over replicates
            cur = replicates[i]
            raw_counts_by_length = cur.raw_counts_by_length
            recip_biases_by_length = cur.recip_biases_by_length
            size_factors = cur.size_factors
            nCounts = len(raw_counts_by_length[l])
            total_counts += nCounts
            if nCounts>1:
                for j in range(nCounts):
                    norm_count = (raw_counts_by_length[l][j])/((recip_biases_by_length[l][j])*size_factors[l])
                    if np.isfinite(norm_count):
                        sum_normalized_counts += norm_count
        mean = 0
        if total_counts > 0:
            mean = sum_normalized_counts / total_counts
        means[l] = mean

    # now do variances
    for l in range(0,nBins+1):
        sumvar = 0.
        sumz = 0.
        total_counts = 0.
        for i in range(nReps): # loop over replicates
            raw_counts_by_length = cur.raw_counts_by_length
            recip_biases_by_length = cur.recip_biases_by_length
            size_factors = cur.size_factors
            nCounts = len(raw_counts_by_length[l])
            total_counts += nCounts
            if nCounts > 1:
                for j in range(nCounts):
                    vadd = (raw_counts_by_length[l][j]/(recip_biases_by_length[l][j]*size_factors[l]) - means[l])**2
                    if np.isfinite(vadd):
                        sumvar += vadd
                    zadd =  1./(recip_biases_by_length[l][j]*size_factors[l])
                    if np.isfinite(zadd):
                        sumz += zadd
        variance = 0.
        z_factor = 0.
        if total_counts > 0:
            variance = sumvar / (total_counts-1)
            z_factor = means[l] * sumz / nCounts
        variances[l] = variance
        z_factors[l] = z_factor
        corrected_variances[l] = variance - z_factor
    # now calculate the total data and N
    # TODO: is this necessary? can get rid of cumulate data?
    for i in range(1,nReps):
        output.cumulate_data += cur.cumulate_data
        output.N += cur.N
        output.data += cur.data
    output.mean = means
    output.vars = variances
    output.z = z_factors
    output.w = corrected_variances
    output.biases = None # this no longer makes sense
    return output
 

def chrgreater(chr1,chr2):
    if not (chr1[0:3] == "chr" and chr2[0:3] == "chr"):
        raise Exception("Error: don't understand chrs %s %s" % (chr1,chr2))
    chrnum1 = chr1[3:]
    chrnum2 = chr2[3:]
    if chrnum1.isdigit() and chrnum2.isdigit():
        return int(chrnum1) > int(chrnum2)
    if chrnum1 == 'Y':
        if chrnum2 =='Y':
            return False
        return True
    if chrnum1 == 'X':
        if chrnum2 == 'X' or chrnum2 == 'Y':
            return False
        return True
    if chrnum1.isdigit() and (chrnum2 == 'X' or chrnum2 == 'Y'):
        return False
    raise Exception("Error: don't know what to do with chr numbers %s %s" % (chrnum1,chrnum2))



