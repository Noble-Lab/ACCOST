#!/usr/local/bin/python
"""
ACCOST assigns statistical significance to differences in contact counts in Hi-C experiments. 
ACCOST uses a negative binomial to model the contact counts, and pools contacts at the same 
genomic distance to aid in estimating the mean and variance of the data.

"""

import logging
import sys
import os
import csv
import numpy as np
import scipy
from scipy import sparse
from numpy.random import negative_binomial
from scipy.stats import nbinom
from scipy.stats import mstats
from scipy.misc import logsumexp
import contact_counts
from NBfit import LogPolyEstimator
import argparse
import math
import heapq
import tempfile
import shutil
import time
from itertools import izip_longest


def process_row(i,fitted_matrix_A,fitted_matrix_B,A_stats,B_stats,q0,percentile_thresh,dist_thresh_lower,dist_thresh_upper,outfh,outfh2,mats_A,mats_B,smooth_dist):
    nBins = fitted_matrix_A.nBins
    pvals = np.empty(nBins)
    pvals[:] = np.nan
    A_counts = A_stats[0]
    tau_A = A_stats[1]
    phi_A = A_stats[2]
    B_counts = B_stats[0]
    tau_B = B_stats[1]
    phi_B = B_stats[2]


    for j in range(i+1,nBins):

        # filter by distance if required
        dist = abs(i-j)
        if dist < dist_thresh_lower or dist > dist_thresh_upper:
            continue

        # filter by sum of counts if required
        CA = A_counts[j]
        CB = B_counts[j]
        total_counts = CA+CB
        if total_counts < percentile_thresh:
            continue

        # filter nans
        if np.isinf(q0[j]) or np.isnan(q0[j]):
            continue
        
        # filter zeros
        if q0[j] == 0.0:
            continue
        
        # get fitted variance function
        A_estimator = fitted_matrix_A.est
        B_estimator = fitted_matrix_B.est

        # calculate NB params
        f_q0_A = A_estimator.predict(q0[j].reshape(-1,1))
        f_q0_B = B_estimator.predict(q0[j].reshape(-1,1))
     
        mean_A = tau_A[j] * q0[j]
        mean_B = tau_B[j] * q0[j]
        var_A = q0[j] * tau_A[j] + f_q0_A * phi_A[j]
        var_B = q0[j] * tau_B[j] + f_q0_A * phi_B[j] 

        if np.isnan(var_A) or np.isnan(var_B) or var_A == 0 or var_B == 0:
            continue

        p_success_A = mean_A / var_A
        p_success_B = mean_B / var_B
        size_A = ( mean_A * mean_A ) / ( var_A - mean_A )
        size_B = ( mean_B * mean_B ) / ( var_B - mean_B )

        # write to .csv
        outfh2.write("%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" % (i,j,q0[j],f_q0_A,f_q0_B,size_A,size_B,p_success_A,p_success_B,tau_A[j],tau_B[j],phi_A[j],phi_B[j],mean_A,mean_B,var_A,var_B))
         
        # joint probability
        log_p_counts_A = nbinom.logpmf(CA, n=size_A, p=p_success_A)
        log_p_counts_B = nbinom.logpmf(CB, n=size_B, p=p_success_B)
        log_p_counts = log_p_counts_A + log_p_counts_B
        # now see how many combinations are less
        numerator = []
        denominator = []
        for a in range(int(total_counts) + 1):
            b = total_counts - a
            log_p_a = nbinom.logpmf(a, size_A, p_success_A)
            log_p_b = nbinom.logpmf(b, size_B, p_success_B)
            log_p_joint = log_p_a + log_p_b
            if log_p_joint <= log_p_counts:
                numerator.append(log_p_joint)
            denominator.append(log_p_joint)
        
        # calculate pvalue
        if len(numerator)>0 and len(denominator)>0:
            pval = logsumexp(numerator) - logsumexp(denominator)
            pvals[j] = pval
            outfh.write("%d,%d,%f\n" % (i,j,pval))



def pvals_filtered(fitted_matrix_A, fitted_matrix_B, matA_file, matB_file, q0_file, percentile_thresh, dist_thresh_lower, dist_thresh_upper, outfh, outfh2, mats_A, mats_B, smooth_dist):
    assert fitted_matrix_A.fitted and fitted_matrix_B.fitted, "need to fit matrices before calculating pvalues"
    assert fitted_matrix_A.nBins == fitted_matrix_B.nBins

    outfh2.write("i,j,q0[j],f_q0_A,f_q0_B,size_A,size_B,p_success_A,p_success_B,tau_A,tau_B,phi_A,phi_B,mean_A,mean_B,var_A,var_B\n")
    
    matA_reader,matB_reader,q0_reader = contact_counts.open_files((matA_file, matB_file, q0_file))
    current_row = 0
    done_A = {}
    done_B = {}
    done_q = {}
    done_A[0] = False
    done_B[0] = False
    done_q[0] = False
    nBins = fitted_matrix_A.nBins
    A_counts = {}
    B_counts = {}
    q0 = {}
    A_counts[0] = (np.zeros(nBins),np.zeros(nBins),np.zeros(nBins))
    B_counts[0] = (np.zeros(nBins),np.zeros(nBins),np.zeros(nBins))
    q0[0] = np.zeros(nBins)

    np.seterr(divide='ignore', invalid='ignore')

    for aline,bline,qline in izip_longest(matA_reader,matB_reader,q0_reader):
        if aline is not None:
            i = int(aline[0])
            if i > current_row:
                if current_row in done_A and current_row in done_B and current_row in done_q and done_A[current_row] and done_B[current_row] and done_q[current_row]:
                    if current_row in A_counts and current_row in B_counts and current_row in q0:
                        process_row(current_row,fitted_matrix_A,fitted_matrix_B,A_counts[current_row],B_counts[current_row],q0[current_row],percentile_thresh,dist_thresh_lower,dist_thresh_upper,outfh,outfh2,mats_A,mats_B,smooth_dist)
                    # delete the stuff we're done with
                    if current_row in A_counts:
                        del A_counts[current_row]
                    if current_row in B_counts:
                        del B_counts[current_row]
                    if current_row in q0:
                        del q0[current_row]
                    current_row+=1
                else:
                    if i not in done_A:
                        done_A[i] = False
                        A_counts[i] = (np.zeros(nBins),np.zeros(nBins),np.zeros(nBins))
                    done_A[current_row] = True
            j = int(aline[1])
            c = int(aline[2])
            t = float(aline[3])
            f = float(aline[4])
            if i not in done_A:
                done_A[i] = False
                A_counts[i] = (np.zeros(nBins),np.zeros(nBins),np.zeros(nBins))
            A_counts[i][0][j] = c
            A_counts[i][1][j] = t
            A_counts[i][2][j] = f
        
        if bline is not None:
            i = int(bline[0])
            if i > current_row:
                if i not in done_B:
                    done_B[i] = False
                    B_counts[i] = (np.zeros(nBins),np.zeros(nBins),np.zeros(nBins)) 
                done_B[current_row] = True
            j = int(bline[1])
            c = int(bline[2])
            t = float(bline[3])
            f = float(bline[4])
            if i not in done_B:
                done_B[i] = False
                B_counts[i] = np.zeros(nBins)
            B_counts[i][0][j] = c
            B_counts[i][1][j] = t
            B_counts[i][2][j] = f
        
        if qline is not None:
            i = int(qline[0])
            if i > current_row:
                if i not in done_q:
                    done_q[i] = False
                    q0[i] = np.zeros(nBins)
                done_q[current_row] = True
            j = int(qline[1])
            q = float(qline[2])
            if i not in done_q:
                done_q[i] = False
                q0[i] = np.zeros(nBins)
            q0[i][j] = q
    
    # process left overs only if it has the full set A,B,q0
    for i in sorted(A_counts.keys()):
        if i not in B_counts:
            B_counts[i] = (np.zeros(nBins),np.zeros(nBins),np.zeros(nBins))
        if i not in q0:
            q0[i] = np.zeros(nBins)
        process_row(i,fitted_matrix_A,fitted_matrix_B,A_counts[i],B_counts[i],q0[i],percentile_thresh,dist_thresh_lower,dist_thresh_upper,outfh,outfh2,mats_A,mats_B,smooth_dist)
    
    outfh.close()
    outfh2.close()

    


def fit_mat(matrix, length_frac = 2./3):
    length = int(len(matrix.w) * length_frac)
    logging.info("only fitting mean/variance up to genomic distance %.3f" % length)
    w, vars, means, z = matrix.w[:length], matrix.vars[:length], matrix.mean[:length], matrix.z[:length]

    X = means
    y = vars - z
 
    # filter nan
    nans = np.logical_or(np.isnan(X),np.isnan(y))
    X = X[np.where(~nans)]
    y = y[np.where(~nans)]
    
    # filter negatives
    positives = np.logical_and(X>0,y>0)
    X = X[np.where(positives)]
    y = y[np.where(positives)]
        
    est = LogPolyEstimator(degree=2)
    est.fit(X, y)

    matrix.est = est
    matrix.fitted = True
    
    return matrix



def read_id_file(idfile,id_A,id_B):
    fh = open(idfile,'r')
    reader = csv.reader(fh,delimiter="\t")
    Afiles = []
    Bfiles = []
    for line in reader:
        lineid,countfile,biasfile = line
        if lineid == id_A:
            Afiles.append((countfile,biasfile))
        elif lineid == id_B:
            Bfiles.append((countfile,biasfile))
        else:
            raise Exception("ID %s in IDfile doesn't match ID_A (%s) or ID_B (%s)" % (lineid,id_A,id_B))
    logging.info("Read %d pairs of files from %s." % (len(Afiles), idfile))
    return Afiles,Bfiles
        


def set_up_argparser():
    parser = argparse.ArgumentParser()
    # required params
    parser.add_argument("binsize", type=int, help="the bin size of the input file")
    parser.add_argument("binfile", help="bin and mappability file")
    parser.add_argument("idfile", help="input file of count and bias filenames")
    parser.add_argument("id_A", help="ID for class A")
    parser.add_argument("id_B", help="ID for class B")
    # optional params
    parser.add_argument("-d", "--distances", default=None, help="pregenerated matrix with distance, to speed up calculations")
    parser.add_argument("-dr", "--distance_reverse", default=None, help="reverse matrix with distance, to speed up calcaulations")
    parser.add_argument("-o", "--output_prefix", default="differential_output", help="the prefix for output. Can be a full path, e.g. /put/my/input/here_ will produce files with the prefix here_ in the directory input")
    parser.add_argument("-mind", "--min_dist", type=int, default=0, help="the lower threshold for distance (in bins, default 0)")
    parser.add_argument("-maxd", "--max_dist", type=int, default=10000, help="the upper threshold for length (in bins, default 10000)")
    parser.add_argument("--filter_diagonal", type=bool, default=True, help="filter counts on the diagonal?" )
    parser.add_argument("-m", "--map_thresh", type=float, default=0.25, help="the mappability threshold below which to ignore bins (default 0.25)")
    parser.add_argument("-p", "--min_percentile", type=int, default=80, help="the count percentile threshold, below which to ignore counts (default 80 (%%))")
    parser.add_argument("-ds", "--dist_smooth", type=int, default=10, help="the number of locus pairs to smooth over for calculating mean/variance (default 10)")
    parser.add_argument("-om", "--output_matrix", help="Whether to output the full, dense matrix of P-values (rather than just those passing a threshold) (default: F)")
    parser.add_argument("--output_p_thresh")
    parser.add_argument("--output_q_thresh")
    parser.add_argument("--no_dist_norm", action='store_true', help="Disable distance size factors, for testing")
    return parser
    



def main():

    # Set up the logger, including printing messages to stderr.
    logging.basicConfig(filename='output.log',format='%(asctime)s %(levelname)s from %(filename)s %(funcName)s :: %(message)s', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    # Parse the user inputs
    parser = set_up_argparser()
    args = parser.parse_args()
    
    binfile = args.binfile
    binsize = args.binsize
    idfile = args.idfile
    id_A = args.id_A
    id_B = args.id_B

    # pre generated matrix with distance
    precalculated_length_file = args.distances
    precalculated_length_file_reversed = args.distance_reverse
    
    # parameters
    mappability_thresh = args.map_thresh
    dist_thresh_lower = args.min_dist
    dist_thresh_upper = args.max_dist
    percentile_thresh = args.min_percentile
    fit_frac = 0.9 # TODO: expose this
    smooth_dist = args.dist_smooth
    filter_diagonal = args.filter_diagonal
    no_dist_norm = args.no_dist_norm    
    total_counts_eps = 1 # TODO: expose this
    
    # output
    outprefix = args.output_prefix
    
    # set up the temp directory to write all the files we will be using
    tempfile.tempdir = tempfile.mkdtemp()
    logging.info("temp directory currently set to: %s" % tempfile.tempdir)

    # set up bins
    (allBins, allBins_reversed, badBins) = contact_counts.generate_bins(binfile, mappability_thresh)
    
    # read input file setting up class A and class B
    Afiles,Bfiles = read_id_file(idfile,id_A,id_B)
   
    # Read input contact count matrices
    logging.info("Loading data")
    mats_A = []
    mats_B = []
    for (matfile_A,biasfile_A) in Afiles:
        matrixA = contact_counts.contactCountMatrix(allBins, allBins_reversed, badBins)
        matrixA.infile = matfile_A
        matrixA.load_biases(biasfile_A)
        matrixA.nDistances = matrixA.nBins #TODO: this should change
        matrixA.binSize = binsize
        # store the processed matrix into the corresponding array mats_A
        mats_A.append(matrixA)
    for (matfile_B,biasfile_B) in Bfiles:
        matrixB = contact_counts.contactCountMatrix(allBins, allBins_reversed, badBins)
        matrixB.infile = matfile_B
        matrixB.load_biases(biasfile_B)
        matrixB.nDistances = matrixB.nBins #TODO: this should change
        matrixB.binSize = binsize
        # store the processed matrix into the corresponding array mats_A
        mats_B.append(matrixB)  

    mats = mats_A + mats_B
    logging.info("Found %d matrix files." % len(mats))
    
    # Make sure all matrices have the same number of bins
    nBins = mats[0].nBins
    nDistances = nBins
    logging.info("Total bins: %d" % nBins)
    for m in mats:
        assert m.nBins == nBins
    


    # For each matrix sort the counts by genomic distance and write them to a tempfile we are going to use later
    logging.info("Sorting data by genomic distance")
    for m in mats:
        # calculate the genomic distance between every two bins and the total number of counts in the matrix
        tmp_length_indexed = tempfile.NamedTemporaryFile(delete=False,prefix="tmp_length_indexed_")
        total_counts = contact_counts.length_index_contact_counts(m.infile,tmp_length_indexed,binsize,index_to_chrMid=allBins_reversed,chrMid_to_index=allBins)
        logging.info("Total counts in %s is %d." % (m.infile, total_counts))

        # sort the file by genomic distance
        tmp_sorted_by_length = tempfile.NamedTemporaryFile(delete=False,prefix="tmp_sorted_by_length_")
        contact_counts.sort_contactfile_by_length(tmp_length_indexed.name,tmp_sorted_by_length)

        # sort the file by counts
        tmp_sorted_by_index = tempfile.NamedTemporaryFile(delete=False,prefix="tmp_sorted_by_index_")
        contact_counts.sort_contactfile_by_index(tmp_length_indexed.name,tmp_sorted_by_index)


        m.infile_sorted_by_length = tmp_sorted_by_length.name
        m.infile_sorted_by_index = tmp_sorted_by_index.name
        m.infile_length_indexed = tmp_length_indexed.name
        
    

    # Now for each genomic distance we need to calculate:
    #   - the sum of the normalized contact counts
    #   - the sum of the square of the normalized contact counts
    #   - the number of observations at that genomic distance
    
  
    # First, take care of size factors
    if no_dist_norm:
        logging.info("Skipping calculation of size factors")
        for mat in mats:
            mat.size_factors = None
    else:
        logging.info("Calculating size factors")
        mats = contact_counts.calculate_size_factors(mats,nBins,nDistances)
        logging.info("Filling nans in size factors with 1")
        for m in mats:
            m.size_factors[np.isnan(m.size_factors)] = 1
        for i in range(len(mats_A)):
            pass
            #np.savetxt("size_factors_A%d.txt"%i,mats_A[i].size_factors)
        for i in range(len(mats_B)):
            pass
            #np.savetxt("size_factors_B%d.txt"%i,mats_B[i].size_factors)


    logging.info("Calculating means and variances") 
    matrixA = contact_counts.combine_replicates_and_calculate_mean_variance(mats_A,smooth_dist)
    matrixB = contact_counts.combine_replicates_and_calculate_mean_variance(mats_B,smooth_dist)
    logging.info("Done calculating means and variances.")
    

    # get total counts for A condition and B condition 
    sumA_fh = tempfile.NamedTemporaryFile(delete=False,prefix="tmp_sumA_")
    contact_counts.sum_mats(mats_A,sumA_fh)
    
    sumB_fh = tempfile.NamedTemporaryFile(delete=False,prefix="tmp_sumB_")
    contact_counts.sum_mats(mats_B,sumB_fh)
    
    
    logging.info("Fitting means & variances for A")
    matrixA = fit_mat(matrixA, fit_frac)

    logging.info("Fitting means and variances for B")
    matrixB = fit_mat(matrixB, fit_frac)

     
    logging.info("calculating common mean q0") 
    q0_fh = tempfile.NamedTemporaryFile(delete=False,prefix="tmp_q0_")
    contact_counts.get_q0(mats,q0_fh,no_dist_norm)
    
    sum_fh = tempfile.NamedTemporaryFile(delete=False,prefix="tmp_sum_all_")
    contact_counts.sum_mats(mats,sum_fh)
    

    logging.info("get %.2f percentile of the contact counts" % percentile_thresh)
    max_perc = contact_counts.get_percentile(sum_fh.name, allBins_reversed, 100-percentile_thresh)
    logging.info("%dth percentile is: %.3f" % (percentile_thresh,max_perc))


    # now calculate the p-values
    logging.info("Calculating pvalues")
    outfh  = open(str(outprefix + "_ln_pvals.txt"),'w')
    outfh2 = open(str(outprefix + "_stats.csv"),'w')
    pvals_filtered(matrixA, matrixB, sumA_fh.name, sumB_fh.name, q0_fh.name, max_perc, dist_thresh_lower, dist_thresh_upper, outfh, outfh2, mats_A, mats_B, smooth_dist)
    
    # clean up
    shutil.rmtree(tempfile.tempdir, ignore_errors=True)
    os.remove("output.log")

    print("Done!")


if __name__ == "__main__":
    main()
