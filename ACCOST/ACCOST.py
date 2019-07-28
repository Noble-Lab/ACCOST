#!/usr/local/bin/python
"""
differential_counts.py

Module to test differential contact counts.

"""

#TODO clean up imports
import warnings
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
# from test-fit-variance import set_up_vars_means
import contact_counts
#from NBfit import LogPolyEstimator, LocfitEstimator
from NBfit import LogPolyEstimator, LowessEstimator
import argparse
import math
import heapq
import tempfile
from itertools import izip_longest

#TODO is this still used? delete
MAX_VARIANCE = 10 ** 8

#TODO clean up unused functions
def get_null_mean(mats, q0):
    biasmats = np.array([np.outer(x.biases,x.biases) for x in mats])
    sum_biasmats = np.sum(biasmats,axis=0)
    return q0*sum_biasmats

def get_null_variance(mats, q0, f_q0):
    mat_size = len(mats[0].biases)
    out = np.zeros((mat_size,mat_size))
    for matrix in mats:
        biasmat = np.outer(matrix.biases,matrix.biases)
        out += biasmat*q0 + biasmat*biasmat*f_q0
    return out

def calc_p_r(mu, sigma):
    assert False, "not used"
    r = np.divide(np.multiply(mu,mu) , (sigma - mu))
    p =  mu / sigma
    # make sure they're positive (well, > some small number)
    r[r > (1/smallval)] = 1/smallval
    #p = np.clip(p, smallval, 1 - smallval)
    return (p,r)

def get_null_NB_params(mats, combined_matrix, q0):
    assert False, "not used"
    estimator = combined_matrix.est
    nan_indices = np.where(np.isnan(q0))
    q0[np.isnan(q0)] = smallval
    q0[q0 <= 0] = smallval
    f_q0 = estimator.predict(q0)
    mu = get_null_mean(mats, q0)
    mu[nan_indices] = np.nan
    sigma = get_null_variance(mats, q0, f_q0)
    sigma[nan_indices] = np.nan
    
    (p,r) = calc_p_r(mu, sigma)
    np.savetxt("f_q0.txt", f_q0, fmt="%5e", delimiter="\t")


    logging.info("min/max mu: %.10e %.10e" % (np.nanmin(mu), np.nanmax(mu)))
    logging.info("min/max sigma: %.10e %.10e" % (np.nanmin(sigma), np.nanmax(sigma)))
    logging.info("min/max r: %.10e %.10e" % (np.nanmin(r), np.nanmax(r)))
    logging.info("min/max p: %.10e %.10e" % (np.nanmin(p), np.nanmax(p)))
    combined_matrix.mu = mu
    combined_matrix.sigma = sigma
    combined_matrix.r = r
    combined_matrix.p = p
    combined_matrix.has_NB_params = True
    
    return combined_matrix


def get_null_NB_params_old(matrix, q0):
    """
    This function calculations dispersion (r) and probability of success (p)
    under the null distribution from the fitted g() and

    """
    assert False, "not used"
    assert matrix.fitted, "Must fit variance before calculating NB params"
    biases = matrix.biases
    biasmat = np.outer(biases, biases)

    est = matrix.est
    # lengths_with_counts = matrix.fitted_lengths
    mat_shape = biasmat.shape

    logging.info("predicting raw variance")
    f_q0 = est.predict(q0)
    r_recip = np.zeros(q0.shape)
    q0_2 = np.multiply(q0, q0)
    positive_indices = np.where(q0_2 > 0)

    #sigma = q0 + np.multiply(np.multiply(biasmat, biasmat), f_q0)
    # np.savetxt("sigma.txt",sigma,delimiter="\t")

    logging.info("calculating r and p")
    r_recip[positive_indices] = np.divide(f_q0[positive_indices], q0_2[positive_indices])
    r_recip[np.where(q0_2 <= smallval)] = smallval

    r = 1 / np.maximum(r_recip, smallval)
    p = np.divide(r, (r + np.multiply(biasmat, q0)))

    r[np.where(r < smallval)] = smallval
    np.clip(p, smallval, 1 - smallval)

    #np.savetxt("q0_2.txt", q0_2, fmt="%5e", delimiter="\t")
    #np.savetxt("q0.txt", q0, fmt="%5e", delimiter="\t")
    np.savetxt("f_q0.txt", f_q0, fmt="%5e", delimiter="\t")
    #np.savetxt("r_recip.txt", r_recip, fmt="%5f", delimiter="\t")
    #np.savetxt("r.txt", r, fmt="%5f", delimiter="\t")
    #np.savetxt("p.txt", p, fmt="%5f", delimiter="\t")

    logging.info("min/max r: %.10e %.10e" % (np.min(r), np.max(r)))
    logging.info("min/max p: %.10e %.10e" % (np.min(p), np.max(p)))
    matrix.r = r
    matrix.p = p
    matrix.mu = np.multiply(biasmat, q0)
    #matrix.sigma = sigma
    matrix.has_NB_params = True
    matrix.mat_shape = mat_shape

    return matrix




def pval(counts_A, dispersion_A, p_success_A, counts_B, dispersion_B, p_success_B):
    """
    Given two observed counts and the dispersions and probability of success for each NS distribution,
    calculate the p-value for those counts
    """
    assert False, "not used"
    # probability of observed data

    #logging.debug("p_a: %f p_b: %f r_a: %f r_b: %f" % (p_success_A, p_success_B, dispersion_A, dispersion_B))
    #logging.debug("counts A: %d  counts B: %d" % (counts_A, counts_B))

    if np.isnan(p_success_A) or np.isnan(p_success_B):
        return np.nan, np.nan, np.nan

    log_p_counts_A = nbinom.logpmf(counts_A, n=dispersion_A, p=p_success_A)
    log_p_counts_B = nbinom.logpmf(counts_B, n=dispersion_B, p=p_success_B)

    log_p_counts = log_p_counts_A + log_p_counts_B

    # now we will calculate the p-value,
    # conditioning on the total count
    total_count = counts_A + counts_B
    numerator = []
    denominator = []
    for a in range(int(total_count) + 1):
        b = total_count - a
        log_p_a = nbinom.logpmf(a, dispersion_A, p_success_A)
        log_p_b = nbinom.logpmf(b, dispersion_B, p_success_B)
        log_p_joint = log_p_a + log_p_b
        #if verbose:
        #    print("a: %.3f %.3f %.3f b: %.3f %.3f %.3f log_p_a: %f log_p_b %f p_counts: %f p_joint: %f" % (a, dispersion_A, p_success_A, b, dispersion_B, p_success_B, log_p_a, log_p_b, log_p_counts, log_p_joint))
        if log_p_joint <= log_p_counts:
            numerator.append(log_p_joint)
        denominator.append(log_p_joint)
    if len(numerator)==0:
        log_num_sum = 0
    else:
        log_num_sum = logsumexp(numerator)
    if len(denominator)==0:
        log_dem_sum = 0
    else:
        log_dem_sum = logsumexp(denominator)
    if log_num_sum != 0 and log_dem_sum != 0:
        p_val = log_num_sum - log_dem_sum
    else:
        p_val = np.nan
    #if verbose:
    #    print("log_num_sum: %f log_dem_sum: %f log_p_val: %f" % (log_num_sum, log_dem_sum, p_val))
    return p_val, log_p_counts_A, log_p_counts_B


def get_normalized_sum(matA, matB):
    biases_A = matA.biases
    biasmat_A = np.outer(biases_A, biases_A)
    q_A = matA.data / biasmat_A

    biases_B = matB.biases
    biasmat_B = np.outer(biases_B, biases_B)
    q_B = matB.data / biasmat_B

    sum_mat = q_A + q_B
    return sum_mat

def sum_size_factors(mats):
    nReplicates = len(mats)
    nDistances = mats[0].nDistances
    all_size_factors = np.ones((nReplicates,nDistances))
    for i,m in enumerate(mats):
        all_size_factors[i,:] = m.size_factors
    summed_size_factors = all_size_factors.sum(0)
    return summed_size_factors

def process_row(i,fitted_matrix_A,fitted_matrix_B,A_stats,B_stats,q0,percentile_thresh,dist_thresh_lower,dist_thresh_upper,outfh,outfh2,count_var,mats_A,mats_B,smooth_dist):
    nBins = fitted_matrix_A.nBins
    pvals = np.empty(nBins)
    pvals[:] = np.nan
    logging.debug("processing row %d" % i )
    A_counts = A_stats[0]
    tau_A = A_stats[1]
    phi_A = A_stats[2]
    B_counts = B_stats[0]
    tau_B = B_stats[1]
    phi_B = B_stats[2]
    # TODO: skip masked rows / cols / loci
    verbose = False
    # TODO: fix this verbosity stuff w/ logging
    # TODO: skips diag, fix this
    for j in range(i+1,nBins):
        logging.debug("(%d,%d): %f" % (i,j,q0[j]))
        # get distance
        #dist = contact_counts.get_length(i,j,fitted_matrix_A.index_to_chrMid,fitted_matrix_A.binSize)
        dist = abs(i-j)
        #TODO get rid of this
        #assert dist == contact_counts.get_length(i,j,fitted_matrix_A.index_to_chrMid,fitted_matrix_A.binSize)

        # filter by distance if required
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
                
        # TODO: need these to be separate per replicate 
        # get fitted variance function
        A_estimator = fitted_matrix_A.est
        B_estimator = fitted_matrix_B.est

        # calculate NB params
        f_q0_A = A_estimator.predict(q0[j])
        f_q0_B = B_estimator.predict(q0[j])
        
        mean_A = tau_A[j] * q0[j]
        mean_B = tau_B[j] * q0[j]
        var_A = q0[j] * tau_A[j] + f_q0_A * phi_A[j]
        var_B = q0[j] * tau_B[j] + f_q0_A * phi_B[j] 

        p_success_A = mean_A / var_A
        p_success_B = mean_B / var_B
        size_A = ( mean_A * mean_A ) / ( var_A - mean_A )
        size_B = ( mean_B * mean_B ) / ( var_B - mean_B )

        
        # TODO this should acually check  no_dist_norm?
        #if fitted_matrix_A.size_factors is not None:
        #    bias_A = fitted_matrix_A.size_factors[dist] * fitted_matrix_A.biases[i] * fitted_matrix_A.biases[j]
        #    bias_B = fitted_matrix_B.size_factors[dist] * fitted_matrix_B.biases[i] * fitted_matrix_B.biases[j]
        #else:
        #    bias_A = fitted_matrix_A.biases[i] * fitted_matrix_A.biases[j]
        #    bias_B = fitted_matrix_B.biases[i] * fitted_matrix_B.biases[j]
        
        # calculate the size
        
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
            if verbose:
                logging.debug("a: %.3f %.3f %.3f b: %.3f %.3f %.3f log_p_a: %f log_p_b %f p_counts: %f p_joint: %f lessthan? %s" % (
                    a, size_A, p_success_A, b, size_B, p_success_B, log_p_a, log_p_b, log_p_counts, log_p_joint,str(log_p_joint <= log_p_counts)))
            if log_p_joint <= log_p_counts:
                numerator.append(log_p_joint)
            denominator.append(log_p_joint)
        # calculate pvalue
        if len(numerator)>0 and len(denominator)>0:
            pval = logsumexp(numerator) - logsumexp(denominator)
            if verbose:
                logging.debug("log_num_sum: %f log_dem_sum: %f log_p_val: %f" % (logsumexp(numerator), logsumexp(denominator), pval))
            pvals[j] = pval
            outfh.write("%d,%d,%f\n" % (i,j,pval))

# not used
def pval_single(CA,dispersion_A,p_success_A,CB,dispersion_B,p_success_B,verbose):
    assert False, "not used"
    log_p_counts_A = nbinom.logpmf(CA, n=dispersion_A, p=p_success_A)
    log_p_counts_B = nbinom.logpmf(CB, n=dispersion_B, p=p_success_B)
    log_p_counts = log_p_counts_A + log_p_counts_B
    numerator = []
    denominator = []
    total_counts = CA+CB
    for a in range(int(total_counts) + 1):
        b = total_counts - a
        log_p_a = nbinom.logpmf(a, dispersion_A, p_success_A)
        log_p_b = nbinom.logpmf(b, dispersion_B, p_success_B)
        log_p_joint = log_p_a + log_p_b
        if verbose:
            print("a: %.3f %.3f %.3f b: %.3f %.3f %.3f log_p_a: %f log_p_b %f p_counts: %f p_joint: %f" % (
                a, dispersion_A, p_success_A, b, dispersion_B, p_success_B, log_p_a, log_p_b, log_p_counts, log_p_joint))
        if log_p_joint <= log_p_counts:
            numerator.append(log_p_joint)
        denominator.append(log_p_joint)
    if verbose:
        print(numerator)
        print(denominator)
    pval = np.nan
    # calculate pvalue
    if len(numerator)>0 and len(denominator)>0:
        pval = logsumexp(numerator) - logsumexp(denominator)
        if verbose:
            logging.debug("log_num_sum: %f log_dem_sum: %f log_p_val: %f" % (logsumexp(numerator), logsumexp(denominator), p_val))
    return pval

def pvals_filtered(fitted_matrix_A, fitted_matrix_B, matA_file, matB_file, q0_file, percentile_thresh, dist_thresh_lower, dist_thresh_upper, outfh, count_var, mats_A, mats_B, smooth_dist):
    assert fitted_matrix_A.fitted and fitted_matrix_B.fitted, "need to fit matrices before calculating pvalues"
    assert fitted_matrix_A.nBins == fitted_matrix_B.nBins

    outfh2 = open("stats.csv",'w')
    #outfh2.write("i,j,q0[j],f_q0_A,f_q0_B,bias_A,bias_B,size_A,size_B,p_success_A,p_success_B\n")
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
    for aline,bline,qline in izip_longest(matA_reader,matB_reader,q0_reader):
        if aline is not None:
            i = int(aline[0])
            if i > current_row:
                if current_row in done_A and current_row in done_B and current_row in done_q and done_A[current_row] and done_B[current_row] and done_q[current_row]:
                    if current_row in A_counts and current_row in B_counts and current_row in q0:
                        logging.debug(A_counts[current_row])
                        logging.debug(B_counts[current_row])
                        logging.debug(q0[current_row])
                        process_row(current_row,fitted_matrix_A,fitted_matrix_B,A_counts[current_row],B_counts[current_row],q0[current_row],percentile_thresh,dist_thresh_lower,dist_thresh_upper,outfh,outfh2,count_var,mats_A,mats_B,smooth_dist)
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
    for i in sorted(A_counts.keys()):
        if i not in B_counts:
            B_counts[i] = (np.zeros(nBins),np.zeros(nBins),np.zeros(nBins))
        if i not in q0:
            q0[i] = np.zeros(nBins)
        process_row(i,fitted_matrix_A,fitted_matrix_B,A_counts[i],B_counts[i],q0[i],percentile_thresh,dist_thresh_lower,dist_thresh_upper,outfh,outfh2,count_var,mats_A,mats_B,smooth_dist)
    outfh.close()
    outfh2.close()

    
    
    


def pval_mat(matA, matB, q0, allBins_reversed, count_cutoff, lower_thresh, upper_thresh):
    assert False, "not used"
    assert matA.has_NB_params and matB.has_NB_params, "Must calculate NB params before calculating p-values"

    # calculate p-value matrix
    log_p_vals = np.empty(matA.data.shape)
    log_p_vals[:] = np.nan
    log_p_A = np.empty(matA.data.shape)
    log_p_A[:] = np.nan
    log_p_B = np.empty(matA.data.shape)
    log_p_B[:] = np.nan
    nan_reasons = np.zeros(matA.data.shape)
    min_counts_mat = np.zeros(matA.data.shape)
    directions = np.zeros(matA.data.shape)

    #biases_A = matA.biases
    #biasmat_A = np.outer(biases_A, biases_A)
    #biases_B = matB.biases
    #biasmat_B = np.outer(biases_B, biases_B)
    
    for i in range(matA.data.shape[0]):
        for j in range(matA.data.shape[1]):
            if i >= j:
                continue  # force upper triangular & diagonal
            if allBins_reversed[i][0] != allBins_reversed[j][0]:
                continue  # skip interchromosomal
            # skip values not in range
            if abs(i - j) < lower_thresh or abs(i - j) > upper_thresh:
                continue
            if i % 10 == 0 and j % 10 == 0:
                logging.info('i,j:' + str((i, j)))
            counts_A = matA.data[i, j]
            counts_B = matB.data[i, j]
            if counts_A+counts_B < count_cutoff:
                nan_reasons[i, j] = 99
                continue
            if counts_A == counts_B:
                log_p_vals[i, j] = 0
            else:
                # NB params
                dispersion_A = matA.r[i, j]
                dispersion_B = matB.r[i, j]
                p_success_A = matA.p[i, j]
                p_success_B = matB.p[i, j]
                # min_counts = get_min_counts(p_success_A,p_success_B,dispersion_A,dispersion_B,target_min_pval)
                # min_counts = 0
                # min_counts_mat[i,j] = min_counts
                if counts_A is np.ma.masked or counts_B is np.ma.masked:
                    # p_vals[i,j] = np.nan
                    log_p_vals[i, j] = np.nan
                    nan_reasons[i, j] = 1  # counts_masked
                elif dispersion_A is np.ma.masked or dispersion_B is np.ma.masked:
                    # p_vals[i,j] = np.nan
                    log_p_vals[i, j] = np.nan
                    nan_reasons[i, j] = 2  # dispersion_masked
                elif p_success_A is np.ma.masked or p_success_B is np.ma.masked:
                    # p_vals[i,j] = np.nan
                    log_p_vals[i, j] = np.nan
                    nan_reasons[i, j] = 3  # p_success_masked
                else:
                    p, p_A, p_B = pval(counts_A, dispersion_A, p_success_A, counts_B, dispersion_B, p_success_B)
                    # p_vals[i,j] = p
                    log_p_vals[i, j] = p
                    log_p_A[i, j] = p_A
                    log_p_B[i, j] = p_B
                    if p == np.nan:
                        nan_reasons[i, j] = 4
                #logging.debug("%f %f %f" % (log_p_A[i, j], log_p_B[i, j], log_p_vals[i, j]))
    return log_p_vals, nan_reasons, directions


def load_lengths(precalculated_length_file):
    logging.info("Loading lengths")
    lengths_rows = {}
    lengths_cols = {}
    with open(precalculated_length_file) as fh:
        reader = csv.reader(fh, delimiter='\t')
        for line in reader:
            l, i, j = [int(x) for x in line]
            if l in lengths_rows:
                lengths_rows[l].append(i)
                lengths_cols[l].append(j)
            else:
                lengths_rows[l] = [i]
                lengths_cols[l] = [j]
    return lengths_rows, lengths_cols


def get_z(mats,q,smooth_dist,i,j):
    sum_recip_bias = 0.
    for m in mats: 
        if m.size_factors is None:
            sum_recip_bias += 1./ m.biases[i]*m.biases[j]
        else:
            sum_recip_bias += 1./ m.biases[i]*m.biases[j]*m.size_factors[abs(i-j)]
    #return sum_recip_bias * q / (len(mats)*smooth_dist)
    return sum_recip_bias * q / (len(mats))

def fit_mat(matrix, length_frac = 2./3):
    w, vars, means, z = matrix.w, matrix.vars, matrix.mean, matrix.z
    length = len(matrix.w) * 2 / 3
    logging.info(matrix.w)
    logging.info("only fitting mean/variance up to genomic distance %.3f" % length)
    w, vars, means, z = matrix.w[:length], matrix.vars[:length], matrix.mean[:length], matrix.z[:length]


    X = means
    y = vars - z

    #if len(X.shape) < 2:
    #    X = X[:, np.newaxis]
    #if len(y.shape) < 2:
    #    y = y[:, np.newaxis]

    np.savetxt("X_before.txt",X)
    np.savetxt("y_before.txt",y)
   
    logging.debug("X and y to fit:")
    logging.debug(X)
    logging.debug(y)
    # filter nan
    nans = np.logical_or(np.isnan(X),np.isnan(y))
    #logging.debug(X.shape)
    #logging.debug(nans.shape)
    X = X[np.where(~nans)]
    y = y[np.where(~nans)]
    # filter negatives
    positives = np.logical_and(X>0,y>0)
    X = X[np.where(positives)]
    y = y[np.where(positives)]
    #logging.debug(X)
    #logging.debug(y)

    np.savetxt("X_after.txt",X)
    np.savetxt("y_after.txt",y)
    
    
    est = LogPolyEstimator(degree=2)
    #est = LowessEstimator()
    #est = LocfitEstimator()
    est.fit(X, y)

    #w_all, v_all, means_all, z_all = matrix.w, matrix.vars, matrix.mean, matrix.z

    #X_all = means_all
    #y_all = w_all

    # set negatives to ~0
    #X_all = X_all.clip(smallval)
    #y_all = y_all.clip(smallval)
    
    ypred = est.predict(X.reshape(-1,1))
    
    #matrix.w_l = w_all
    #matrix.v_l = v_all
    matrix.est = est
    matrix.mean_fitted = X.reshape(-1,1)
    matrix.g_fitted = ypred
    #matrix.q_l = means_all
    #matrix.z_l = z_all
    #matrix.fitted_lengths = None
    matrix.fitted = True
    return matrix


def precalculate_lengths(allBins_reversed, outfile, outfile_reversed, mappability):
    lengths, lengths_reversed = contact_counts.get_lengths(allBins_reversed)
    logging.info("saving length matrix")
    np.savetxt(outfile, lengths, delimiter='\t', newline='\n')
    logging.info("saving reversed length matrix")
    fh = open(outfile_reversed, 'w')
    for l in lengths_reversed:
        pairs = lengths_reversed[l]
        for (i, j) in pairs:
            fh.write("%d\t%d\t%d\n" % (l, i, j))
    fh.close()

def parse_chrom_file(chrom_file):
    chrom_size = []
    chr_size_file = open(chrom_file, 'r')
    for line in chr_size_file:
        line = line[:-1]
        pair = line.split('\t')
        chrom_size.append(int(pair[1]))
    chr_size_file.close()
    return chrom_size


def get_chrom_data(chrom_array, bin_size):
    max_length = 0
    counts = 0
    for l in chrom_array:
        l = int(l) / bin_size + 1
        max_length = max(max_length, l)
        f = math.factorial
        cur_possible_count = f(l) / f(l - 2) / 2
        counts += cur_possible_count
    return max_length, counts


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
    return Afiles,Bfiles
        


def set_up_argparser():
    parser = argparse.ArgumentParser()
    # required params
    parser.add_argument("binsize", type=int, help="the bin size of the input file")
    parser.add_argument("binfile", help="list of immapable index")
    parser.add_argument("idfile", help="input file of count and bias filenames")
    parser.add_argument("id_A", help="ID for class A")
    parser.add_argument("id_B", help="ID for class B")
    # optional params
    parser.add_argument("-d", "--distances", default=None, help="pregenerated matrix with distance")
    parser.add_argument("-dr", "--distance_reverse", default=None, help="reverse matrix with distance")
    parser.add_argument("-o", "--output_prefix", default="differential_output", help="the prefix for output")
    # TODO implement min and max distance at which to run
    parser.add_argument("-mind", "--min_dist", type=int, default=0, help="the lower threshold for length")
    parser.add_argument("-maxd", "--max_dist", type=int, default=10000, help="the upper threshold for length")
    parser.add_argument("--filter_diagonal", type=bool, default=True, help="filter counts on the diagonal?" )
    parser.add_argument("-m", "--map_thresh", type=float, default=0.25, help="the mappability threshold")
    parser.add_argument("-p", "--min_percentile", type=int, default=100, help="the percentage threshold")
    parser.add_argument("-ds", "--dist_smooth", type=int, default=10, help="the number of locus pairs to smooth over for calculating mean/variance")
    parser.add_argument("-c", "--chromsize", help="chromosome size file")
    parser.add_argument("-om", "--output_matrix", help="Whether to output the full, dense matrix of P-values (rather than just those passing a threshold) (default: F)")
    parser.add_argument("--output_p_thresh")
    parser.add_argument("--output_q_thresh")
    parser.add_argument("--no_dist_norm", action='store_true', help="Disable distance size factors")
    return parser
    
def debug_write(mat,outfile):
    ofh = open(outfile,'w')
    ofh.write("index\tmean\tvar\tz\tcorrected_var\tfitted_mean\tfitted_var\n")
    d = 0
    for m,v,z,w,mm,g in izip_longest(mat.mean,mat.vars,mat.z,mat.w,mat.mean_fitted,mat.g_fitted):
        if g is not None:
            ofh.write("%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n" % (d,m,v,z,w,mm,g))
        else:
            ofh.write("%d\t%.3f\t%.3f\t%.3f\t%.3f\tNA\n" % (d,m,v,z,w))
        d += 1



def main():
    logging.basicConfig(filename='output.log',format='%(asctime)s %(levelname)s from %(filename)s %(funcName)s :: %(message)s', level=logging.INFO)
    #warnings.simplefilter("error")
    
    parser = set_up_argparser()
    args = parser.parse_args()
    

    #TODO: wtf?
    #chrom_array = [249250621, 243199373, 198022430, 191154276, 180915260, 171115067, 159138663, 155270560, 146364022, 141213431,
    #               135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 63025520,
    #               59373566, 59128983, 51304566, 48129895]

    binfile = args.binfile
    binsize = args.binsize
    # pre generated matrix with distance
    precalculated_length_file = args.distances
    precalculated_length_file_reversed = args.distance_reverse
    # input data in tab delimited
    idfile = args.idfile
    id_A = args.id_A
    id_B = args.id_B
    
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
    # in documentation, should make a note that if distance normalization is turned off than this should be set to some large number
    
    # output
    outprefix = args.output_prefix
    
    #logging.info("setting tmpdir to current WD")
    tempfile.tempdir = os.getcwd()
    #tempfile.tempdir = '/scratch'
    #tempfile.tempdir = os.environ['TMPDIR']
    logging.info("temp directory currently set to: %s" % tempfile.tempdir)
     
    # set up bins
    logging.info("Setting up bins")
    (allBins, allBins_reversed, badBins) = contact_counts.generate_bins(binfile, mappability_thresh)
    
    
    # read input file setting up class A and class B
    Afiles,Bfiles = read_id_file(idfile,id_A,id_B)
   
    # read input contact count matrices
    # when reading the input, calculate for each genomic distance:
    #   - the sum of the normalized contact counts
    #   - the sum of the square of the normalized contact counts
    #   - the number of observations at that genomic distance
    logging.info("Loading data")
    mats_A = []
    mats_B = []
    for (matfile_A,biasfile_A) in Afiles:
        matrixA = contact_counts.contactCountMatrix(allBins, allBins_reversed, badBins)
        #matrixA.load_from_file(matfile_A, biasfile_A, binsize, filter_diagonal)
        matrixA.infile = matfile_A
        matrixA.load_biases(biasfile_A)
        matrixA.nDistances = matrixA.nBins #TODO: this should change
        matrixA.binSize = binsize
        mats_A.append(matrixA)
    for (matfile_B,biasfile_B) in Bfiles:
        matrixB = contact_counts.contactCountMatrix(allBins, allBins_reversed, badBins)
        #matrixB.load_from_file(matfile_B, biasfile_B, binsize, filter_diagonal)
        matrixB.infile = matfile_B
        matrixB.load_biases(biasfile_B)
        matrixB.nDistances = matrixB.nBins #TODO: this should change
        matrixB.binSize = binsize
        mats_B.append(matrixB)  

    mats = mats_A + mats_B
   
    nBins = mats[0].nBins
    #TODO: bag distances by number of bins at that distance, for the further out ones
    nDistances = nBins
    
    #TODO: remove references to this 
    # load count vars
    #count_var = contact_counts.load_variances("/net/noble/vol2/home/katecook/proj/2016ACCOST/results/katecook/20180425_9_replicates_size_factors/count_var_normalized.tab",nBins)
    count_var = None    

    logging.info("Sorting data by genomic distance")
    
    
    for m in mats:
        tmp_length_indexed = tempfile.NamedTemporaryFile(delete=False,prefix="tmp_length_indexed_")
        total_counts = contact_counts.length_index_contact_counts(m.infile,tmp_length_indexed,binsize,index_to_chrMid=allBins_reversed,chrMid_to_index=allBins)
        logging.debug("total counts is %d" % total_counts)

        tmp_sorted_by_length = tempfile.NamedTemporaryFile(delete=False,prefix="tmp_sorted_by_length_")
        contact_counts.sort_contactfile_by_length(tmp_length_indexed.name,tmp_sorted_by_length)

        tmp_sorted_by_index = tempfile.NamedTemporaryFile(delete=False,prefix="tmp_sorted_by_index_")
        contact_counts.sort_contactfile_by_index(tmp_length_indexed.name,tmp_sorted_by_index)

        m.infile_sorted_by_length = tmp_sorted_by_length.name
        m.infile_sorted_by_index = tmp_sorted_by_index.name
        m.infile_length_indexed = tmp_length_indexed.name
        #m.infile = m.sortedfile
        #os.remove(tmp_length_indexed.name)
        
        # calculate normalized values & make sure the biases are scaled appropriately
        #tmp_normalized = tempfile.NamedTemporaryFile(delete=False,prefix="tmp_normalized_")
        #total_normalized = contact_counts.normalize_contact_counts(tmp_length_indexed.name,tmp_normalized,m.biases)
        #logging.debug("normalized total is %f" % total_normalized)
        #if abs(total_normalized - total_counts) > total_counts_eps:
        #    logging.info("Difference between total raw and normalized was greater than %f, scaling biases" % total_counts_eps)
        #    m.biases *= np.sqrt(total_normalized/total_counts)
        #    os.remove(tmp_normalized.name)
        #    tmp_normalized = tempfile.NamedTemporaryFile(delete=False,prefix="tmp_normalized_")
        #    total_normalized = contact_counts.normalize_contact_counts(tmp_length_indexed.name,tmp_normalized,m.biases)
        #    logging.debug("normalized total now %f" % total_normalized)
        #    assert abs(total_normalized - total_counts) <= total_counts_eps
        #m.total_counts = total_counts
        #m.total_normalized = total_normalized
        #m.infile_normalized = tmp_normalized.name
    #TODO: this should probably be moved somewhere      
    for m in mats:
        assert m.nBins == nBins
    
     
    
    #logging.getLogger().setLevel(logging.DEBUG)
    
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
            np.savetxt("size_factors_A%d.txt"%i,mats_A[i].size_factors)
        for i in range(len(mats_B)):
            np.savetxt("size_factors_B%d.txt"%i,mats_B[i].size_factors)
        #logging.info("Summing size factors")
        #summed_size_factors_A = sum_size_factors(mats_A)
        #summed_size_factors_B = sum_size_factors(mats_B)
     
    #sys.exit()

    logging.info("Calculating means and variances") 
    matrixA = contact_counts.combine_replicates_and_calculate_mean_variance(mats_A,smooth_dist)
    matrixB = contact_counts.combine_replicates_and_calculate_mean_variance(mats_B,smooth_dist)
    #if not no_dist_norm:
    #    matrixA.summed_size_factors = summed_size_factors_A
    #    matrixB.summed_size_factors = summed_size_factors_B
    
    
    
    # get total counts for A condition and B condition 
    sumA_fh = tempfile.NamedTemporaryFile(delete=False,prefix="tmp_sumA_")
    contact_counts.sum_mats(mats_A,sumA_fh)
    
    sumB_fh = tempfile.NamedTemporaryFile(delete=False,prefix="tmp_sumB_")
    contact_counts.sum_mats(mats_B,sumB_fh)
    
    # get bias factors for ease of calculation later
    #tauA_fh = tempfile.NamedTemporaryFile(delete=False,prefix="tmp_tau_phi_A_")
    #contact_counts.calculate_tau_phi(mats_A,tauA_fh)
    #tauB_fh = tempfile.NamedTemporaryFile(delete=False,prefix="tmp_tau_phi_B_")
    #contact_counts.calculate_tau_phi(mats_B,tauB_fh)
     
    logging.info("Done calculating means and variances.")

    logging.info("Fitting means & variances for A")
    matrixA = fit_mat(matrixA, fit_frac)

    logging.info("Fitting means and variances for B")
    matrixB = fit_mat(matrixB, fit_frac)

    debug_write(matrixA,"fitted_matrix_A.txt")
    debug_write(matrixB,"fitted_matrix_B.txt")

     
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
    # don't calculate mean/variance/p/r unless we need to
    outfile = outprefix + "_ln_pvals.txt"
    outfh = open(outfile,'w')
    #logging.getLogger().setLevel(logging.DEBUG)
    pvals_filtered(matrixA, matrixB, sumA_fh.name, sumB_fh.name, q0_fh.name, max_perc, dist_thresh_lower, dist_thresh_upper, outfh, count_var, mats_A, mats_B, smooth_dist)

    #TODO: save log pvalues, not ln
    #np.savetxt(outprefix + "_ln_pvals.txt", ln_pvals, fmt='%.3f', delimiter='\t')
    #np.savetxt(outprefix + "_nan_reasons.txt", nan_reasons, fmt='%d', delimiter='\t')
    
    # delete temporary files storing distance-sorted counts
    #TODO: do this

if __name__ == "__main__":
    main()
