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
    r = np.divide(np.multiply(mu,mu) , (sigma - mu))
    p =  mu / sigma
    # make sure they're positive (well, > some small number)
    r[r > (1/smallval)] = 1/smallval
    #p = np.clip(p, smallval, 1 - smallval)
    return (p,r)

def get_null_NB_params(mats, combined_matrix, q0):
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

