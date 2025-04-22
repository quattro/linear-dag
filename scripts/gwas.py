import scipy.sparse as sp
import argparse
import numpy as np
import os
import time
import polars as pl
import linear_dag as ld
from linear_dag.core.lineararg import LinearARG
import pandas as pd


def load_linarg(linarg_dir, partition_id):
    start = time.time()
    linarg = ld.LinearARG.read(f'{linarg_dir}/{partition_id}/linear_arg.npz', f'{linarg_dir}/{partition_id}/linear_arg.pvar', f'{linarg_dir}/{partition_id}/linear_arg.psam')    
    end = time.time()
    return linarg, end-start


def load_genotypes(linarg_dir, partition_id):
    start = time.time()
    mtx_files = os.listdir(f'{linarg_dir}/{partition_id}/genotype_matrices/')
    ind_arr = np.array([int(f.split('_')[0]) for f in mtx_files])
    order = ind_arr.argsort()
    mtx_files = np.array(mtx_files)[order].tolist() # sort files by index
    genotypes = sp.hstack([sp.load_npz(f'{linarg_dir}/{partition_id}/genotype_matrices/{m}') for m in mtx_files])   
    end = time.time() 
    return genotypes, end-start


def get_phenotype_covariates():
    
    phenotype = 'p50_i0'
    covariates = ['p21022', 'sex'] + [f'p22009_a{i}' for i in range(1,41)]
    
    sample_metadata = pd.read_csv('/mnt/project/sample_metadata/ukb20279/250122_sampleIndex_sampleID_withdrawnRemoved.csv')
    phenotypes = pd.read_csv('/mnt/project/phenotypes/age_sex_height_pcs.csv')
    phenotypes['sex'] = [0 if phenotypes.p31[i]=='Male' else 1 for i in range(phenotypes.shape[0])]
    phenotypes = phenotypes[['eid', phenotype]+covariates]
    phenotypes.index = phenotypes.eid
    phenotypes = phenotypes.loc[list(sample_metadata.sample_id)] # filter and order phenotypes by sample_metadata
    
    N = len(phenotypes)
    rows_to_drop = np.where(phenotypes.isnull().any(axis=1))[0] # remove any samples with missing phenotypes or covariates
    phenotypes = phenotypes.drop(phenotypes.index[rows_to_drop])
    C = sp.csr_matrix(phenotypes[covariates].to_numpy())
    y = np.array(phenotypes[phenotype])
    y_norm = (y - np.mean(y)) / np.std(y)

    data = np.ones(N-len(rows_to_drop))
    row_indices = np.arange(N-len(rows_to_drop))
    col_indices = np.setdiff1d(np.arange(N), rows_to_drop)
    R = sp.csr_matrix((data, (row_indices, col_indices)), shape=(N-len(rows_to_drop), N))
        
    P = sp.linalg.aslinearoperator(sp.eye(R.shape[0])) - sp.linalg.aslinearoperator(C) @ sp.linalg.aslinearoperator(sp.linalg.spsolve(C.T @ C, C.T))
    y_resid = P @ y_norm.T
    y_resid = (y_resid - np.mean(y_resid)) / np.std(y_resid)
    
    return y_resid, R


def get_sum_matrix(N):
    data = np.ones(2*N)
    row_indices = np.repeat(np.arange(N), 2)
    col_indices = np.arange(2*N)
    S = sp.csr_matrix((data, (row_indices, col_indices)), shape=(N, 2*N))
    return S


def linarg_regression(linarg, y_resid, R):
    
    N_individuals = int(linarg.shape[0] / 2)
    S = get_sum_matrix(N_individuals)
    N = R.shape[0]

    start = time.time()
    X = sp.linalg.aslinearoperator(R) @ sp.linalg.aslinearoperator(S) @ linarg.normalized
    beta_hat = (X.T @ y_resid) / (2**0.5 * N)
    end = time.time()
        
    return beta_hat, end-start


def genotypes_regression(genotypes, y_resid, R):
    
    N_individuals = int(genotypes.shape[0] / 2)
    S = get_sum_matrix(N_individuals)
    N = R.shape[0]
    
    start = time.time()
    allele_frequencies = np.ones(genotypes.shape[0]) @ genotypes / genotypes.shape[0]
    mean = sp.linalg.aslinearoperator(np.ones((genotypes.shape[0], 1))) @ sp.linalg.aslinearoperator(allele_frequencies)
    pq = allele_frequencies * (1 - allele_frequencies)
    pq[pq == 0] = 1
    G = (sp.linalg.aslinearoperator(genotypes) - mean) * sp.linalg.aslinearoperator(sp.diags(pq**-0.5))
    X = sp.linalg.aslinearoperator(R) @ sp.linalg.aslinearoperator(S) @ G    
    beta_hat = (X.T @ y_resid) / (2**0.5 * N)
    end = time.time()
    
    return beta_hat, end-start


def benchmark_regression(linarg_dir, partition_id, data_type, res_dir):
    
    if not os.path.exists(f'{res_dir}/'): os.makedirs(f'{res_dir}/')
    if not os.path.exists(f'{res_dir}/statistics/'): os.makedirs(f'{res_dir}/statistics/')
    if not os.path.exists(f'{res_dir}/beta_hats/'): os.makedirs(f'{res_dir}/beta_hats/')
    
    y_resid, R = get_phenotype_covariates()
    
    if data_type == 'linarg':
        linarg, load_time = load_linarg(linarg_dir, partition_id)
        beta_hats, reg_time = linarg_regression(linarg, y_resid, R)
    else:
        genotypes, load_time = load_genotypes(linarg_dir, partition_id)
        beta_hats, reg_time = genotypes_regression(genotypes, y_resid, R)
        
    np.save(f'{res_dir}/beta_hats/{partition_id}_{data_type}.npy', beta_hats)
    with open(f'{res_dir}/statistics/{partition_id}_{data_type}.txt', 'w') as file:
        file.write(" ".join(['partition_id', 'data_type', 'load_time', 'reg_time'])+'\n')
        file.write(" ".join([partition_id, data_type, str(np.round(load_time, 3)), str(np.round(reg_time, 3))])+'\n')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('linarg_dir', type=str)
    parser.add_argument('partition_id', type=str)
    parser.add_argument('data_type', type=str)
    parser.add_argument('res_dir', type=str)
    args = parser.parse_args()
    
    benchmark_regression(args.linarg_dir, args.partition_id, args.data_type, args.res_dir)
    
    
    