import linear_dag as ld


def main():
    linarg = ld.Linarg(genotype_matrix_mtx="/Users/loconnor/Dropbox/linearArg/linearg_shared/genotypes.mtx")
    linarg.apply_maf_threshold(0.001)
    linarg.flip_alleles()
    linarg.form_initial_linarg()
    linarg.create_triolist()
    linarg.find_recombinations()

    assert linarg.A.shape == (40270, 40270)
    assert linarg.A.nnz == 310345

    linarg.print()
    print("OK")


if __name__ == "__main__":
    main()
