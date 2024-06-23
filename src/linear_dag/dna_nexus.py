import os

import allel
import dxpy
import numpy as np
import pyspark

from scipy.io import mmwrite
from scipy.sparse import csr_matrix

from linear_dag import LinearARG


def download_from_dx(dx_data_object, local_file_name):
    file_id = dx_data_object["id"]
    project_id = dx_data_object["describe"]["project"]
    dxpy.download_dxfile(file_id, local_file_name, project=project_id)


# Function to process a single VCF file
def process_vcf(vcf_dx_data_object: dict, tabix_dx_data_object: dict = None, region: str = None) -> csr_matrix:
    vcf_file_name = vcf_dx_data_object["describe"]["name"]
    project_id = vcf_dx_data_object["describe"]["project"]
    download_from_dx(vcf_dx_data_object, vcf_file_name)

    if tabix_dx_data_object is not None:
        tabix_file_name = tabix_dx_data_object["describe"]["name"]
        download_from_dx(tabix_dx_data_object, tabix_file_name)

    print("Reading vcf...")
    callset = allel.read_vcf(
        vcf_file_name,
        fields=["calldata/GT", "variants/ALT", "variants/REF", "variants/CHROM", "variants/POS", "variants/AF"],
        chunk_length=1000,
        region=region,
    )

    os.remove(vcf_file_name)
    if callset is None:
        return None

    print("Loading genotypes...")
    genotypes = allel.GenotypeArray(callset["calldata/GT"])

    alleles = callset["variants/ALT"]

    # Data manipulation and allele flipping
    dense_matrix = genotypes.to_n_alt().T
    af = np.mean(dense_matrix, axis=0) / 2
    flip = af > 0.5
    dense_matrix[:, flip] = 2 - dense_matrix[:, flip]
    sparse_matrix = csr_matrix(dense_matrix)

    # Metadata preparation
    chromosomes = callset["variants/CHROM"]
    positions = callset["variants/POS"]
    refs = callset["variants/REF"]
    alts = np.array([",".join(x) for x in alleles])

    metadata_dtype = [("chromosome", "U10"), ("position", int), ("ref", "U10"), ("alt", "U10")]
    metadata = np.empty(len(chromosomes), dtype=metadata_dtype)
    metadata["chromosome"] = chromosomes
    metadata["position"] = positions
    metadata["ref"] = refs
    metadata["alt"] = alts

    # Flip ref/alt where needed
    metadata["ref"][flip] = alts[flip]
    metadata["alt"][flip] = refs[flip]

    # Save files locally
    save_file_name = vcf_file_name.split(".vcf")[0]
    mtx_filename = f"{save_file_name}.genos.mtx"
    metadata_filename = f"{save_file_name}.metadata.txt"
    mmwrite(mtx_filename, csr_matrix(dense_matrix))
    np.savetxt(
        metadata_filename, metadata, fmt="%s,%d,%s,%s", delimiter=",", header="chromosome,position,ref,alt", comments=""
    )

    # Upload files to DNAnexus
    dxpy.upload_local_file(mtx_filename, project=project_id, folder="genotypes")
    dxpy.upload_local_file(metadata_filename, project=project_id, folder="genotype_metadata")

    # Optional: Clean up local files if not needed locally
    os.remove(mtx_filename)
    os.remove(metadata_filename)

    return sparse_matrix


def process_directory(directory_path):
    sc = pyspark.SparkContext()
    spark = pyspark.sql.SparkSession(sc)

    project_id = "project-GfvX1pjJg9g7QV7qfYG0fjFv"
    vcf_files = dxpy.find_data_objects(
        classname="file",
        project=project_id,
        folder=directory_path,
        recurse=True,
        name="*.vcf.gz",
        name_mode="glob",
        describe=True,
    )

    vcf_rdd = spark.sparkContext.parallelize(list(vcf_files))

    # Run process_vcf on each file
    vcf_rdd.map(process_vcf)


def download_mtx(
    genotypes_directory: str, metadata_directory: str, which_files: tuple[int, int] = None
) -> tuple[list, list]:
    from itertools import islice

    project_id = "project-GfvX1pjJg9g7QV7qfYG0fjFv"

    # Find .mtx files to download
    dx_data_objects_mtx = dxpy.find_data_objects(
        classname="file",
        project=project_id,
        folder=genotypes_directory,
        recurse=False,
        name="*.mtx",
        name_mode="glob",
        describe=True,
    )

    dx_data_objects_mtx = islice(dx_data_objects_mtx, *which_files)

    mtx_files = []
    metadata_files = []
    for dx_data_object in dx_data_objects_mtx:
        # Download the .mtx file
        file_id = dx_data_object["id"]
        local_file_name = dx_data_object["describe"]["name"]
        dxpy.download_dxfile(file_id, local_file_name, project=project_id)
        mtx_files.append(local_file_name)

        # Find and download corresponding metadata file
        metadata_file_name = local_file_name.split(".vcf.gz")[0] + ".vcf.gz.metadata.txt"
        metadata_file_data = dxpy.find_data_objects(
            classname="file",
            project=project_id,
            folder=metadata_directory,
            recurse=False,
            name=metadata_file_name,
            describe=False,
        )
        try:
            metadata_file_data = next(metadata_file_data)
        except StopIteration:
            raise FileNotFoundError(f"Did not find metadata for file {local_file_name}")

        dxpy.download_dxfile(metadata_file_data["id"], filename=metadata_file_name)
        metadata_files.append(metadata_file_name)

    return mtx_files, metadata_files


def load_mtx(mtx_files: list[str], metadata_files: list[str]) -> tuple:
    from scipy.io import mmread
    from scipy.sparse import csc_matrix, hstack

    matrices = [mmread(mtx_file).astype(np.intc) for mtx_file in mtx_files]
    genotype_matrix = csc_matrix(hstack(matrices))

    metadata = load_metadata_to_dict(metadata_files)

    return genotype_matrix, metadata


def load_metadata_to_dict(files) -> dict:
    # Initialize a dictionary to hold our metadata
    metadata = {"chromosome": [], "position": [], "ref": [], "alt": []}

    for filename in files:
        # Open the file and read through it line by line
        with open(filename, "r") as file:
            # Skip the header line
            next(file)
            # Read each line in the file
            for line in file:
                # Strip whitespace and split by comma
                chromosome, position, ref, alt = line.strip().split(",")[:4]
                # Append data to each list in the dictionary
                metadata["chromosome"].append(chromosome)
                metadata["position"].append(int(position))  # Convert position to integer
                metadata["ref"].append(ref)
                metadata["alt"].append(alt)

    return metadata


def run_dnanexus_infer_linarg_workflow(which_files, output_filename) -> None:
    mtx_files = download_mtx("genotypes/", "genotype_metadata/", which_files)
    genotypes, metadata = load_mtx(mtx_files)

    # discard singleton and monomorphic alleles
    include_variant = np.diff(genotypes.indptr) > 1
    genotypes = genotypes[:, include_variant]
    metadata = {key: val[include_variant] for key, val in metadata.items()}

    linarg = LinearARG.from_genotypes(genotypes)
    linarg = linarg.find_recombinations()
    linarg = linarg.make_triangular()
    linarg.write(output_filename)
