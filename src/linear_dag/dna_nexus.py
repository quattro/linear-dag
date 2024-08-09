import dxpy
import numpy as np
import pyspark
from cyvcf2 import VCF

from scipy.sparse import csr_matrix, csc_matrix

from .genotype import read_vcf
from .lineararg import LinearARG

PROJECT_ID = 'project-GfvX1pjJg9g7QV7qfYG0fjFv'


def find_shapeit200k_vcf(chromosome_number: int) -> tuple:
    directory_path = '/Bulk/Previous WGS releases/GATK and GraphTyper WGS/SHAPEIT Phased VCFs'
    file_pattern = f'ukb20279_c{chromosome_number}_b0_v1.vcf.gz'
    vcf_files = dxpy.find_data_objects(classname='file',
                                       project=PROJECT_ID,
                                       folder=directory_path,
                                       name=file_pattern, name_mode='glob',
                                       describe=True)

    file_pattern = f'ukb20279_c{chromosome_number}_b0_v1.vcf.gz.tbi'
    tabix_files = dxpy.find_data_objects(classname='file',
                                         project=PROJECT_ID,
                                         folder=directory_path,
                                         name=file_pattern, name_mode='glob',
                                         describe=True)
    vcf_object = next(vcf_files)
    tabix_object = next(tabix_files)
    return vcf_object, tabix_object

def download_from_dx(dx_data_object, local_file_name):
    file_id = dx_data_object["id"]
    project_id = dx_data_object["describe"]["project"]
    dxpy.download_dxfile(file_id, local_file_name, project=project_id)


def download_vcf(vcf_dx_data_object: dict,
                tabix_dx_data_object: dict = None) -> str:

    vcf_file_name = vcf_dx_data_object["describe"]["name"]
    project_id = vcf_dx_data_object["describe"]["project"]
    download_from_dx(vcf_dx_data_object, vcf_file_name)

    if tabix_dx_data_object is not None:
        tabix_file_name = tabix_dx_data_object["describe"]["name"]
        download_from_dx(tabix_dx_data_object, tabix_file_name)

    # sparse_matrix, variant_info = vcf_to_csc(vcf_file_name, region, phased=phased)
    return vcf_file_name

def vcf_to_csc(path: str, region: str, phased: bool = False, flip_minor_alleles: bool = True) -> tuple[csc_matrix, np.ndarray, list[dict]]:
    """
    Codes unphased genotypes as 0/1/2/3, where 3 means that at least one of the two alleles is unknown.
    Codes phased genotypes as 0/1, and there are 2n rows, where rows 2*k and 2*k+1 correspond to individual k.
    """
    vcf = VCF(path, gts012=True, strict_gt=True)
    data = []
    idxs = []
    ptrs = [0]
    info = []
    flip = []

    ploidy = 1 if phased else 2

    # TODO: handle missing data
    for var in vcf(region):
        if phased:
            gts = np.ravel(np.asarray(var.genotype.array())[:, :2])
        else:
            gts = var.gt_types
        if flip_minor_alleles:
            af = np.mean(gts) / ploidy
            if af > 0.5:
                gts = ploidy - gts
                flip.append(True)
            else:
                flip.append(False)

        (idx,) = np.where(gts != 0)
        data.append(gts[idx])
        idxs.append(idx)
        ptrs.append(ptrs[-1] + len(idx))
        info.append(var.INFO)

    data = np.concatenate(data)
    idxs = np.concatenate(idxs)
    ptrs = np.array(ptrs)
    genotypes = csc_matrix((data, idxs, ptrs))
    flip = np.array(flip)
    return genotypes, flip, info


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
