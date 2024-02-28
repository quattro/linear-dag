import csv
import time

import numpy as np

from linear_dag import Trios


def read_trios_file(filename):
    """
    Read trio data from a file and return it as a list of tuples.
    """
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=",")  # Assuming the file is tab-delimited
        trios_data = [
            (int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[6]))
            for row in reader
        ]
    return trios_data


def create_trios_instance_from_file(filename, index_offset):
    trios_data = read_trios_file(filename)
    trios_instance = Trios(len(trios_data))  # Initialize Trios instance with the correct length
    trios_instance.add_trios(trios_data, index_offset)  # Add all trios at once using the modified add_trios method
    trios_instance.collect_cliques()
    trios_instance.check_properties(10)
    return trios_instance


def main():
    filename = "/Users/nicholas/Dropbox/collab/nick-luke/linearg_shared/triolist.txt"
    T = create_trios_instance_from_file(filename, 1)
    cs = T.clique_size()
    print(np.sum(cs))
    start_time = time.time()
    c = T.max_clique()
    while c > 0:
        T.factor_clique(c)
        c = T.max_clique()

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
    T.check_properties(1000)
    print(T.extract_trio(1003))
    edges = T.fill_edgelist()
    print(np.array(edges[-10:, :]))


if __name__ == "__main__":
    main()
