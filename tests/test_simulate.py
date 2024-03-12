import linear_dag as ld


if __name__ == "__main__":
    simulated = ld.Simulate()
    simulated.simulate_example(example="3-2-1", ns=100)
    linarg = simulated.linarg()
    linarg.form_initial_linarg()
    linarg.create_triolist()
    linarg.find_recombinations()
