include("LinearARG.jl")
using .LinearARG
using MatrixMarket, CSV, DataFrames, SparseArrays, LinearAlgebra

# Parse command line arguments
function parse_arguments()
    if length(ARGS) >= 2
        linarg_path = ARGS[1]
        num_vectors = parse(Int, ARGS[2])
    elseif length(ARGS) == 1
        linarg_path = ARGS[1]
        num_vectors = 1  # Default value
    else
        # Default values if no arguments provided
        linarg_path = "../data/1kg_chr1"
        num_vectors = 1
    end
    
    return linarg_path, num_vectors
end

linarg_path, num_vectors = parse_arguments()

linarg = read_linarg(linarg_path);
n,m = size(linarg);
println("n,m: ", n, ", ", m)

# Compiles the first time it is run
v = ones(1, n);
z = v * linarg;

println("Right-multiply:")
v = rand(num_vectors, n);
@time z = v * linarg;

u = ones(m, 1);
z = linarg * u;

println("Left-multiply:")
u = rand(m, num_vectors);
@time z = linarg * u;
