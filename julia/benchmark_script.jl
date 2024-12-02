include("LinearARG.jl")
using .LinearARG
using MatrixMarket, CSV, DataFrames, SparseArrays, LinearAlgebra

linarg_path = "/Users/loconnor/Dropbox/linearARG/linearg_shared/data/linarg/1kg_med_101924"

linarg = read_linarg(linarg_path);
n,m = size(linarg);
v = randn(1000, n);
@time z = v * linarg;
@time z = v * linarg;
# linarg_reversed = LinearARG.reverse_indices(linarg);
# @time z = v * linarg_reversed;

u = randn(m, 1000);
@time z = linarg * u;
@time z = linarg * u;

# @time z = linarg_reversed * u;



