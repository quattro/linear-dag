using Pkg

# Install all dependencies for LinearARG.jl
Pkg.add("MatrixMarket")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("SparseArrays")
Pkg.add("LinearAlgebra")

println("All dependencies for LinearARG have been installed!")