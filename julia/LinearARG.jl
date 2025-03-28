module LinearARG
using MatrixMarket, CSV, DataFrames, SparseArrays, LinearAlgebra

export read_linarg, Linarg

struct Linarg
    A::SparseMatrixCSC
    sample_indices::Vector{Int}
    variants::DataFrame
end


"Read a linear ARG from a trio of files named *.mtx, *.psam, *.pvar"
function read_linarg(linarg_path::String)

    samples = CSV.read(linarg_path * ".psam", DataFrame; delim=' ', comment="##")
    variants = CSV.read(linarg_path * ".pvar", DataFrame; delim='\t', comment="##")
    
    A = MatrixMarket.mmread(linarg_path * ".mtx")
    @assert istril(A) "Adjacency matrix should be lower-triangular"

    function fix_index_column(df::DataFrame)
        rename!(df, Symbol.(replace.(string.(names(df)), r"^#" => ""))) # strip '#' from first column name
        @assert "IDX" in names(df)
        df.IDX .= df.IDX .+ 1  # 1-indexed
        return df
    end

    function parse_info_field(df::DataFrame)
        @assert "INFO" in names(df)

        pattern = r"IDX=(\d+);FLIP=([0-9.]+|True|False)"
        df = transform(df, :INFO => ByRow(x -> match(pattern, x)) => :match)

        df.IDX = parse.(Int, getindex.(df.match, 1))
        df.FLIP = map(m -> begin
            flip_value = m[2]
            if flip_value == "True"
                return true
            elseif flip_value == "False"
                return false
            else
                return parse(Float64, flip_value) == 1.0
            end
        end, df.match)
        
        return df
    end
    
    samples = fix_index_column(samples)
    variants = parse_info_field(variants)
    variants = fix_index_column(variants)

    return Linarg(LowerTriangular(A), samples.IDX, variants)
end

"Size of a Linarg object is number of samples by number of variants"
function Base.:size(linarg::Linarg, dim::Union{Int, Nothing}=nothing)
    result = length(linarg.sample_indices), size(linarg.variants,1)
    if dim in (1,2)
        return result[dim]
    elseif dim === nothing
        return result
    end
    throw(ArgumentError("Dimension argument must be 1, 2, or nothing"))
    
end

"Length of a Linarg object is the dimension of its adjacency matrix"
function Base.:length(linarg::Linarg)
    return size(linarg.A, 1) 
end

function I_minus_A(linarg::Linarg)
    return sparse(I,length(linarg), length(linarg)) - linarg.A
end
function nnz(linarg::Linarg)
    return nnz(linarg.A)
end
function reverse_indices(linarg::Linarg)
    A = linarg.A[end:-1:1, end:-1:1]
    n = length(linarg)
    sample_indices = (n+1) .- linarg.sample_indices
    variants = copy(linarg.variants)
    variants.IDX = (n+1) .- linarg.variants.IDX
    return Linarg(A, sample_indices, variants)
end

"Linarg-by-matrix multiplication"
function Base.:*(linarg::Linarg, other::Matrix{Float64})
    @assert size(other,1) == size(linarg,2) "other should have number of rows equal to the number of variants"
    y = zeros(length(linarg), size(other,2));
    y[linarg.variants.IDX, :] .= other;
    y .= I_minus_A(linarg) \ y;
    return y[linarg.sample_indices, :]
end

"Matrix-by-linarg multiplication"
function Base.:*(other::Matrix{Float64}, linarg::Linarg)
    @assert size(other,2) == size(linarg,1) "other should have number of columns equal to the number of samples"
    y = zeros(size(other,1), length(linarg));
    y[:, linarg.sample_indices] .= other;
    y .= y / I_minus_A(linarg);
    return y[:, linarg.variants.IDX]
end

end