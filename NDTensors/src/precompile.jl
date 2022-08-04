function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    # Base.precompile(Tuple{typeof(contract),DenseTensor{Float64, 2, Tuple{Index{Int64}, Index{Int64}}, Dense{Float64, Vector{Float64}}},Tuple{Int64, Int64},DenseTensor{Float64, 2, Tuple{Index{Int64}, Index{Int64}}, Dense{Float64, Vector{Float64}}},Tuple{Int64, Int64},Tuple{Int64, Int64}})   # time: 0.24672069
end
