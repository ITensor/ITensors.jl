module ITensorsTensorOperationsExt

using ITensors: ITensors, ITensor, dim, inds
using NDTensors.AlgorithmSelection: @Algorithm_str
using TensorOperations: TensorOperations, optimaltree

"""
    optimal_contraction_sequence(T)

Returns a contraction sequence for contracting the tensors `T`. The sequence is
generally optimal and is found via the optimaltree function in TensorOperations.jl which must be loaded.
"""
function ITensors.optimal_contraction_sequence(
  As::Union{Vector{<:ITensor},Tuple{Vararg{ITensor}}}
)
  network = collect.(inds.(As))
  inds_to_dims = Dict(i => Float64(dim(i)) for i in unique(reduce(vcat, network)))
  seq, _ = optimaltree(network, inds_to_dims)
  return seq
end

end
