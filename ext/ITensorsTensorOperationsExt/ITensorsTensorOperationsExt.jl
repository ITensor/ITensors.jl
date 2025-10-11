module ITensorsTensorOperationsExt

using ITensors: ITensors, ITensor, dim, inds
using TensorOperations: optimaltree

function ITensors.optimal_contraction_sequence(
        As::Union{Vector{<:ITensor}, Tuple{Vararg{ITensor}}}
    )
    network = collect.(inds.(As))
    inds_to_dims = Dict(i => Float64(dim(i)) for i in unique(reduce(vcat, network)))
    seq, _ = optimaltree(network, inds_to_dims)
    return seq
end

end
