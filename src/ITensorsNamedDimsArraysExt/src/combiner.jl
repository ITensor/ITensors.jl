# Combiner
using ..NDTensors.NamedDimsArrays: AbstractNamedDimsArray, dimnames, name
using ..NDTensors.TensorAlgebra: TensorAlgebra, fusedims, splitdims
using NDTensors: NDTensors, Tensor, Combiner

function ITensors._contract(na::AbstractNamedDimsArray, c::Tensor{<:Any,<:Any,<:Combiner})
  split_names = name.(NDTensors.uncombinedinds(c))
  fused_name = name(NDTensors.combinedind(c))

  # Use to determine if we are doing fusion or splitting.
  split_dims = map(split_name -> findfirst(isequal(split_name), dimnames(na)), split_names)
  fused_dim = findfirst(isequal(fused_name), dimnames(na))

  return if isnothing(fused_dim)
    # Dimension fusion (joining, combining)
    @assert all(!isnothing, split_dims)
    fusedims(na, split_names => fused_name)
  else
    # Dimension unfusion (splitting, uncombining)
    @assert all(isnothing, split_dims)

    split_dims = NamedInt.(NDTensors.uncombinedinds(c))
    splitdims(na, fused_name => split_dims)
  end
end
