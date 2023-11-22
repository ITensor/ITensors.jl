using ...NDTensors.TensorAlgebra: fusedims, splitdims

function TensorAlgebra.fusedims(na::AbstractNamedDimsArray, fusions::Pair...)
  # TODO: generalize to multiple fused groups of dimensions
  @assert isone(length(fusions))
  fusion = only(fusions)

  split_names = first(fusion)
  fused_name = last(fusion)

  split_dims = map(split_name -> findfirst(isequal(split_name), dimnames(na)), split_names)
  fused_dim = findfirst(isequal(fused_name), dimnames(na))
  @assert isnothing(fused_dim)

  unfused_dims = Tuple.(setdiff(1:ndims(na), split_dims))
  partitioned_perm = (unfused_dims..., split_dims)

  a_fused = fusedims(unname(na), partitioned_perm...)
  names_fused = (setdiff(dimnames(na), split_names)..., fused_name)
  return named(a_fused, names_fused)
end

function TensorAlgebra.splitdims(na::AbstractNamedDimsArray, splitters::Pair...)
  fused_names = first.(splitters)
  split_namedlengths = last.(splitters)
  splitters_unnamed = map(splitters) do splitter
    fused_name, split_namedlengths = splitter
    fused_dim = findfirst(isequal(fused_name), dimnames(na))
    split_lengths = unname.(split_namedlengths)
    return fused_dim => split_lengths
  end
  a_split = splitdims(unname(na), splitters_unnamed...)
  names_split = Any[tuple.(dimnames(na))...]
  for splitter in splitters
    fused_name, split_namedlengths = splitter
    fused_dim = findfirst(isequal(fused_name), dimnames(na))
    split_names = name.(split_namedlengths)
    names_split[fused_dim] = split_names
  end
  names_split = reduce((x, y) -> (x..., y...), names_split)
  return named(a_split, names_split)
end
