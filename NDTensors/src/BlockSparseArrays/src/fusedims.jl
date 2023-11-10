function fusedims(a::AbstractArray, subperms::Tuple...)
  @assert ndims(a) == sum(length, subperms)
  perm = tuple_cat(subperms...)
  @assert isperm(perm)
  # TODO: Do this lazily?
  a_permuted = permutedims(a, perm)
  sublengths = length.(subperms)
  substops = cumsum(sublengths)
  substarts = (1, (Base.front(substops) .+ 1)...)
  # TODO: `step=1` is not required as of Julia 1.7.
  # Remove once we drop support for Julia 1.6.
  subranges = range.(substarts, substops; step=1)
  # Get a naive product of the axes in the subrange
  axes_prod = map(subranges) do subrange
    return ⊗(map(i -> axes(a_permuted, i), subrange)...)
  end
  a_reshaped = reshape(a_permuted, axes_prod)
  # Permute and merge the axes
  mergeperms = blockmergesortperm.(axes_prod)
  block_perms = map(mergeperm -> Block.(reduce(vcat, mergeperm)), mergeperms)
  a_blockperm = getindices(a_reshaped, block_perms...)
  axes_merged = blockmerge.(axes_prod, mergeperms)
  # TODO: Use `similar` here to preserve the type of `a`.
  a_merged = BlockSparseArray{eltype(a)}(blocklengths.(axes_merged)...)
  # TODO: Make this take advantage of the sparsity of `a_blockperm`.
  copyto!(a_merged, a_blockperm)
  return a_merged
end

function matricize(a::AbstractArray, left_dims::Tuple, right_dims::Tuple)
  return fusedims(a, left_dims, right_dims)
end
