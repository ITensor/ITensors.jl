fuse(a1::AbstractUnitRange, a2::AbstractUnitRange) = Base.OneTo(length(a1) * length(a2))
fuse(a...) = foldl(fuse, a; init=Base.OneTo(1))

## # TODO: Support subperm that is an `Int`, representing
## # a dimension that won't be self-fused (i.e. won't involve
## # and block permutations and mergers).
## function fusedims(a::AbstractArray, subperms::Tuple...)
##   @assert ndims(a) == sum(length, subperms)
##   perm = reduce((x, y) -> (x..., y...), subperms)
##   @assert isperm(perm)
##   # TODO: Do this lazily?
##   a_permuted = permutedims(a, perm)
##   sublengths = length.(subperms)
##   substops = cumsum(sublengths)
##   substarts = (1, (Base.front(substops) .+ 1)...)
##   # TODO: `step=1` is not required as of Julia 1.7.
##   # Remove once we drop support for Julia 1.6.
##   subranges = range.(substarts, substops; step=1)
##   # Get a naive product of the axes in the subrange
##   axes_prod = map(subranges) do subrange
##     return fuse(map(i -> axes(a_permuted, i), subrange)...)
##   end
##   a_reshaped = reshape(a_permuted, axes_prod)
## 
##   # Particular to `BlockSparseArray`
##   # with graded unit range, add this back.
##   # Permute and merge the axes
##   mergeperms = blockmergesortperm.(axes_prod)
## end

# TODO: Support subperm that is an `Int`, representing
# a dimension that won't be self-fused (i.e. won't involve
# and block permutations and mergers).
# This is based on the `BlockSparseArray` version
# TODO: Restrict `subperms` to `Tuple`s?
function fusedims(a::AbstractArray, subperms...)
  @assert ndims(a) == sum(length, subperms)
  # perm = tuple_cat(subperms...)
  perm = reduce((x, y) -> (x..., y...), subperms)

  @assert isperm(perm)
  # TODO: Do this lazily?
  a_permuted = if iszero(ndims(a)) && iszero(length(perm))
    # TODO: Raise an issue with Base Julia.
    # TODO: `copy` here?
    a
  else
    permutedims(a, perm)
  end
  sublengths = length.(subperms)
  substops = cumsum(sublengths)
  substarts = (1, (Base.front(substops) .+ 1)...)
  # TODO: `step=1` is not required as of Julia 1.7.
  # Remove once we drop support for Julia 1.6.
  subranges = range.(substarts, substops; step=1)
  # Get a naive product of the axes in the subrange
  axes_prod = map(subranges) do subrange
    return fuse(map(i -> axes(a_permuted, i), subrange)...)
  end
  a_reshaped = reshape(a_permuted, axes_prod)
  return a_reshaped

  ## # TODO: Particular to `BlockSparseArray`
  ## # with graded unit range, add this back.
  ## # Permute and merge the axes
  ## # Permute and merge the axes
  ## mergeperms = blockmergesortperm.(axes_prod)
  ## block_perms = map(mergeperm -> Block.(reduce(vcat, mergeperm)), mergeperms)
  ## a_blockperm = getindices(a_reshaped, block_perms...)
  ## axes_merged = blockmerge.(axes_prod, mergeperms)
  ## # TODO: Use `similar` here to preserve the type of `a`.
  ## a_merged = BlockSparseArray{eltype(a)}(blocklengths.(axes_merged)...)
  ## a_merged = similar(a, axes_merged)
  ## TODO: Make this take advantage of the sparsity of `a_blockperm`.
  ## TODO: Use `VectorInterface.zerovector` and `VectorInterface.add!!`?
  ## copyto!(a_merged, a_blockperm)
  ## return a_merged
end

function matricize(a::AbstractArray, left_dims::Tuple, right_dims::Tuple)
  return fusedims(a, left_dims, right_dims)
end

# TODO: Remove this version.
function matricize(a::AbstractArray, biperm)
  return fusedims(a, biperm[1], biperm[2])
end

function matricize(a::AbstractArray, biperm::BipartitionedPermutation)
  return fusedims(a, biperm[1], biperm[2])
end

# splitdims(randn(12, 5, 12), 1 => (3, 4), 2 => (4, 3)) -> 3 × 4 × 5 × 4 × 3
function splitdims(a::AbstractArray, splitters::Pair...)
  splitdims = first.(splitters)
  split_sizes = last.(splitters)
  size_unsplit = size(a)
  size_split = Any[size_unsplit...]
  for (split_dim, split_size) in splitters
    size_split[split_dim] = split_size
  end
  size_split = reduce((x, y) -> (x..., y...), size_split)
  return reshape(a, size_split)
end

# TODO: Make this more generic, i.e. for `BlockSparseArray`.
function unmatricize(a::AbstractArray, axes_codomain, axes_domain)
  # TODO: Call `splitdims`.
  return reshape(a, (axes_codomain..., axes_domain...))
end

## function fusedims(a::AbstractArray, perm_partitions...)
##   error("Not implemented")
## end
## 
## matricize(a::AbstractArray, biperm) = matricize(a, BipartitionedPermutation(biperm...))
## 
## # TODO: Make this more generic, i.e. for `BlockSparseArray`.
## function matricize(a::AbstractArray, biperm::BipartitionedPermutation)
##   # Permute and fuse the axes
##   axes_src = axes(a)
##   axes_codomain = map(i -> axes_src[i], biperm[1])
##   axes_domain = map(i -> axes_src[i], biperm[2])
##   axis_codomain_fused = fuse(axes_codomain...)
##   axis_domain_fused = fuse(axes_domain...)
##   # Permute the array
##   perm = flatten(biperm)
##   a_permuted = permutedims(a, perm)
##   return reshape(a_permuted, (axis_codomain_fused, axis_domain_fused))
## end
