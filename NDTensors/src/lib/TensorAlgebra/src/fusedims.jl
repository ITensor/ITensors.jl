⊗(a::AbstractUnitRange) = a
function ⊗(a1::AbstractUnitRange, a2::AbstractUnitRange, as::AbstractUnitRange...)
  return ⊗(a1, ⊗(a2, as...))
end
⊗(a1::AbstractUnitRange, a2::AbstractUnitRange) = Base.OneTo(length(a1) * length(a2))

function fusedims(a::AbstractArray, permblocks...)
  return fusedims(a, blockedperm(permblocks...; length=Val(ndims(a))))
end

function fusedims_adjacent(a::AbstractArray, ndims)
  lasts = cumsum(ndims)
  firsts = ntuple(i -> isone(i) ? 1 : lasts[i - 1] + 1, length(ndims))
  ranges = ntuple(i -> firsts[i]:lasts[i], length(ndims))
  fusedaxes = map(range -> ⊗(map(i -> axes(a, i), range)...), ranges)
  # TODO: Add `canonicalizedims`.
  return reshape(a, fusedaxes)
end

function fusedims(a::AbstractArray, blockedperm::BlockedPermutation)
  a_perm = permutedims(a, Tuple(blockedperm))
  return fusedims_adjacent(a_perm, length.(blocks(blockedperm)))
end

# splitdims(randn(4, 4), (1:2, 1:2), (1:2, 1:2))
function splitdims(a::AbstractArray, axesblocks::Tuple{Vararg{AbstractUnitRange}}...)
  # TODO: Add `uncanonicalizedims`.
  return reshape(a, length.(flatten_tuples(axesblocks)))
end

# splitdims(randn(4, 4), (2, 2), (2, 2))
function splitdims(a::AbstractArray, sizeblocks::Tuple{Vararg{Integer}}...)
  return splitdims(a, map(x -> Base.OneTo.(x), sizeblocks)...)
end

function blockedaxes(a::AbstractArray, sizeblocks::Pair...)
  axes_a = axes(a)
  axes_split = tuple.(axes(a))
  for (dim, sizeblock) in sizeblocks
    axes_split = Base.setindex(axes_split, Base.OneTo.(sizeblock), dim)
  end
  return axes_split
end

# splitdims(randn(4, 4), 2 => (1:2, 1:2))
function splitdims(a::AbstractArray, sizeblocks::Pair...)
  return splitdims(a, blockedaxes(a, sizeblocks...)...)
end

# splitdims(randn(4, 4), (1:2, 1:2, 1:2, 1:2))
function splitdims_adjacent(a::AbstractArray, axes)
  # TODO: Add `uncanonicalizedims`.
  return reshape(a, axes)
end

# splitdims(randn(4, 4), (1:2, 1:2, 1:2, 1:2))
function splitdims(a::AbstractArray, axes_dest::Tuple{Vararg{AbstractUnitRange}})
  # TODO: Pass grouped axes.
  return splitdims_adjacent(a, axes_dest)
end

# TODO: Is this needed?
function splitdims(
  a::AbstractArray,
  axes_dest::Tuple{Vararg{AbstractUnitRange}},
  blockedperm::BlockedPermutation,
)
  # TODO: Pass grouped axes.
  a_dest_perm = splitdims_adjacent(a, axes_dest)
  a_dest = permutedims(a_dest_perm, invperm(Tuple(blockedperm)))
  return a_dest
end

function splitdims!(
  a_dest::AbstractArray, a::AbstractArray, blockedperm::BlockedPermutation
)
  axes_dest = map(i -> axes(a_dest, i), Tuple(blockedperm))
  # TODO: Pass grouped axes.
  a_dest_perm = splitdims_adjacent(a, axes_dest)
  permutedims!(a_dest, a_dest_perm, invperm(Tuple(blockedperm)))
  return a_dest
end
