⊗(a::AbstractUnitRange) = a
function ⊗(a1::AbstractUnitRange, a2::AbstractUnitRange, as::AbstractUnitRange...)
  return ⊗(a1, ⊗(a2, as...))
end
⊗(a1::AbstractUnitRange, a2::AbstractUnitRange) = Base.OneTo(length(a1) * length(a2))

function fuse_adjacent_dims(a::AbstractArray, ndims)
  lasts = cumsum(ndims)
  firsts = ntuple(i -> isone(i) ? 1 : lasts[i - 1] + 1, length(ndims))
  ranges = ntuple(i -> firsts[i]:lasts[i], length(ndims))
  fusedaxes = map(range -> ⊗(map(i -> axes(a, i), range)...), ranges)
  # TODO: Add `canonicalizedims`.
  return reshape(a, fusedaxes)
end

function fusedims(a::AbstractArray, blockedperm::BlockedPermutation)
  a_perm = permutedims(a, Tuple(blockedperm))
  return fuse_adjacent_dims(a_perm, length.(blocks(blockedperm)))
end

function split_adjacent_dims(a::AbstractArray, axes)
  # TODO: Add `uncanonicalizedims`.
  return reshape(a, axes)
end

function splitdims!(
  a_dest::AbstractArray, a::AbstractArray, blockedperm::BlockedPermutation
)
  axes_dest = map(i -> axes(a_dest, i), Tuple(blockedperm))
  # TODO: Pass grouped axes.
  a_dest_perm = split_adjacent_dims(a, axes_dest)
  permutedims!(a_dest, a_dest_perm, invperm(Tuple(blockedperm)))
  return a_dest
end
