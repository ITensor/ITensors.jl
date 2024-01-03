# Workaround for https://github.com/JuliaLang/julia/issues/52615
function _permutedims!(
  a_dest::AbstractArray{<:Any,N}, a_src::AbstractArray{<:Any,N}, perm::Tuple{Vararg{Int,N}}
) where {N}
  return permutedims!(a_dest, a_src, perm)
end
function _permutedims!(
  a_dest::AbstractArray{<:Any,0}, a_src::AbstractArray{<:Any,0}, perm::Tuple{}
)
  a_dest[] = a_src[]
  return a_dest
end
function _permutedims(a::AbstractArray{<:Any,N}, perm::Tuple{Vararg{Int,N}}) where {N}
  return permutedims(a, perm)
end
function _permutedims(a::AbstractArray{<:Any,0}, perm::Tuple{})
  return copy(a)
end

⊗(a::AbstractUnitRange) = a
function ⊗(a1::AbstractUnitRange, a2::AbstractUnitRange, as::AbstractUnitRange...)
  return ⊗(a1, ⊗(a2, as...))
end
⊗(a1::AbstractUnitRange, a2::AbstractUnitRange) = Base.OneTo(length(a1) * length(a2))
⊗() = Base.OneTo(1)

function fusedims(a::AbstractArray, permblocks...)
  return fusedims(a, blockedperm(permblocks...; length=Val(ndims(a))))
end

function fuseaxes(
  axes::Tuple{Vararg{AbstractUnitRange}}, blockedperm::AbstractBlockedPermutation
)
  blockedaxes = blockpermute(axes, blockedperm)
  return map(block -> ⊗(block...), blockedaxes)
end

function fuseaxes(a::AbstractArray, blockedperm::AbstractBlockedPermutation)
  return fuseaxes(axes(a), blockedperm)
end

function fusedims(a::AbstractArray, blockedperm::Tuple{Vararg{AbstractUnitRange}}...)
  ## # TODO: Add `canonicalizedims`.
  return reshape(a, flatten_tuples(blockedperm))
end

# Fuse adjacent dimensions
function fusedims(a::AbstractArray, blockedperm::BlockedTrivialPermutation)
  axes_fused = fuseaxes(a, blockedperm)
  return fusedims(a, axes_fused)
end

function fusedims(a::AbstractArray, blockedperm::BlockedPermutation)
  a_perm = _permutedims(a, Tuple(blockedperm))
  return fusedims(a_perm, trivialperm(blockedperm))
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
  a_dest = _permutedims(a_dest_perm, invperm(Tuple(blockedperm)))
  return a_dest
end

function splitdims!(
  a_dest::AbstractArray, a::AbstractArray, blockedperm::BlockedPermutation
)
  axes_dest = map(i -> axes(a_dest, i), Tuple(blockedperm))
  # TODO: Pass grouped axes.
  a_dest_perm = splitdims_adjacent(a, axes_dest)
  _permutedims!(a_dest, a_dest_perm, invperm(Tuple(blockedperm)))
  return a_dest
end
