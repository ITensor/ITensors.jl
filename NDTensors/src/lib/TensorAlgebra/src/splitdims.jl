using .BaseExtensions: _permutedims, _permutedims!

to_axis(a::AbstractUnitRange) = a
to_axis(n::Integer) = Base.OneTo(n)

function blockedaxes(a::AbstractArray, sizeblocks::Pair...)
  axes_a = axes(a)
  axes_split = tuple.(axes(a))
  for (dim, sizeblock) in sizeblocks
    # TODO: Handle conversion from length to range!
    axes_split = Base.setindex(axes_split, to_axis.(sizeblock), dim)
  end
  return axes_split
end

# splitdims(randn(4, 4), 1:2, 1:2, 1:2, 1:2)
function splitdims(a::AbstractArray, axes::AbstractUnitRange...)
  # TODO: Add `uncanonicalizedims`.
  # TODO: Need `length` since `reshape` doesn't accept `axes`,
  # maybe make a `reshape_axes` function.
  return reshape(a, length.(axes)...)
end

# splitdims(randn(4, 4), (1:2, 1:2), (1:2, 1:2))
function splitdims(a::AbstractArray, axesblocks::Tuple{Vararg{AbstractUnitRange}}...)
  # TODO: Add `uncanonicalizedims`.
  return splitdims(a, flatten_tuples(axesblocks)...)
end

# Fix ambiguity issue
splitdims(a::AbstractArray) = a

# splitdims(randn(4, 4), (2, 2), (2, 2))
function splitdims(a::AbstractArray, sizeblocks::Tuple{Vararg{Integer}}...)
  return splitdims(a, map(x -> Base.OneTo.(x), sizeblocks)...)
end

# splitdims(randn(4, 4), 2 => (1:2, 1:2))
function splitdims(a::AbstractArray, sizeblocks::Pair...)
  return splitdims(a, blockedaxes(a, sizeblocks...)...)
end

# TODO: Is this needed?
function splitdims(
  a::AbstractArray,
  axes_dest::Tuple{Vararg{AbstractUnitRange}},
  blockedperm::BlockedPermutation,
)
  # TODO: Pass grouped axes.
  a_dest_perm = splitdims(a, axes_dest...)
  a_dest = _permutedims(a_dest_perm, invperm(Tuple(blockedperm)))
  return a_dest
end

function splitdims!(
  a_dest::AbstractArray, a::AbstractArray, blockedperm::BlockedPermutation
)
  axes_dest = map(i -> axes(a_dest, i), Tuple(blockedperm))
  # TODO: Pass grouped axes.
  a_dest_perm = splitdims(a, axes_dest...)
  _permutedims!(a_dest, a_dest_perm, invperm(Tuple(blockedperm)))
  return a_dest
end
