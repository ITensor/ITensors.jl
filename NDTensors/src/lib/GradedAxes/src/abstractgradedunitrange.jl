using BlockArrays:
  BlockArrays,
  AbstractBlockVector,
  Block,
  BlockRange,
  BlockedUnitRange,
  blockaxes,
  blockedrange,
  blockfirsts,
  blocklasts,
  blocklength,
  blocklengths,
  findblock
using Dictionaries: Dictionary

# Fuse two symmetry labels
fuse(l1, l2) = error("Not implemented")

abstract type AbstractGradedUnitRange{T,G} <: AbstractUnitRange{Int} end

"""
    blockedrange(::AbstractGradedUnitRange)

The blocked range of values the graded space can take.
"""
BlockArrays.blockedrange(::AbstractGradedUnitRange) = error("Not implemented")

"""
    nondual_sectors(::AbstractGradedUnitRange)

A vector of the non-dual sectors of the graded space, one for each block in the space.
"""
nondual_sectors(::AbstractGradedUnitRange) = error("Not implemented")

"""
    isdual(::AbstractGradedUnitRange)

If the graded space is dual or not.
"""
isdual(::AbstractGradedUnitRange) = error("Not implemented")

# Overload if there are contravariant and covariant
# spaces.
dual(a::AbstractGradedUnitRange) = a

# BlockArrays block axis interface
BlockArrays.blockaxes(a::AbstractGradedUnitRange) = blockaxes(blockedrange(a))
Base.getindex(a::AbstractGradedUnitRange, b::Block{1}) = blockedrange(a)[b]
BlockArrays.blockfirsts(a::AbstractGradedUnitRange) = blockfirsts(blockedrange(a))
BlockArrays.blocklasts(a::AbstractGradedUnitRange) = blocklasts(blockedrange(a))
function BlockArrays.findblock(a::AbstractGradedUnitRange, k::Integer)
  return findblock(blockedrange(a), k)
end

# Base axis interface
Base.getindex(a::AbstractGradedUnitRange, I::Integer) = blockedrange(a)[I]
Base.first(a::AbstractGradedUnitRange) = first(blockedrange(a))
Base.last(a::AbstractGradedUnitRange) = last(blockedrange(a))
Base.length(a::AbstractGradedUnitRange) = length(blockedrange(a))
Base.step(a::AbstractGradedUnitRange) = step(blockedrange(a))
Base.unitrange(b::AbstractGradedUnitRange) = first(b):last(b)

nondual_sector(a::AbstractGradedUnitRange, b::Block{1}) = nondual_sectors(a)[only(b.n)]
function sector(a::AbstractGradedUnitRange, b::Block{1})
  return isdual(a) ? dual(nondual_sector(a, b)) : nondual_sector(a, b)
end
sector(a::AbstractGradedUnitRange, I::Integer) = sector(a, findblock(a, I))
sectors(a) = map(s -> isdual(a) ? dual(s) : s, nondual_sectors(a))

function default_isdual(a1::AbstractGradedUnitRange, a2::AbstractGradedUnitRange)
  return isdual(a1) && isdual(a2)
end

# Tensor product, no sorting
function tensor_product(
  a1::AbstractGradedUnitRange, a2::AbstractGradedUnitRange; isdual=default_isdual(a1, a2)
)
  a = tensor_product(blockedrange(a1), blockedrange(a2))
  nondual_sectors_a = vec(
    map(Iterators.product(sectors(a1), sectors(a2))) do (l1, l2)
      return fuse(isdual ? dual(l1) : l1, isdual ? dual(l2) : l2)
    end,
  )
  return gradedrange(a, nondual_sectors_a, isdual)
end

function Base.show(io::IO, mimetype::MIME"text/plain", a::AbstractGradedUnitRange)
  show(io, mimetype, nondual_sectors(a))
  println(io)
  println(io, "isdual = ", isdual(a))
  return show(io, mimetype, blockedrange(a))
end

# TODO: This is not part of the `BlockArrays` interface, should
# we give this a different name?
function Base.length(a::AbstractGradedUnitRange, b::Block{1})
  return blocklengths(a)[Int(b)]
end

# Sort and merge by the grade of the blocks.
function blockmergesort(a::AbstractGradedUnitRange)
  return a[blockmergesortperm(a)]
end

function blocksortperm(a::AbstractGradedUnitRange)
  # TODO: `rev=isdual(a)`  may not be correct for symmetries beyond `U(1)`.
  return Block.(sortperm(nondual_sectors(a); rev=isdual(a)))
end

# Get the permutation for sorting, then group by common elements.
# groupsortperm([2, 1, 2, 3]) == [[2], [1, 3], [4]]
function blockmergesortperm(a::AbstractGradedUnitRange)
  # If it is dual, reverse the sorting so the sectors
  # end up sorted in the same way whether or not the space
  # is dual.
  # TODO: `rev=isdual(a)`  may not be correct for symmetries beyond `U(1)`.
  return Block.(groupsortperm(nondual_sectors(a); rev=isdual(a)))
end

function Base.getindex(a::AbstractGradedUnitRange, I::AbstractVector{<:Block})
  nondual_sectors_sub = map(b -> nondual_sector(a, b), I)
  blocklengths_sub = map(b -> length(a, b), I)
  return gradedrange(nondual_sectors_sub, blocklengths_sub, isdual(a))
end

function Base.getindex(
  a::AbstractGradedUnitRange, grouped_perm::AbstractBlockVector{<:Block}
)
  merged_nondual_sectors = map(blocks(grouped_perm)) do group
    return nondual_sector(a, first(group))
  end
  # Length of each block
  merged_lengths = map(blocks(grouped_perm)) do group
    return sum(b -> length(a, b), group)
  end
  return gradedrange(merged_nondual_sectors, merged_lengths, isdual(a))
end

function fuse(
  a1::AbstractGradedUnitRange, a2::AbstractGradedUnitRange; isdual=default_isdual(a1, a2)
)
  a = tensor_product(a1, a2; isdual)
  return blockmergesort(a)
end

# Broadcasting
# This removes the block structure when mixing dense and graded blocked arrays,
# maybe keep the block structure (like `BlockArrays` does).
Broadcast.axistype(a1::AbstractGradedUnitRange, a2::Base.OneTo) = a2
Broadcast.axistype(a1::Base.OneTo, a2::AbstractGradedUnitRange) = a1
