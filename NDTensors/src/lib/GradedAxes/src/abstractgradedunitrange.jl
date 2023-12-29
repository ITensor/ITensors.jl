using BlockArrays:
  BlockArrays,
  Block,
  BlockRange,
  BlockedUnitRange,
  blockaxes,
  blockedrange,
  blockfirsts,
  blocklasts,
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

function blockmerge(a::AbstractGradedUnitRange, grouped_perm::Vector{Vector{Int}})
  merged_nondual_sectors = map(
    group -> nondual_sector(a, Block(first(group))), grouped_perm
  )
  lengths = blocklengths(a)
  merged_lengths = map(group -> sum(@view(lengths[group])), grouped_perm)
  return gradedrange(merged_nondual_sectors, merged_lengths, isdual(a))
end

# Sort and merge by the grade of the blocks.
function blockmergesort(a::AbstractGradedUnitRange)
  grouped_perm = blockmergesortperm(a)
  return blockmerge(a, grouped_perm)
end

# Get the permutation for sorting, then group by common elements.
# groupsortperm([2, 1, 2, 3]) == [[2], [1, 3], [4]]
function blockmergesortperm(a::AbstractGradedUnitRange)
  # If it is dual, reverse the sorting so the sectors
  # end up sorted in the same way whether or not the space
  # is dual.
  return groupsortperm(nondual_sectors(a); rev=isdual(a))
end

function sub_axis(a::AbstractGradedUnitRange, blocks)
  a_sub = sub_axis(blockedrange(a), blocks)
  sectors_sub = map(b -> sector(a, b), Indices(blocks))
  return AbstractGradedUnitRange(a_sub, sectors_sub)
end

function fuse(
  a1::AbstractGradedUnitRange, a2::AbstractGradedUnitRange; isdual=default_isdual(a1, a2)
)
  a = tensor_product(a1, a2; isdual)
  return blockmergesort(a)
end

## TODO: Add this back.
## # Slicing
## ## using BlockArrays: BlockRange, _BlockedUnitRange
## Base.@propagate_inbounds function Base.getindex(
##   b::AbstractGradedUnitRange, KR::BlockRange{1}
## )
##   cs = blocklasts(b)
##   isempty(KR) && return _BlockedUnitRange(1, cs[1:0])
##   K, J = first(KR), last(KR)
##   k, j = Integer(K), Integer(J)
##   bax = blockaxes(b, 1)
##   @boundscheck K in bax || throw(BlockBoundsError(b, K))
##   @boundscheck J in bax || throw(BlockBoundsError(b, J))
##   K == first(bax) && return _BlockedUnitRange(first(b), cs[k:j])
##   return _BlockedUnitRange(cs[k - 1] + 1, cs[k:j])
## end
