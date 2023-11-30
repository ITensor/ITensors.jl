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

BlockArrays.blockedrange(a::AbstractGradedUnitRange) = error("Not implemented")
sectors(a::AbstractGradedUnitRange) = error("Not implemented")
scale_factor(a::AbstractGradedUnitRange) = error("Not implemented")

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

sector(a::AbstractGradedUnitRange, b::Block{1}) = sectors(a)[only(b.n)]
sector(a::AbstractGradedUnitRange, I::Integer) = sector(a, findblock(a, I))

# Tensor product, no sorting
function tensor_product(a1::AbstractGradedUnitRange, a2::AbstractGradedUnitRange)
  a = tensor_product(blockedrange(a1), blockedrange(a2))
  sectors_a = map(Iterators.product(sectors(a1), sectors(a2))) do (l1, l2)
    return fuse(scale_factor(a1) * l1, scale_factor(a2) * l2)
  end
  return gradedrange(a, vec(sectors_a))
end

function Base.show(io::IO, mimetype::MIME"text/plain", a::AbstractGradedUnitRange)
  show(io, mimetype, sectors(a))
  println(io)
  println(io, "Scale factor = ", scale_factor(a))
  return show(io, mimetype, blockedrange(a))
end

function blockmerge(a::AbstractGradedUnitRange, grouped_perm::Vector{Vector{Int}})
  merged_sectors = map(group -> sector(a, Block(first(group))), grouped_perm)
  lengths = blocklengths(a)
  merged_lengths = map(group -> sum(@view(lengths[group])), grouped_perm)
  return gradedrange(merged_sectors, merged_lengths)
end

# Sort and merge by the grade of the blocks.
function blockmergesort(a::AbstractGradedUnitRange)
  grouped_perm = blockmergesortperm(a)
  return blockmerge(a, grouped_perm)
end

# Get the permutation for sorting, then group by common elements.
# groupsortperm([2, 1, 2, 3]) == [[2], [1, 3], [4]]
function blockmergesortperm(a::AbstractGradedUnitRange)
  return groupsortperm(sectors(a))
end

function sub_axis(a::AbstractGradedUnitRange, blocks)
  a_sub = sub_axis(blockedrange(a), blocks)
  sectors_sub = map(b -> sector(a, b), Indices(blocks))
  return AbstractGradedUnitRange(a_sub, sectors_sub)
end

function fuse(a1::AbstractGradedUnitRange, a2::AbstractGradedUnitRange)
  a = tensor_product(a1, a2)
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
