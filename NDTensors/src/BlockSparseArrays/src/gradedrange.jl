struct GradedBlockedUnitRange{T,G} <: AbstractUnitRange{Int}
  blockedrange::BlockedUnitRange{T}
  sectors::Dictionary{Block{1,Int},G}
end

# TODO: Rename `GradedBlockedUnitRange`?
function gradedrange(a::BlockedUnitRange, sectors::Vector)
  return GradedBlockedUnitRange(a, Dictionary(BlockRange(blocklength(a)), sectors))
end

function gradedrange(lengths::Vector{<:Integer}, sectors::Vector)
  return gradedrange(blockedrange(lengths), sectors)
end

BlockArrays.blockaxes(a::GradedBlockedUnitRange) = blockaxes(a.blockedrange)
Base.getindex(a::GradedBlockedUnitRange, b::Block{1}) = a.blockedrange[b]
BlockArrays.blockfirsts(a::GradedBlockedUnitRange) = blockfirsts(a.blockedrange)
BlockArrays.blocklasts(a::GradedBlockedUnitRange) = blocklasts(a.blockedrange)
BlockArrays.findblock(a::GradedBlockedUnitRange, k) = findblock(a.blockedrange, k)

Base.getindex(a::GradedBlockedUnitRange, I::Integer) = a.blockedrange[I]
Base.first(a::GradedBlockedUnitRange) = first(a.blockedrange)
Base.last(a::GradedBlockedUnitRange) = last(a.blockedrange)
Base.length(a::GradedBlockedUnitRange) = length(a.blockedrange)
Base.step(a::GradedBlockedUnitRange) = step(a.blockedrange)
Base.unitrange(b::GradedBlockedUnitRange) = first(b):last(b)

sector(a::GradedBlockedUnitRange, b::Block{1}) = a.sectors[b]

# Tensor product
function ⊗(a1::GradedBlockedUnitRange, a2::GradedBlockedUnitRange)
  a = a1.blockedrange ⊗ a2.blockedrange
  sectors = vec(map(sum, Iterators.product(a1.sectors, a2.sectors)))
  return gradedrange(a, sectors)
end

# Slicing
## using BlockArrays: BlockRange, _BlockedUnitRange
Base.@propagate_inbounds function Base.getindex(
  b::GradedBlockedUnitRange, KR::BlockRange{1}
)
  cs = blocklasts(b)
  isempty(KR) && return _BlockedUnitRange(1, cs[1:0])
  K, J = first(KR), last(KR)
  k, j = Integer(K), Integer(J)
  bax = blockaxes(b, 1)
  @boundscheck K in bax || throw(BlockBoundsError(b, K))
  @boundscheck J in bax || throw(BlockBoundsError(b, J))
  K == first(bax) && return _BlockedUnitRange(first(b), cs[k:j])
  return _BlockedUnitRange(cs[k - 1] + 1, cs[k:j])
end

function Base.show(io::IO, mimetype::MIME"text/plain", a::GradedBlockedUnitRange)
  return Base.show(io, mimetype, a.blockedrange)
end

function blockmerge(s::GradedBlockedUnitRange, grouped_perm::Vector{Vector{Int}})
  merged_sectors = map(group -> s.sectors[Block(first(group))], grouped_perm)
  lengths = blocklengths(s)
  merged_lengths = map(group -> sum(@view(lengths[group])), grouped_perm)
  return gradedrange(merged_lengths, merged_sectors)
end

# Sort and merge by the grade of the blocks.
function blockmergesort(s::GradedBlockedUnitRange)
  grouped_perm = blockmergesortperm(s)
  return blockmerge(s, grouped_perm)
end

# Get the permutation for sorting, then group by common elements.
# groupsortperm([2, 1, 2, 3]) == [[2], [1, 3], [4]]
function blockmergesortperm(s::GradedBlockedUnitRange)
  return groupsortperm(collect(s.sectors))
end

function sub_axis(a::GradedBlockedUnitRange, blocks)
  a_sub = sub_axis(a.blockedrange, blocks)
  sectors_sub = map(b -> sector(a, b), Indices(blocks))
  return GradedBlockedUnitRange(a_sub, sectors_sub)
end
