struct GradedBlockedUnitRange{T,G} <: AbstractUnitRange{Int}
  blockedrange::BlockedUnitRange{T}
  grades::Vector{G}
end
gradedblockedrange(lengths::Vector{<:Integer}, grades::Vector) = GradedBlockedUnitRange(blockedrange(lengths), grades)
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

# TODO: Make `grades` a `Dictionary` with keys of `Block{1}`?
grade(a::GradedBlockedUnitRange, b::Block{1}) = a.grades[Int(b)]

# Slicing
## using BlockArrays: BlockRange, _BlockedUnitRange
Base.@propagate_inbounds function Base.getindex(b::GradedBlockedUnitRange, KR::BlockRange{1})
  cs = blocklasts(b)
  isempty(KR) && return _BlockedUnitRange(1,cs[1:0])
  K,J = first(KR),last(KR)
  k,j = Integer(K),Integer(J)
  bax = blockaxes(b,1)
  @boundscheck K in bax || throw(BlockBoundsError(b,K))
  @boundscheck J in bax || throw(BlockBoundsError(b,J))
  K == first(bax) && return _BlockedUnitRange(first(b),cs[k:j])
  _BlockedUnitRange(cs[k-1]+1,cs[k:j])
end

Base.show(io::IO, mimetype::MIME"text/plain", a::GradedBlockedUnitRange) =
  Base.show(io, mimetype, a.blockedrange)

# TODO: Delete
## # Fuse the blocks, sorting and merging according to the grades.
## function NDTensors.outer(s1::GradedBlockedUnitRange, s2::GradedBlockedUnitRange)
##   fused_range = s1.blockedrange ⊗ s2.blockedrange
##   fused_grades = vec(map(sum, Iterators.product(s1.grades, s2.grades)))
##   return GradedBlockedUnitRange(fused_range, fused_grades)
## end

function blockmerge(s::GradedBlockedUnitRange, grouped_perm::Vector{Vector{Int}})
  merged_grades = map(group -> s.grades[first(group)], grouped_perm)
  lengths = blocklengths(s)
  merged_lengths = map(group -> sum(@view(lengths[group])), grouped_perm)
  return gradedblockedrange(merged_lengths, merged_grades)
end

# Sort and merge by the grade of the blocks.
function blockmergesort(s::GradedBlockedUnitRange)
  grouped_perm = blockmergesortperm(s)
  return blockmerge(s, grouped_perm)
end

# Get the permutation for sorting, then group by common elements.
# groupsortperm([2, 1, 2, 3]) == [[2], [1, 3], [4]]
function blockmergesortperm(s::GradedBlockedUnitRange)
  return groupsortperm(s.grades)
end

function sub_axis(a::GradedBlockedUnitRange, blocks)
  return GradedBlockedUnitRange(sub_axis(a.blockedrange, blocks), map(b -> grade(a, b), blocks))
end
