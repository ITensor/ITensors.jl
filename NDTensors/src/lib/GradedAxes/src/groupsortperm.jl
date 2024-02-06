using BlockArrays: Block, BlockVector
using SplitApplyCombine: groupcount

invblockperm(a::Vector{<:Block{1}}) = Block.(invperm(Int.(a)))

# Get the permutation for sorting, then group by common elements.
# groupsortperm([2, 1, 2, 3]) == [[2], [1, 3], [4]]
function groupsortperm(v; kwargs...)
  perm = sortperm(v; kwargs...)
  v_sorted = @view v[perm]
  group_lengths = collect(groupcount(identity, v_sorted))
  return BlockVector(perm, group_lengths)
end
