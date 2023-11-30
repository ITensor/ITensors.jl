using BlockArrays: BlockVector
using SplitApplyCombine: groupcount

function groupsorted(v)
  return groupcount(identity, v)
end

# Get the permutation for sorting, then group by common elements.
# groupsortperm([2, 1, 2, 3]) == [[2], [1, 3], [4]]
function groupsortperm(v)
  perm = sortperm(v)
  v_sorted = @view v[perm]
  group_lengths = groupsorted(v_sorted)
  return blocks(BlockVector(perm, collect(group_lengths)))
end
