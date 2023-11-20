struct BipartitionedPermutation{P1,P2}
  partition1::P1
  partition2::P2
end

function Base.getindex(biperm::BipartitionedPermutation, i)
  if i == 1
    return biperm.partition1
  elseif i == 2
    return biperm.partition2
  end
  return error("Only 2 partitions")
end

function flatten(biperm::BipartitionedPermutation)
  return (biperm[1]..., biperm[2]...)
end

# Bipartition a vector according to the
# bipartitioned permutation.
function bipartition(v, biperm::BipartitionedPermutation)
  # TODO: Use `TupleTools.getindices`.
  v1 = map(i -> v[i], biperm[1])
  v2 = map(i -> v[i], biperm[2])
  return v1, v2
end
