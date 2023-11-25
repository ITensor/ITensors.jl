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
