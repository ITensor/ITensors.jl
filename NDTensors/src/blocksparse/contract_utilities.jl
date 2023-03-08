# <fermions>
function compute_alpha(
  ElR,
  labelsR,
  blockR,
  indsR,
  labelstensor1,
  blocktensor1,
  indstensor1,
  labelstensor2,
  blocktensor2,
  indstensor2,
)
  return one(ElR)
end

function maybe_contract_blocks!(
  contraction_plan,
  block1,
  block2,
  labels1_to_labels2,
  labels1_to_labelsR,
  labels2_to_labelsR,
  ValNR,
)
  if are_blocks_contracted(block1, block2, labels1_to_labels2)
    blockR = contract_blocks(block1, labels1_to_labelsR, block2, labels2_to_labelsR, ValNR)
    push!(contraction_plan, (block1, block2, blockR))
  end
  return nothing
end

function contract_labels(labels1, labels2, labelsR)
  labels1_to_labels2 = find_matching_positions(labels1, labels2)
  labels1_to_labelsR = find_matching_positions(labels1, labelsR)
  labels2_to_labelsR = find_matching_positions(labels2, labelsR)
  return labels1_to_labels2, labels1_to_labelsR, labels2_to_labelsR
end

"""
    find_matching_positions(t1,t2) -> t1_to_t2

In a tuple of length(t1), store the positions in t2
where the element of t1 is found. Otherwise, store 0
to indicate that the element of t1 is not in t2.

For example, for all t1[pos1] == t2[pos2], t1_to_t2[pos1] == pos2,
otherwise t1_to_t2[pos1] == 0.
"""
function find_matching_positions(t1, t2)
  t1_to_t2 = @MVector zeros(Int, length(t1))
  for pos1 in 1:length(t1)
    for pos2 in 1:length(t2)
      if t1[pos1] == t2[pos2]
        t1_to_t2[pos1] = pos2
      end
    end
  end
  return Tuple(t1_to_t2)
end

function are_blocks_contracted(block1::Block, block2::Block, labels1_to_labels2::Tuple)
  t1 = Tuple(block1)
  t2 = Tuple(block2)
  for i1 in 1:length(block1)
    i2 = @inbounds labels1_to_labels2[i1]
    if i2 > 0
      # This dimension is contracted
      if @inbounds t1[i1] != @inbounds t2[i2]
        return false
      end
    end
  end
  return true
end

function contract_blocks(
  block1::Block, labels1_to_labelsR, block2::Block, labels2_to_labelsR, ::Val{NR}
) where {NR}
  blockR = ntuple(_ -> UInt(0), Val(NR))
  t1 = Tuple(block1)
  t2 = Tuple(block2)
  for i1 in 1:length(block1)
    iR = @inbounds labels1_to_labelsR[i1]
    if iR > 0
      blockR = @inbounds setindex(blockR, t1[i1], iR)
    end
  end
  for i2 in 1:length(block2)
    iR = @inbounds labels2_to_labelsR[i2]
    if iR > 0
      blockR = @inbounds setindex(blockR, t2[i2], iR)
    end
  end
  return Block{NR}(blockR)
end
