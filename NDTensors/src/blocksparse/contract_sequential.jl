function contract_blockoffsets(
  ::Algorithm"sequential",
  boffs1::BlockOffsets,
  inds1,
  labels1,
  boffs2::BlockOffsets,
  inds2,
  labels2,
  indsR,
  labelsR,
)
  N1 = ndims(boffs1)
  N2 = ndims(boffs2)
  NR = length(labelsR)
  ValNR = ValLength(labelsR)
  labels1_to_labels2, labels1_to_labelsR, labels2_to_labelsR = contract_labels(
    labels1, labels2, labelsR
  )
  blockoffsetsR = BlockOffsets{NR}()
  nnzR = 0
  contraction_plan = Tuple{Block{N1},Block{N2},Block{NR}}[]
  # Reserve some capacity
  # In theory the maximum is length(boffs1) * length(boffs2)
  # but in practice that is too much
  sizehint!(contraction_plan, max(length(boffs1), length(boffs2)))
  for block1 in keys(boffs1)
    for block2 in keys(boffs2)
      if are_blocks_contracted(block1, block2, labels1_to_labels2)
        blockR = contract_blocks(
          block1, labels1_to_labelsR, block2, labels2_to_labelsR, ValNR
        )
        push!(contraction_plan, (block1, block2, blockR))
        if !isassigned(blockoffsetsR, blockR)
          insert!(blockoffsetsR, blockR, nnzR)
          nnzR += blockdim(indsR, blockR)
        end
      end
    end
  end
  return blockoffsetsR, contraction_plan
end

function contract!(
  ::Algorithm"sequential",
  R::BlockSparseTensor,
  labelsR,
  tensor1::BlockSparseTensor,
  labelstensor1,
  tensor2::BlockSparseTensor,
  labelstensor2,
  contraction_plan,
)
  executor = SequentialEx()
  return contract!(
    R, labelsR, tensor1, labelstensor1, tensor2, labelstensor2, contraction_plan, executor
  )
end

###########################################################################
# Old version
# TODO: DELETE, keeping around for now for testing/benchmarking.
function contract!(
  ::Algorithm"sequential_deprecated",
  R::BlockSparseTensor{ElR,NR},
  labelsR,
  T1::BlockSparseTensor{ElT1,N1},
  labelsT1,
  T2::BlockSparseTensor{ElT2,N2},
  labelsT2,
  contraction_plan,
) where {ElR,ElT1,ElT2,N1,N2,NR}
  if isempty(contraction_plan)
    return R
  end
  if using_threaded_blocksparse() && nthreads() > 1
    _contract_threaded_deprecated!(R, labelsR, T1, labelsT1, T2, labelsT2, contraction_plan)
    return R
  end
  already_written_to = Dict{Block{NR},Bool}()
  indsR = inds(R)
  indsT1 = inds(T1)
  indsT2 = inds(T2)
  # In R .= α .* (T1 * T2) .+ β .* R
  for (block1, block2, blockR) in contraction_plan

    #<fermions>
    α = compute_alpha(
      ElR, labelsR, blockR, indsR, labelsT1, block1, indsT1, labelsT2, block2, indsT2
    )

    T1block = T1[block1]
    T2block = T2[block2]
    Rblock = R[blockR]
    β = one(ElR)
    if !haskey(already_written_to, blockR)
      already_written_to[blockR] = true
      # Overwrite the block of R
      β = zero(ElR)
    end
    contract!(Rblock, labelsR, T1block, labelsT1, T2block, labelsT2, α, β)
  end
  return R
end
