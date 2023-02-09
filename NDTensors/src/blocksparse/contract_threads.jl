# TODO: This seems to be faster than the newer version using `Folds.jl`
# in `contract_folds.jl`, investigate why.
function contract_blockoffsets(
  alg::Algorithm"threaded_threads",
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
  contraction_plans = Vector{Tuple{Block{N1},Block{N2},Block{NR}}}[
    Tuple{Block{N1},Block{N2},Block{NR}}[] for _ in 1:nthreads()
  ]

  #
  # Reserve some capacity
  # In theory the maximum is length(boffs1) * length(boffs2)
  # but in practice that is too much
  #for contraction_plan in contraction_plans
  #  sizehint!(contraction_plan, max(length(boffs1), length(boffs2)))
  #end
  #

  blocks1 = keys(boffs1)
  blocks2 = keys(boffs2)
  if length(blocks1) > length(blocks2)
    @sync for blocks1_partition in
              Iterators.partition(blocks1, max(1, length(blocks1) ÷ nthreads()))
      @spawn for block1 in blocks1_partition
        for block2 in blocks2
          if are_blocks_contracted(block1, block2, labels1_to_labels2)
            blockR = contract_blocks(
              block1, labels1_to_labelsR, block2, labels2_to_labelsR, ValNR
            )
            push!(contraction_plans[threadid()], (block1, block2, blockR))
          end
        end
      end
    end
  else
    @sync for blocks2_partition in
              Iterators.partition(blocks2, max(1, length(blocks2) ÷ nthreads()))
      @spawn for block2 in blocks2_partition
        for block1 in blocks1
          if are_blocks_contracted(block1, block2, labels1_to_labels2)
            blockR = contract_blocks(
              block1, labels1_to_labelsR, block2, labels2_to_labelsR, ValNR
            )
            push!(contraction_plans[threadid()], (block1, block2, blockR))
          end
        end
      end
    end
  end

  contraction_plan = reduce(vcat, contraction_plans)
  blockoffsetsR = BlockOffsets{NR}()
  nnzR = 0
  for (_, _, blockR) in contraction_plan
    if !isassigned(blockoffsetsR, blockR)
      insert!(blockoffsetsR, blockR, nnzR)
      nnzR += blockdim(indsR, blockR)
    end
  end

  return blockoffsetsR, contraction_plan
end

###########################################################################
# Old version
# TODO: DELETE, keeping around for testing/benchmarking.
function contract!(
  ::Algorithm"threaded_threads",
  R::BlockSparseTensor{ElR,NR},
  labelsR,
  T1::BlockSparseTensor{ElT1,N1},
  labelsT1,
  T2::BlockSparseTensor{ElT2,N2},
  labelsT2,
  contraction_plan,
) where {ElR,ElT1,ElT2,N1,N2,NR}
  # Sort the contraction plan by the output blocks
  # This is to help determine which output blocks are the result
  # of multiple contractions
  sort!(contraction_plan; by=last)

  # Ranges of contractions to the same block
  repeats = Vector{UnitRange{Int}}(undef, nnzblocks(R))
  ncontracted = 1
  posR = last(contraction_plan[1])
  posR_unique = posR
  for n in 1:(nnzblocks(R) - 1)
    start = ncontracted
    while posR == posR_unique
      ncontracted += 1
      posR = last(contraction_plan[ncontracted])
    end
    posR_unique = posR
    repeats[n] = start:(ncontracted - 1)
  end
  repeats[end] = ncontracted:length(contraction_plan)

  contraction_plan_blocks = Vector{Tuple{Tensor,Tensor,Tensor}}(
    undef, length(contraction_plan)
  )
  for ncontracted in 1:length(contraction_plan)
    block1, block2, blockR = contraction_plan[ncontracted]
    T1block = T1[block1]
    T2block = T2[block2]
    Rblock = R[blockR]
    contraction_plan_blocks[ncontracted] = (T1block, T2block, Rblock)
  end

  indsR = inds(R)
  indsT1 = inds(T1)
  indsT2 = inds(T2)

  α = one(ElR)
  @sync for repeats_partition in
            Iterators.partition(repeats, max(1, length(repeats) ÷ nthreads()))
    @spawn for ncontracted_range in repeats_partition
      # Overwrite the block since it hasn't been written to
      # R .= α .* (T1 * T2)
      β = zero(ElR)
      for ncontracted in ncontracted_range
        blockT1, blockT2, blockR = contraction_plan_blocks[ncontracted]
        # R .= α .* (T1 * T2) .+ β .* R

        # <fermions>:
        α = compute_alpha(
          ElR, labelsR, blockR, indsR, labelsT1, blockT1, indsT1, labelsT2, blockT2, indsT2
        )

        contract!(blockR, labelsR, blockT1, labelsT1, blockT2, labelsT2, α, β)
        # Now keep adding to the block, since it has
        # been written to
        # R .= α .* (T1 * T2) .+ R
        β = one(ElR)
      end
    end
  end
  return R
end
