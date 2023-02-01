## Deprecated versions, for testing

# Old version
# TODO: DELETE
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

# Old version
# TODO: DELETE
function contract!(
  ::Algorithm"threaded_deprecated",
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
