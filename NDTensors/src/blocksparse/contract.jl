# <fermions>
function compute_alpha(
  ElR, labelsR, blockR, indsR, labelsT1, blockT1, indsT1, labelsT2, blockT2, indsT2
)
  return one(ElR)
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

# TODO: complete this function: determine the output blocks from the input blocks
# Also, save the contraction list (which block-offsets contract with which),
# may not be generic with other contraction functions!
function contraction_output(T1::BlockSparseTensor, T2::BlockSparseTensor, indsR)
  TensorR = contraction_output_type(typeof(T1), typeof(T2), typeof(indsR))
  return similar(TensorR, blockoffsetsR, indsR)
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

function contract_blockoffsets(args...)
  if using_threaded_blocksparse() && nthreads() > 1
    return _contract_blockoffsets_threaded(args...)
  end
  return _contract_blockoffsets(args...)
end

function _contract_blockoffsets(
  boffs1::BlockOffsets, inds1, labels1, boffs2::BlockOffsets, inds2, labels2, indsR, labelsR
)
  ValNR = ValLength(labelsR)
  labels1_to_labels2, labels1_to_labelsR, labels2_to_labelsR = contract_labels(
    labels1, labels2, labelsR
  )
  blockoffsetsR = BlockOffsets{length(labelsR)}()
  nnzR = 0
  contraction_plan = Tuple{
    Block{ndims(boffs1)},Block{ndims(boffs2)},Block{length(labelsR)}
  }[]
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

function _contract_blockoffsets_threaded(
  boffs1::BlockOffsets,
  inds1,
  labels1,
  boffs2::BlockOffsets,
  inds2,
  labels2,
  indsR,
  labelsR,
)
  ValNR = ValLength(labelsR)
  labels1_to_labels2, labels1_to_labelsR, labels2_to_labelsR = contract_labels(
    labels1, labels2, labelsR
  )

  T = Tuple{blocktype(boffs1),blocktype(boffs2),Block{length(labelsR)}}

  # Thread-local collections of block contractions.
  # Could use:
  # ```julia
  # FLoops.@reduce(contraction_plans = append!(T[], [(block1, block2, blockR)]))
  # ```
  # as a simpler alternative but it is too slow.
  contraction_plans = Vector{T}[Vector{T}() for _ in 1:nthreads()]

  # # Reserve some capacity.
  # # In theory the maximum is `length(boffs1) * length(boffs2)`
  # # but in practice that is too much.
  # # This didn't seem to help the performance.
  # for contraction_plan in contraction_plans
  #   sizehint!(contraction_plan, max(length(boffs1), length(boffs2)))
  # End

  if nnzblocks(boffs1) > nnzblocks(boffs2)
    @floop ThreadedEx() for block1 in eachnzblock(boffs1).values
      for block2 in eachnzblock(boffs2)
        if are_blocks_contracted(block1, block2, labels1_to_labels2)
          blockR = contract_blocks(
            block1, labels1_to_labelsR, block2, labels2_to_labelsR, ValNR
          )
          push!(contraction_plans[threadid()], (block1, block2, blockR))
        end
      end
    end
  else
    @floop ThreadedEx() for block2 in eachnzblock(boffs2).values
      for block1 in eachnzblock(boffs1)
        if are_blocks_contracted(block1, block2, labels1_to_labels2)
          blockR = contract_blocks(
            block1, labels1_to_labelsR, block2, labels2_to_labelsR, ValNR
          )
          push!(contraction_plans[threadid()], (block1, block2, blockR))
        end
      end
    end
  end

  # Collect the results across threads
  contraction_plan = reduce(vcat, contraction_plans)

  blockoffsetsR = BlockOffsets{length(labelsR)}()
  nnzR = 0
  for (_, _, blockR) in contraction_plan
    if !isassigned(blockoffsetsR, blockR)
      insert!(blockoffsetsR, blockR, nnzR)
      nnzR += blockdim(indsR, blockR)
    end
  end

  return blockoffsetsR, contraction_plan
end

function contraction_output(
  T1::BlockSparseTensor, labelsT1, T2::BlockSparseTensor, labelsT2, labelsR
)
  indsR = contract_inds(inds(T1), labelsT1, inds(T2), labelsT2, labelsR)
  TensorR = contraction_output_type(typeof(T1), typeof(T2), typeof(indsR))
  blockoffsetsR, contraction_plan = contract_blockoffsets(
    blockoffsets(T1),
    inds(T1),
    labelsT1,
    blockoffsets(T2),
    inds(T2),
    labelsT2,
    indsR,
    labelsR,
  )
  R = similar(TensorR, blockoffsetsR, indsR)
  return R, contraction_plan
end

function contract(
  T1::BlockSparseTensor,
  labelsT1,
  T2::BlockSparseTensor,
  labelsT2,
  labelsR=contract_labels(labelsT1, labelsT2),
)
  R, contraction_plan = contraction_output(T1, labelsT1, T2, labelsT2, labelsR)
  R = contract!(R, labelsR, T1, labelsT1, T2, labelsT2, contraction_plan)
  return R
end

function contract!(
  R::BlockSparseTensor,
  labelsR,
  T1::BlockSparseTensor,
  labelsT1,
  T2::BlockSparseTensor,
  labelsT2,
  contraction_plan,
)
  if isempty(contraction_plan)
    return R
  end
  executor = SequentialEx()
  if using_threaded_blocksparse() && nthreads() > 1
    executor = ThreadedEx()
  end
  return _contract!(R, labelsR, T1, labelsT1, T2, labelsT2, contraction_plan, executor)
end

function _contract!(
  R::BlockSparseTensor,
  labelsR,
  T1::BlockSparseTensor,
  labelsT1,
  T2::BlockSparseTensor,
  labelsT2,
  contraction_plan,
  executor,
)
  # Group the contraction plan by the output block,
  # since the sets of contractions into the same block
  # must be performed sequentially to reduce over those
  # sets of contractions properly (and avoid race conditions).
  # Same as:
  # ```julia
  # grouped_contraction_plan = group(last, contraction_plan)
  # ```
  # but more efficient since we know the groups/keys already,
  # since they are the nonzero blocks of the output tensor `R`.
  grouped_contraction_plan = map(_ -> empty(contraction_plan), eachnzblock(R))
  for block_contraction in contraction_plan
    push!(grouped_contraction_plan[last(block_contraction)], block_contraction)
  end

  @floop executor for contraction_plan_group in grouped_contraction_plan.values
    # Start by overwriting the block:
    # R .= α .* (T1 * T2)
    β = zero(eltype(R))
    for block_contraction in contraction_plan_group
      blockT1, blockT2, blockR = block_contraction

      # <fermions>:
      α = compute_alpha(
        eltype(R),
        labelsR,
        blockR,
        inds(R),
        labelsT1,
        blockT1,
        inds(T1),
        labelsT2,
        blockT2,
        inds(T2),
      )

      contract!(R[blockR], labelsR, T1[blockT1], labelsT1, T2[blockT2], labelsT2, α, β)

      if iszero(β)
        # After the block has been overwritten,
        # add into it:
        # R .= α .* (T1 * T2) .+ β .* R
        β = one(eltype(R))
      end
    end
  end
  return R
end

# Old version
# TODO: DELETE
function contract_deprecated!(
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
function _contract_threaded_deprecated!(
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
