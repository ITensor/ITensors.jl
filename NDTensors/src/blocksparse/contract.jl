# <fermions>
function compute_alpha(
  ElR, labelsR, blockR, indsR, labelsT1, blockT1, indsT1, labelsT2, blockT2, indsT2
)
  return one(ElR)
end

## Utilities

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

## Determine the contraction output and block contractions

# TODO: complete this function: determine the output blocks from the input blocks
# Also, save the contraction list (which block-offsets contract with which),
# may not be generic with other contraction functions!
function contraction_output(T1::BlockSparseTensor, T2::BlockSparseTensor, indsR)
  TensorR = contraction_output_type(typeof(T1), typeof(T2), typeof(indsR))
  return similar(TensorR, blockoffsetsR, indsR)
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
  alg = Algorithm"sequential"()
  if using_threaded_blocksparse() && nthreads() > 1
    alg = Algorithm"threaded"()
  end
  return contract_blockoffsets(alg, args...)
end

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

function contract_blockoffsets(
  alg::Algorithm"threaded",
  boffs1::BlockOffsets,
  inds1,
  labels1,
  boffs2::BlockOffsets,
  inds2,
  labels2,
  indsR,
  labelsR,
)
  NR = length(labelsR)
  ValNR = ValLength(labelsR)
  labels1_to_labels2, labels1_to_labelsR, labels2_to_labelsR = contract_labels(
    labels1, labels2, labelsR
  )

  T = Tuple{blocktype(boffs1),blocktype(boffs2),Block{NR}}

  # Thread-local collections of block contractions.
  # Could use:
  # ```julia
  # FLoops.@reduce(contraction_plans = append!(T[], [(block1, block2, blockR)]))
  # ```
  # as a simpler alternative but it is slower.
  contraction_plans = Vector{T}[Vector{T}() for _ in 1:nthreads()]

  # # Reserve some capacity.
  # # In theory the maximum is `length(boffs1) * length(boffs2)`
  # # but in practice that is too much.
  # # This didn't seem to help the performance.
  # for contraction_plan in contraction_plans
  #   sizehint!(contraction_plan, max(length(boffs1), length(boffs2)))
  # End

  _contract_blocks!(
    alg,
    contraction_plans,
    boffs1,
    boffs2,
    labels1_to_labels2,
    labels1_to_labelsR,
    labels2_to_labelsR,
    ValNR,
  )

  # Collect the results across threads
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

# Function barrier to improve type stability,
# since `Folds`/`FLoops` is not type stable:
# https://discourse.julialang.org/t/type-instability-in-floop-reduction/68598
function _contract_blocks!(
  alg::Algorithm"threaded",
  contraction_plans,
  boffs1,
  boffs2,
  labels1_to_labels2,
  labels1_to_labelsR,
  labels2_to_labelsR,
  ValNR,
)
  if nnzblocks(boffs1) > nnzblocks(boffs2)
    Folds.foreach(eachnzblock(boffs1).values, ThreadedEx()) do block1
      for block2 in eachnzblock(boffs2)
        _maybe_contract_blocks!(
          alg,
          contraction_plans,
          block1,
          block2,
          labels1_to_labels2,
          labels1_to_labelsR,
          labels2_to_labelsR,
          ValNR,
        )
      end
    end
  else
    Folds.foreach(eachnzblock(boffs2).values, ThreadedEx()) do block2
      for block1 in eachnzblock(boffs1)
        _maybe_contract_blocks!(
          alg,
          contraction_plans,
          block1,
          block2,
          labels1_to_labels2,
          labels1_to_labelsR,
          labels2_to_labelsR,
          ValNR,
        )
      end
    end
  end
  return nothing
end

function _maybe_contract_blocks!(
  ::Algorithm"threaded",
  contraction_plans,
  block1,
  block2,
  labels1_to_labels2,
  labels1_to_labelsR,
  labels2_to_labelsR,
  ValNR,
)
  if are_blocks_contracted(block1, block2, labels1_to_labels2)
    blockR = contract_blocks(block1, labels1_to_labelsR, block2, labels2_to_labelsR, ValNR)
    block_contraction = (block1, block2, blockR)
    push!(contraction_plans[threadid()], block_contraction)
  end
  return nothing
end

## Perform the contraction

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
  alg = Algorithm"sequential"()
  if isempty(contraction_plan)
    alg = Algorithm"no_op"()
  elseif using_threaded_blocksparse() && nthreads() > 1
    alg = Algorithm"threaded"()
  end
  return contract!(alg, R, labelsR, T1, labelsT1, T2, labelsT2, contraction_plan)
end

function contract!(
  ::Algorithm"sequential",
  R::BlockSparseTensor,
  labelsR,
  T1::BlockSparseTensor,
  labelsT1,
  T2::BlockSparseTensor,
  labelsT2,
  contraction_plan,
)
  executor = SequentialEx()
  return contract!(R, labelsR, T1, labelsT1, T2, labelsT2, contraction_plan, executor)
end

function contract!(
  ::Algorithm"threaded",
  R::BlockSparseTensor,
  labelsR,
  T1::BlockSparseTensor,
  labelsT1,
  T2::BlockSparseTensor,
  labelsT2,
  contraction_plan,
)
  executor = ThreadedEx()
  return contract!(R, labelsR, T1, labelsT1, T2, labelsT2, contraction_plan, executor)
end

function contract!(
  ::Algorithm"no_op",
  R::BlockSparseTensor,
  labelsR,
  T1::BlockSparseTensor,
  labelsT1,
  T2::BlockSparseTensor,
  labelsT2,
  contraction_plan,
)
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
  _contract!(R, labelsR, T1, labelsT1, T2, labelsT2, grouped_contraction_plan, executor)
  return R
end

# Function barrier to improve type stability,
# since `Folds`/`FLoops` is not type stable:
# https://discourse.julialang.org/t/type-instability-in-floop-reduction/68598
function _contract!(
  R::BlockSparseTensor,
  labelsR,
  T1::BlockSparseTensor,
  labelsT1,
  T2::BlockSparseTensor,
  labelsT2,
  grouped_contraction_plan,
  executor,
)
  Folds.foreach(grouped_contraction_plan.values, executor) do contraction_plan_group
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
  return nothing
end
