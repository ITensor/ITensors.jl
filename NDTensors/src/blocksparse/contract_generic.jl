# A generic version that is used by both
# "threaded_folds" and "threaded"threads".
function contract_blockoffsets(
  alg::Algorithm,
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

  # Contraction plan element type
  T = Tuple{Block{N1},Block{N2},Block{NR}}

  # Thread-local collections of block contractions.
  # Could use:
  # ```julia
  # FLoops.@reduce(contraction_plans = append!(T[], [(block1, block2, blockR)]))
  # ```
  # as a simpler alternative but it is slower.

  contraction_plans = Vector{T}[T[] for _ in 1:nthreads()]

  #
  # Reserve some capacity
  # In theory the maximum is length(boffs1) * length(boffs2)
  # but in practice that is too much
  #for contraction_plan in contraction_plans
  #  sizehint!(contraction_plan, max(length(boffs1), length(boffs2)))
  #end
  #

  contract_blocks!(
    alg,
    contraction_plans,
    boffs1,
    boffs2,
    labels1_to_labels2,
    labels1_to_labelsR,
    labels2_to_labelsR,
    ValNR,
  )

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

# A generic version making use of `Folds.jl` which
# can take various Executor backends.
# Used for sequential and threaded contract functions.
function contract!(
  R::BlockSparseTensor,
  labelsR,
  tensor1::BlockSparseTensor,
  labelstensor1,
  tensor2::BlockSparseTensor,
  labelstensor2,
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
  _contract!(
    R,
    labelsR,
    tensor1,
    labelstensor1,
    tensor2,
    labelstensor2,
    grouped_contraction_plan,
    executor,
  )
  return R
end

# Function barrier to improve type stability,
# since `Folds`/`FLoops` is not type stable:
# https://discourse.julialang.org/t/type-instability-in-floop-reduction/68598
function _contract!(
  R::BlockSparseTensor,
  labelsR,
  tensor1::BlockSparseTensor,
  labelstensor1,
  tensor2::BlockSparseTensor,
  labelstensor2,
  grouped_contraction_plan,
  executor,
)
  Folds.foreach(grouped_contraction_plan.values, executor) do contraction_plan_group
    # Start by overwriting the block:
    # R .= α .* (tensor1 * tensor2)
    β = zero(eltype(R))
    for block_contraction in contraction_plan_group
      blocktensor1, blocktensor2, blockR = block_contraction

      # <fermions>:
      α = compute_alpha(
        eltype(R),
        labelsR,
        blockR,
        inds(R),
        labelstensor1,
        blocktensor1,
        inds(tensor1),
        labelstensor2,
        blocktensor2,
        inds(tensor2),
      )

      contract!(
        R[blockR],
        labelsR,
        tensor1[blocktensor1],
        labelstensor1,
        tensor2[blocktensor2],
        labelstensor2,
        α,
        β,
      )

      if iszero(β)
        # After the block has been overwritten,
        # add into it:
        # R .= α .* (tensor1 * tensor2) .+ β .* R
        β = one(eltype(R))
      end
    end
  end
  return nothing
end
