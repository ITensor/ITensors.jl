function contract_uncombine!!(
  ::ArrayStorage,
  ::Any,
  tensor::BlockSparseArray,
  tensor_labels,
  combiner_tensor::CombinerTensor,
  combiner_tensor_labels,
)
  error("Not implemented")

  # Get the label marking the combined index
  # By convention the combined index is the first one
  # TODO: Consider storing the location of the combined
  # index in preperation for multiple combined indices
  # TODO: Use `combinedind_label(...)`, `uncombinedind_labels(...)`, etc.
  cpos_in_combiner_tensor_labels = 1
  clabel = combiner_tensor_labels[cpos_in_combiner_tensor_labels]
  c = combinedind(combiner_tensor)
  labels_uc = deleteat(combiner_tensor_labels, cpos_in_combiner_tensor_labels)
  output_tensor_labels = tensor_labels
  cpos_in_output_tensor_labels = findfirst(==(clabel), output_tensor_labels)
  # Move combined index to first position
  if cpos_in_output_tensor_labels != 1
    output_tensor_labels_orig = output_tensor_labels
    output_tensor_labels = deleteat(output_tensor_labels, cpos_in_output_tensor_labels)
    output_tensor_labels = insertafter(output_tensor_labels, clabel, 0)
    cpos_in_output_tensor_labels = 1
    perm = getperm(output_tensor_labels, output_tensor_labels_orig)
    tensor = permutedims(tensor, perm)
    tensor_labels = permute(tensor_labels, perm)
  end
  output_tensor_labels_uc = insertat(
    output_tensor_labels, labels_uc, cpos_in_output_tensor_labels
  )
  output_tensor_inds_uc = contract_inds(
    inds(combiner_tensor),
    combiner_tensor_labels,
    inds(tensor),
    tensor_labels,
    output_tensor_labels_uc,
  )

  # <fermions>:
  tensor = before_combiner_signs(
    tensor,
    tensor_labels,
    inds(tensor),
    combiner_tensor,
    combiner_tensor_labels,
    inds(combiner_tensor),
    output_tensor_labels_uc,
    output_tensor_inds_uc,
  )

  output_tensor = uncombine(
    tensor,
    tensor_labels,
    output_tensor_inds_uc,
    output_tensor_labels_uc,
    cpos_in_output_tensor_labels,
    blockperm(combiner_tensor),
    blockcomb(combiner_tensor),
  )

  # <fermions>:
  output_tensor = after_combiner_signs(
    output_tensor,
    output_tensor_labels_uc,
    output_tensor_inds_uc,
    combiner_tensor,
    combiner_tensor_labels,
    inds(combiner_tensor),
  )

  return output_tensor
end

function uncombine(
  T::BlockSparseArray{ElT,NT},
  T_labels,
  is,
  is_labels,
  combdim::Int,
  blockperm::Vector{Int},
  blockcomb::Vector{Int},
) where {ElT<:Number,NT}
  NR = length(is)
  R = uncombine_output(T, T_labels, is, is_labels, combdim, blockperm, blockcomb)
  invblockperm = invperm(blockperm)
  # This is needed for reshaping the block
  # TODO: It is already calculated in uncombine_output, use it from there
  labels_uncomb_perm = setdiff(is_labels, T_labels)
  ind_uncomb_perm = ⊗(is[map(x -> findfirst(==(x), is_labels), labels_uncomb_perm)]...)
  ind_uncomb = permuteblocks(ind_uncomb_perm, blockperm)
  # Same as inds(T) but with the blocks uncombined
  inds_uncomb = insertat(inds(T), ind_uncomb, combdim)
  inds_uncomb_perm = insertat(inds(T), ind_uncomb_perm, combdim)
  for b in nzblocks(storage(T))
    Tb_tot = @view storage(T)[b]
    dimsTb_tot = size(Tb_tot)
    bs_uncomb = uncombine_block(b, combdim, blockcomb)
    offset = 0
    for i in 1:length(bs_uncomb)
      b_uncomb = bs_uncomb[i]
      b_uncomb_perm = perm_block(b_uncomb, combdim, invblockperm)
      b_uncomb_perm_reshape = reshape(b_uncomb_perm, inds_uncomb_perm, is)
      Rb = @view storage(R)[b_uncomb_perm_reshape]
      b_uncomb_in_combined_dim = b_uncomb_perm.n[combdim]
      start = offset + 1
      stop = offset + blockdim(ind_uncomb_perm, b_uncomb_in_combined_dim)
      subind = ntuple(
        i -> i == combdim ? range(start; stop=stop) : range(1; stop=dimsTb_tot[i]), NT
      )
      offset = stop
      Tb = @view Tb_tot[subind...]

      # Alternative (but maybe slower):
      #copyto!(Rb,Tb)

      if length(Tb) == 1
        # Call `cpu` to avoid allowscalar error on GPU.
        # TODO: Replace with `@allowscalar`, requires adding
        # `GPUArraysCore.jl` as a dependency.
        Rb[] = cpu(Tb)[]
      else
        # XXX: this used to be:
        # Rbₐᵣ = ReshapedArray(parent(Rbₐ), size(Tb), ())
        # however that doesn't work with subarrays
        Rbₐ = convert(Array, Rb)
        ## Rbₐᵣ = ReshapedArray(Rbₐ, size(Tb), ())
        Rbₐᵣ = reshape(Rbₐ, size(Tb))
        ## @strided Rbₐᵣ .= Tb
        copyto!(expose(Rbₐᵣ), expose(Tb))
      end
    end
  end
  return R
end

function uncombine_output(
  T::BlockSparseArray{ElT,N},
  T_labels,
  is,
  is_labels,
  combdim::Int,
  blockperm::Vector{Int},
  blockcomb::Vector{Int},
) where {ElT<:Number,N}
  labels_uncomb_perm = setdiff(is_labels, T_labels)
  ind_uncomb_perm = ⊗(is[map(x -> findfirst(==(x), is_labels), labels_uncomb_perm)]...)
  inds_uncomb_perm = insertat(inds(T), ind_uncomb_perm, combdim)
  # Uncombine the blocks of T
  blocks_uncomb = uncombine_blocks(nzblocks(T), combdim, blockcomb)
  blocks_uncomb_perm = perm_blocks(blocks_uncomb, combdim, invperm(blockperm))
  blockinds_uncomb_perm = map(i -> [blockdim(i, b) for b in 1:nblocks(i)], inds_uncomb_perm)
  ## boffs_uncomb_perm, nnz_uncomb_perm = blockoffsets(blocks_uncomb_perm, inds_uncomb_perm)
  ## T_uncomb_perm = tensor(
  ##   BlockSparse(unwrap_type(T), boffs_uncomb_perm, nnz_uncomb_perm), inds_uncomb_perm
  ## )
  # TODO: Should this be zero data instead of undef?
  a_uncomb_perm = BlockSparseArray{ElT,length(inds_uncomb_perm),unwrap_type(T)}(
    undef, blocks_uncomb_perm, blockinds_uncomb_perm
  )
  blockinds = map(i -> [blockdim(i, b) for b in 1:nblocks(i)], is)
  R = tensor(reshape(a_uncomb_perm, blockinds), is)
  return R
end
