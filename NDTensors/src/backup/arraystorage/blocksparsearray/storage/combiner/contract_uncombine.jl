using .TypeParameterAccessors: unwrap_array_type
function contract_inds_uncombine(inds_src::Tuple, labels_src, inds_comb::Tuple, labels_comb)
  cpos_in_labels_comb = 1
  clabel = labels_comb[cpos_in_labels_comb]
  labels_uc = deleteat(labels_comb, cpos_in_labels_comb)
  labels_dest = labels_src
  cpos_in_labels_dest = findfirst(==(clabel), labels_dest)
  # Move combined index to first position
  perm = ntuple(identity, length(inds_src))
  if cpos_in_labels_dest != 1
    labels_dest_orig = labels_dest
    labels_dest = deleteat(labels_dest, cpos_in_labels_dest)
    labels_dest = insertafter(labels_dest, clabel, 0)
    cpos_in_labels_dest = 1
    perm = getperm(labels_dest, labels_dest_orig)
    inds_src = permute(inds_src, perm)
    labels_src = permute(labels_src, perm)
  end
  labels_dest = insertat(labels_dest, labels_uc, cpos_in_labels_dest)
  inds_dest = contract_inds(inds_comb, labels_comb, inds_src, labels_src, labels_dest)
  return inds_dest, labels_dest, perm, cpos_in_labels_dest
end

function contract_uncombine(
  a_src::BlockSparseArray, labels_src, a_comb::CombinerArray, labels_comb
)
  axes_dest, labels_dest, perm, cpos_in_labels_dest = contract_inds_uncombine(
    axes(a_src), labels_src, axes(a_comb), labels_comb
  )
  a_src = permutedims(a_src, perm)

  ## TODO: Add this back.
  ## # <fermions>:
  ## a_src = before_combiner_signs(
  ##   a_src,
  ##   labels_src,
  ##   axes(a_src),
  ##   a_comb,
  ##   labels_comb,
  ##   axes(a_comb),
  ##   labels_dest,
  ##   axes_dest,
  ## )

  a_dest = uncombine(
    a_src,
    labels_src,
    axes_dest,
    labels_dest,
    cpos_in_labels_dest,
    blockperm(a_comb),
    blockcomb(a_comb),
  )

  ## TODO: Add this back.
  ## # <fermions>:
  ## a_dest = after_combiner_signs(
  ##   a_dest,
  ##   labels_dest,
  ##   axes_dest,
  ##   a_comb,
  ##   labels_comb,
  ##   axes(a_comb),
  ## )

  return a_dest, labels_dest
end

function uncombine(
  a_src::BlockSparseArray,
  labels_src,
  axes_dest,
  labels_dest,
  combdim::Int,
  blockperm::Vector{Int},
  blockcomb::Vector{Int},
)
  a_dest = uncombine_output(
    a_src, labels_src, axes_dest, labels_dest, combdim, blockperm, blockcomb
  )
  invblockperm = invperm(blockperm)
  # This is needed for reshaping the block
  # TODO: It is already calculated in uncombine_output, use it from there
  labels_uncomb_perm = setdiff(labels_dest, labels_src)
  ind_uncomb_perm = ⊗(axes_dest[map(
    x -> findfirst(==(x), labels_dest), labels_uncomb_perm
  )]...)
  ind_uncomb = BlockArrays.blockedrange(
    length.(BlockArrays.blocks(ind_uncomb_perm)[blockperm])
  )
  # Same as axes(a_src) but with the blocks uncombined
  axes_uncomb = insertat(axes(a_src), ind_uncomb, combdim)
  axes_uncomb_perm = insertat(axes(a_src), ind_uncomb_perm, combdim)
  for b in nzblocks(a_src)
    a_src_b_tot = @view a_src[b]
    bs_uncomb = uncombine_block(b, combdim, blockcomb)
    offset = 0
    for i in 1:length(bs_uncomb)
      b_uncomb = bs_uncomb[i]
      b_uncomb_perm = perm_block(b_uncomb, combdim, invblockperm)
      b_uncomb_perm_reshape = reshape(b_uncomb_perm, axes_uncomb_perm, axes_dest)
      a_dest_b = @view a_dest[b_uncomb_perm_reshape]
      b_uncomb_in_combined_dim = b_uncomb_perm.n[combdim]
      start = offset + 1
      stop = offset + length(ind_uncomb_perm[BlockArrays.Block(b_uncomb_in_combined_dim)])
      subind = ntuple(
        i -> i == combdim ? range(start; stop=stop) : range(1; stop=size(a_src_b_tot)[i]),
        ndims(a_src),
      )
      offset = stop
      a_src_b = @view a_src_b_tot[subind...]

      # Alternative (but maybe slower):
      #copyto!(a_dest_b, a_src_b)

      if length(a_src_b) == 1
        # Call `cpu` to avoid allowscalar error on GPU.
        # TODO: a_desteplace with `@allowscalar`, requires adding
        # `GPUArraysCore.jl` as a dependency, or use `expose`.
        a_dest_b[] = cpu(a_src_b)[]
      else
        # TODO: Use `unspecify_parameters(unwrap_array_type(a_src))` intead of `Array`.
        a_dest_bₐ = convert(Array, a_dest_b)
        a_dest_bₐᵣ = reshape(a_dest_bₐ, size(a_src_b))
        copyto!(expose(a_dest_bₐᵣ), expose(a_src_b))
      end
    end
  end
  return a_dest
end

function uncombine_output(
  a_src::BlockSparseArray,
  labels_src,
  axes_dest,
  labels_dest,
  combdim::Int,
  blockperm::Vector{Int},
  blockcomb::Vector{Int},
)
  labels_uncomb_perm = setdiff(labels_dest, labels_src)
  ind_uncomb_perm = ⊗(axes_dest[map(
    x -> findfirst(==(x), labels_dest), labels_uncomb_perm
  )]...)
  axes_uncomb_perm = insertat(axes(a_src), ind_uncomb_perm, combdim)
  # Uncombine the blocks of a_src
  blocks_uncomb = uncombine_blocks(nzblocks(a_src), combdim, blockcomb)
  blocks_uncomb_perm = perm_blocks(blocks_uncomb, combdim, invperm(blockperm))

  # TODO: Should this be zero data instead of undef?
  T = eltype(a_src)
  N = length(axes_uncomb_perm)
  B = unwrap_array_type(a_src)
  a_uncomb_perm = BlockSparseArray{T,N,B}(undef, blocks_uncomb_perm, axes_uncomb_perm)
  return reshape(a_uncomb_perm, axes_dest)
end
