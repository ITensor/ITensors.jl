function _permute_combdims(combdims::NTuple{NC,Int}, perm::NTuple{NP,Int}) where {NC,NP}
  res = MVector{NC,Int}(undef)
  iperm = invperm(perm)
  for i in 1:NC
    res[i] = iperm[combdims[i]]
  end
  return Tuple(res)
end

#
# These are functions to help with combining and uncombining
#

# Note that combdims is expected to be contiguous and ordered
# smallest to largest
function combine_dims(blocks::Vector{Block{N}}, inds, combdims::NTuple{NC,Int}) where {N,NC}
  nblcks = nblocks(inds, combdims)
  blocks_comb = Vector{Block{N - NC + 1}}(undef, length(blocks))
  for (i, block) in enumerate(blocks)
    blocks_comb[i] = combine_dims(block, inds, combdims)
  end
  return blocks_comb
end

function combine_dims(block::Block, inds, combdims::NTuple{NC,Int}) where {NC}
  nblcks = nblocks(inds, combdims)
  slice = getindices(block, combdims)
  slice_comb = LinearIndices(nblcks)[slice...]
  block_comb = deleteat(block, combdims)
  block_comb = insertafter(block_comb, tuple(slice_comb), minimum(combdims) - 1)
  return block_comb
end

# In the dimension dim, permute the blocks
function perm_blocks(blocks::Blocks{N}, dim::Int, perm) where {N}
  blocks_perm = Blocks{N}(undef, nnzblocks(blocks))
  iperm = invperm(perm)
  for (i, block) in enumerate(blocks)
    blocks_perm[i] = setindex(block, iperm[block[dim]], dim)
  end
  return blocks_perm
end

# In the dimension dim, permute the block
function perm_block(block::Block, dim::Int, perm)
  iperm = invperm(perm)
  return setindex(block, iperm[block[dim]], dim)
end

# In the dimension dim, combine the specified blocks
function combine_blocks(blocks::Blocks, dim::Int, blockcomb::Vector{Int})
  blocks_comb = copy(blocks)
  nnz_comb = nnzblocks(blocks)
  for (i, block) in enumerate(blocks)
    dimval = block[dim]
    blocks_comb[i] = setindex(block, blockcomb[dimval], dim)
  end
  unique!(blocks_comb)
  return blocks_comb
end

function permutedims_combine_output(
  T::BlockSparseTensor{ElT,N},
  is,
  perm::NTuple{N,Int},
  combdims::NTuple{NC,Int},
  blockperm::Vector{Int},
  blockcomb::Vector{Int},
) where {ElT,N,NC}
  # Permute the indices
  indsT = inds(T)
  inds_perm = permute(indsT, perm)

  # Now that the indices are permuted, compute
  # which indices are now combined
  combdims_perm = sort(_permute_combdims(combdims, perm))

  # Permute the nonzero blocks (dimension-wise)
  blocks = nzblocks(T)
  blocks_perm = permutedims(blocks, perm)

  # Combine the nonzero blocks (dimension-wise)
  blocks_perm_comb = combine_dims(blocks_perm, inds_perm, combdims_perm)

  # Permute the blocks (within the newly combined dimension)
  comb_ind_loc = minimum(combdims_perm)
  blocks_perm_comb = perm_blocks(blocks_perm_comb, comb_ind_loc, blockperm)
  blocks_perm_comb = sort(blocks_perm_comb; lt=isblockless)

  # Combine the blocks (within the newly combined and permuted dimension)
  blocks_perm_comb = combine_blocks(blocks_perm_comb, comb_ind_loc, blockcomb)

  return BlockSparseTensor(ElT, blocks_perm_comb, is)
end

function permutedims_combine(
  T::BlockSparseTensor{ElT,N},
  is,
  perm::NTuple{N,Int},
  combdims::NTuple{NC,Int},
  blockperm::Vector{Int},
  blockcomb::Vector{Int},
) where {ElT,N,NC}
  R = permutedims_combine_output(T, is, perm, combdims, blockperm, blockcomb)

  # Permute the indices
  inds_perm = permute(inds(T), perm)

  # Now that the indices are permuted, compute
  # which indices are now combined
  combdims_perm = sort(_permute_combdims(combdims, perm))
  comb_ind_loc = minimum(combdims_perm)

  # Determine the new index before combining
  inds_to_combine = getindices(inds_perm, combdims_perm)
  ind_comb = ⊗(inds_to_combine...)
  ind_comb = permuteblocks(ind_comb, blockperm)

  for bof in pairs(blockoffsets(T))
    Tb = blockview(T, bof)
    b = nzblock(bof)
    b_perm = permute(b, perm)
    b_perm_comb = combine_dims(b_perm, inds_perm, combdims_perm)
    b_perm_comb = perm_block(b_perm_comb, comb_ind_loc, blockperm)
    b_in_combined_dim = b_perm_comb[comb_ind_loc]
    new_b_in_combined_dim = blockcomb[b_in_combined_dim]
    offset = 0
    pos_in_new_combined_block = 1
    while b_in_combined_dim - pos_in_new_combined_block > 0 &&
      blockcomb[b_in_combined_dim - pos_in_new_combined_block] == new_b_in_combined_dim
      offset += blockdim(ind_comb, b_in_combined_dim - pos_in_new_combined_block)
      pos_in_new_combined_block += 1
    end
    b_new = setindex(b_perm_comb, new_b_in_combined_dim, comb_ind_loc)

    Rb_total = blockview(R, b_new)
    dimsRb_tot = dims(Rb_total)
    subind = ntuple(
      i -> if i == comb_ind_loc
        range(1 + offset; stop=offset + blockdim(ind_comb, b_in_combined_dim))
      else
        range(1; stop=dimsRb_tot[i])
      end,
      N - NC + 1,
    )
    Rb = @view array(Rb_total)[subind...]

    # XXX Are these equivalent?
    #Tb_perm = permutedims(Tb,perm)
    #copyto!(Rb,Tb_perm)

    # XXX Not sure what this was for
    Rb = reshape(Rb, permute(dims(Tb), perm))
    Tbₐ = convert(Array, Tb)
    @strided Rb .= permutedims(Tbₐ, perm)
  end

  return R
end

# TODO: optimize by avoiding findfirst
function _number_uncombined(blockval::Integer, blockcomb::Vector)
  if blockval == blockcomb[end]
    return length(blockcomb) - findfirst(==(blockval), blockcomb) + 1
  end
  return findfirst(==(blockval + 1), blockcomb) - findfirst(==(blockval), blockcomb)
end

# TODO: optimize by avoiding findfirst
function _number_uncombined_shift(blockval::Integer, blockcomb::Vector)
  if blockval == 1
    return 0
  end
  ncomb_shift = 0
  for i in 1:(blockval - 1)
    ncomb_shift += findfirst(==(i + 1), blockcomb) - findfirst(==(i), blockcomb) - 1
  end
  return ncomb_shift
end

# Uncombine the blocks along the dimension dim
# according to the pattern in blockcomb (for example, blockcomb
# is [1,2,2,3] and dim = 2, so the blocks (1,2),(2,3) get
# split into (1,2),(1,3),(2,4))
function uncombine_blocks(blocks::Blocks{N}, dim::Int, blockcomb::Vector{Int}) where {N}
  blocks_uncomb = Blocks{N}()
  ncomb_tot = 0
  for i in 1:length(blocks)
    block = blocks[i]
    blockval = block[dim]
    ncomb = _number_uncombined(blockval, blockcomb)
    ncomb_shift = _number_uncombined_shift(blockval, blockcomb)
    push!(blocks_uncomb, setindex(block, blockval + ncomb_shift, dim))
    for j in 1:(ncomb - 1)
      push!(blocks_uncomb, setindex(block, blockval + ncomb_shift + j, dim))
    end
  end
  return blocks_uncomb
end

function uncombine_block(block::Block{N}, dim::Int, blockcomb::Vector{Int}) where {N}
  blocks_uncomb = Blocks{N}()
  ncomb_tot = 0
  blockval = block[dim]
  ncomb = _number_uncombined(blockval, blockcomb)
  ncomb_shift = _number_uncombined_shift(blockval, blockcomb)
  push!(blocks_uncomb, setindex(block, blockval + ncomb_shift, dim))
  for j in 1:(ncomb - 1)
    push!(blocks_uncomb, setindex(block, blockval + ncomb_shift + j, dim))
  end
  return blocks_uncomb
end

function uncombine_output(
  T::BlockSparseTensor{ElT,N},
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
  boffs_uncomb_perm, nnz_uncomb_perm = blockoffsets(blocks_uncomb_perm, inds_uncomb_perm)
  T_uncomb_perm = tensor(
    BlockSparse(ElT, boffs_uncomb_perm, nnz_uncomb_perm), inds_uncomb_perm
  )
  R = reshape(T_uncomb_perm, is)
  return R
end

function reshape(blockT::Block{NT}, indsT, indsR) where {NT}
  nblocksT = nblocks(indsT)
  nblocksR = nblocks(indsR)
  blockR = Tuple(
    CartesianIndices(nblocksR)[LinearIndices(nblocksT)[CartesianIndex(blockT)]]
  )
  return blockR
end

function uncombine(
  T::BlockSparseTensor{<:Number,NT},
  T_labels,
  is,
  is_labels,
  combdim::Int,
  blockperm::Vector{Int},
  blockcomb::Vector{Int},
) where {NT}
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
  for bof in pairs(blockoffsets(T))
    b = nzblock(bof)
    Tb_tot = blockview(T, bof)
    dimsTb_tot = dims(Tb_tot)
    bs_uncomb = uncombine_block(b, combdim, blockcomb)
    offset = 0
    for i in 1:length(bs_uncomb)
      b_uncomb = bs_uncomb[i]
      b_uncomb_perm = perm_block(b_uncomb, combdim, invblockperm)
      b_uncomb_perm_reshape = reshape(b_uncomb_perm, inds_uncomb_perm, is)
      Rb = blockview(R, b_uncomb_perm_reshape)
      b_uncomb_in_combined_dim = b_uncomb_perm[combdim]
      start = offset + 1
      stop = offset + blockdim(ind_uncomb_perm, b_uncomb_in_combined_dim)
      subind = ntuple(
        i -> i == combdim ? range(start; stop=stop) : range(1; stop=dimsTb_tot[i]), NT
      )
      offset = stop
      Tb = @view array(Tb_tot)[subind...]

      # Alternative (but maybe slower):
      #copyto!(Rb,Tb)

      if length(Tb) == 1
        Rb[1] = Tb[1]
      else
        # XXX: this used to be:
        # Rbₐᵣ = ReshapedArray(parent(Rbₐ), size(Tb), ())
        # however that doesn't work with subarrays
        Rbₐ = convert(Array, Rb)
        Rbₐᵣ = ReshapedArray(Rbₐ, size(Tb), ())
        @strided Rbₐᵣ .= Tb
      end
    end
  end
  return R
end
