function uncombine(
  T::Tensor{ElT,NT,<:BlockSparseArray{ElT,NT}},
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
  T::Tensor{ElT,N,<:BlockSparseArray{ElT,N}},
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
  a_uncomb_perm =
    BlockSparseArray{ElT,length(inds_uncomb_perm),unwrap_type(T)}(undef, blocks_uncomb_perm, blockinds_uncomb_perm)
  blockinds = map(i -> [blockdim(i, b) for b in 1:nblocks(i)], is)
  R = tensor(reshape(a_uncomb_perm, blockinds), is)
  return R
end

# Uncombine the blocks along the dimension dim
# according to the pattern in blockcomb (for example, blockcomb
# is [1,2,2,3] and dim = 2, so the blocks (1,2),(2,3) get
# split into (1,2),(1,3),(2,4))
function uncombine_blocks(blocks::Vector{BlockArrays.Block{N,Int}}, dim::Int, blockcomb::Vector{Int}) where {N}
  blocks_uncomb = Vector{BlockArrays.Block{N,Int}}()
  ncomb_tot = 0
  for i in 1:length(blocks)
    block = blocks[i]
    blockval = block.n[dim]
    ncomb = _number_uncombined(blockval, blockcomb)
    ncomb_shift = _number_uncombined_shift(blockval, blockcomb)
    push!(blocks_uncomb, setindex(block, blockval + ncomb_shift, dim))
    for j in 1:(ncomb - 1)
      push!(blocks_uncomb, setindex(block, blockval + ncomb_shift + j, dim))
    end
  end
  return blocks_uncomb
end

function uncombine_block(block::BlockArrays.Block{N}, dim::Int, blockcomb::Vector{Int}) where {N}
  blocks_uncomb = Vector{BlockArrays.Block{N,Int}}()
  ncomb_tot = 0
  blockval = block.n[dim]
  ncomb = _number_uncombined(blockval, blockcomb)
  ncomb_shift = _number_uncombined_shift(blockval, blockcomb)
  push!(blocks_uncomb, setindex(block, blockval + ncomb_shift, dim))
  for j in 1:(ncomb - 1)
    push!(blocks_uncomb, setindex(block, blockval + ncomb_shift + j, dim))
  end
  return blocks_uncomb
end

# TODO: Rethink this function.
function reshape(blockT::BlockArrays.Block{NT}, indsT, indsR) where {NT}
  nblocksT = nblocks(indsT)
  nblocksR = nblocks(indsR)
  blockR = Tuple(
    CartesianIndices(nblocksR)[LinearIndices(nblocksT)[CartesianIndex(blockT.n)]]
  )
  return BlockArrays.Block(blockR)
end
