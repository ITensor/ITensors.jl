#
# BlockSparseTensor (Tensor using BlockSparse storage)
#

const BlockSparseTensor{ElT,N,StoreT,IndsT} =
  Tensor{ElT,N,StoreT,IndsT} where {StoreT<:BlockSparse}

nonzeros(T::Tensor) = data(T)

function BlockSparseTensor(
  ::Type{ElT}, ::UndefInitializer, boffs::BlockOffsets, inds
) where {ElT<:Number}
  nnz_tot = nnz(boffs, inds)
  storage = BlockSparse(ElT, undef, boffs, nnz_tot)
  return tensor(storage, inds)
end

function BlockSparseTensor(
  ::Type{ElT}, ::UndefInitializer, blocks::Vector{BlockT}, inds
) where {ElT<:Number,BlockT<:Union{Block,NTuple}}
  boffs, nnz = blockoffsets(blocks, inds)
  storage = BlockSparse(ElT, undef, boffs, nnz)
  return tensor(storage, inds)
end

"""
    BlockSparseTensor(::UndefInitializer, blocks, inds)

Construct a block sparse tensor with uninitialized memory
from indices and locations of non-zero blocks.
"""
function BlockSparseTensor(::UndefInitializer, blockoffsets, inds)
  return BlockSparseTensor(Float64, undef, blockoffsets, inds)
end

function BlockSparseTensor(
  datatype::Type{<:AbstractArray}, blockoffsets::BlockOffsets, inds
)
  nnz_tot = nnz(blockoffsets, inds)
  storage = BlockSparse(datatype, blockoffsets, nnz_tot)
  return tensor(storage, inds)
end

function BlockSparseTensor(eltype::Type{<:Number}, blockoffsets::BlockOffsets, inds)
  return BlockSparseTensor(Vector{eltype}, blockoffsets, inds)
end

function BlockSparseTensor(blockoffsets::BlockOffsets, inds)
  return BlockSparseTensor(Float64, blockoffsets, inds)
end

"""
    BlockSparseTensor(inds)

Construct a block sparse tensor with no blocks.
"""
BlockSparseTensor(inds) = BlockSparseTensor(Float64, inds)

function BlockSparseTensor(datatype::Type{<:AbstractArray}, inds)
  return BlockSparseTensor(datatype, BlockOffsets{length(inds)}(), inds)
end

function BlockSparseTensor(eltype::Type{<:Number}, inds)
  return BlockSparseTensor(Vector{eltype}, inds)
end

"""
    BlockSparseTensor(inds)

Construct a block sparse tensor with no blocks.
"""
function BlockSparseTensor(inds::Vararg{DimT,N}) where {DimT,N}
  return BlockSparseTensor(BlockOffsets{N}(), inds)
end

"""
    BlockSparseTensor(blocks::Vector{Block{N}}, inds)

Construct a block sparse tensor with the specified blocks.
Defaults to setting structurally non-zero blocks to zero.
"""
function BlockSparseTensor(blocks::Vector{BlockT}, inds) where {BlockT<:Union{Block,NTuple}}
  return BlockSparseTensor(Float64, blocks, inds)
end

function BlockSparseTensor(
  ::Type{ElT}, blocks::Vector{BlockT}, inds
) where {ElT<:Number,BlockT<:Union{Block,NTuple}}
  boffs, nnz = blockoffsets(blocks, inds)
  storage = BlockSparse(ElT, boffs, nnz)
  return tensor(storage, inds)
end

function BlockSparseTensor(
  x::Number, blocks::Vector{BlockT}, inds
) where {BlockT<:Union{Block,NTuple}}
  boffs, nnz = blockoffsets(blocks, inds)
  storage = BlockSparse(x, boffs, nnz)
  return tensor(storage, inds)
end

#complex(::Type{BlockSparseTensor{ElT,N,StoreT,IndsT}}) where {ElT<:Number,N,StoreT<:BlockSparse
#  = Tensor{ElT,N,StoreT,IndsT} where {StoreT<:BlockSparse}

function randn(
  ::Type{<:BlockSparseTensor{ElT,N}}, blocks::Vector{<:BlockT}, inds
) where {ElT,BlockT<:Union{Block{N},NTuple{N,<:Integer}}} where {N}
  boffs, nnz = blockoffsets(blocks, inds)
  storage = randn(BlockSparse{ElT}, boffs, nnz)
  return tensor(storage, inds)
end

# XXX: use the syntax:
# BlockSparseTensor(randn, ElT, blocks, inds)
function randomBlockSparseTensor(
  ::Type{ElT}, blocks::Vector{<:BlockT}, inds
) where {ElT,BlockT<:Union{Block{N},NTuple{N,<:Integer}}} where {N}
  return randn(BlockSparseTensor{ElT,N}, blocks, inds)
end

function randomBlockSparseTensor(blocks::Vector, inds)
  return randomBlockSparseTensor(Float64, blocks, inds)
end

"""
BlockSparseTensor(blocks::Vector{Block{N}},
                  inds::BlockDims...)

Construct a block sparse tensor with the specified blocks.
Defaults to setting structurally non-zero blocks to zero.
"""
function BlockSparseTensor(
  blocks::Vector{BlockT}, inds::Vararg{BlockDim,N}
) where {BlockT<:Union{Block{N},NTuple{N,<:Integer}}} where {N}
  return BlockSparseTensor(blocks, inds)
end

function zeros(
  ::BlockSparseTensor{ElT,N}, blockoffsets::BlockOffsets{N}, inds
) where {ElT,N}
  return BlockSparseTensor(ElT, blockoffsets, inds)
end

function zeros(
  ::Type{<:BlockSparseTensor{ElT,N}}, blockoffsets::BlockOffsets{N}, inds
) where {ElT,N}
  return BlockSparseTensor(ElT, blockoffsets, inds)
end

function zeros(tensortype::Type{<:BlockSparseTensor}, inds)
  return BlockSparseTensor(datatype(tensortype), inds)
end

zeros(tensor::BlockSparseTensor, inds) = zeros(typeof(tensor), inds)

# Basic functionality for AbstractArray interface
IndexStyle(::Type{<:BlockSparseTensor}) = IndexCartesian()

# Get the CartesianIndices for the range of indices
# of the specified
function blockindices(T::BlockSparseTensor{ElT,N}, block) where {ElT,N}
  return CartesianIndex(blockstart(T, block)):CartesianIndex(blockend(T, block))
end

"""
indexoffset(T::BlockSparseTensor,i::Int...) -> offset,block,blockoffset

Get the offset in the data of the specified
CartesianIndex. If it falls in a block that doesn't
exist, return nothing for the offset.
Also returns the block the index is found in and the offset
within the block.
"""
function indexoffset(T::BlockSparseTensor{ElT,N}, i::Vararg{Int,N}) where {ElT,N}
  index_within_block, block = blockindex(T, i...)
  block_dims = blockdims(T, block)
  offset_within_block = LinearIndices(block_dims)[CartesianIndex(index_within_block)]
  offset_of_block = offset(T, block)
  offset_of_i = isnothing(offset_of_block) ? nothing : offset_of_block + offset_within_block
  return offset_of_i, block, offset_within_block
end

# TODO: Add a checkbounds
# TODO: write this nicer in terms of blockview?
#       Could write: 
#       block,index_within_block = blockindex(T,i...)
#       return blockview(T,block)[index_within_block]
@propagate_inbounds function getindex(
  T::BlockSparseTensor{ElT,N}, i::Vararg{Int,N}
) where {ElT,N}
  offset, _ = indexoffset(T, i...)
  isnothing(offset) && return zero(ElT)
  return storage(T)[offset]
end

@propagate_inbounds function getindex(T::BlockSparseTensor{ElT,0}) where {ElT}
  nnzblocks(T) == 0 && return zero(ElT)
  return storage(T)[]
end

# These may not be valid if the Tensor has no blocks
#@propagate_inbounds getindex(T::BlockSparseTensor{<:Number,1},ind::Int) = storage(T)[ind]

#@propagate_inbounds getindex(T::BlockSparseTensor{<:Number,0}) = storage(T)[1]

# Add the specified block to the BlockSparseTensor
# Insert it such that the blocks remain ordered.
# Defaults to adding zeros.
# Returns the offset of the new block added.
# XXX rename to insertblock!, no need to return offset
function insertblock_offset!(T::BlockSparseTensor{ElT,N}, newblock::Block{N}) where {ElT,N}
  newdim = blockdim(T, newblock)
  newoffset = nnz(T)
  insert!(blockoffsets(T), newblock, newoffset)
  # Insert new block into data
  # TODO: Make GPU-friendly
  splice!(data(storage(T)), (newoffset + 1):newoffset, zeros(ElT, newdim))
  return newoffset
end

function insertblock!(T::BlockSparseTensor{<:Number,N}, block::Block{N}) where {N}
  insertblock_offset!(T, block)
  return T
end

insertblock!(T::BlockSparseTensor, block) = insertblock!(T, Block(block))

# Insert missing diagonal blocks as zero blocks
function insert_diag_blocks!(T::AbstractArray)
  for b in eachdiagblock(T)
    blockT = blockview(T, b)
    if isnothing(blockT)
      # Block was not found in the list, insert it
      insertblock!(T, b)
    end
  end
end

# TODO: Add a checkbounds
@propagate_inbounds function setindex!(
  T::BlockSparseTensor{ElT,N}, val, i::Vararg{Int,N}
) where {ElT,N}
  offset, block, offset_within_block = indexoffset(T, i...)
  if isnothing(offset)
    offset_of_block = insertblock_offset!(T, block)
    offset = offset_of_block + offset_within_block
  end
  storage(T)[offset] = val
  return T
end

hasblock(T::Tensor, block::Block) = isassigned(blockoffsets(T), block)

@propagate_inbounds function setindex!(
  T::BlockSparseTensor{ElT,N}, val, b::Block{N}
) where {ElT,N}
  if !hasblock(T, b)
    insertblock!(T, b)
  end
  Tb = T[b]
  Tb .= val
  return T
end

getindex(T::BlockSparseTensor, block::Block) = blockview(T, block)

to_indices(T::Tensor{<:Any,N}, b::Tuple{Block{N}}) where {N} = blockindices(T, b...)

function blockview(T::BlockSparseTensor, block::Block)
  return blockview(T, block, offset(T, block))
end

function blockview(T::BlockSparseTensor, block::Block, offset::Integer)
  return blockview(T, BlockOffset(block, offset))
end

# Case where the block isn't found, return nothing
function blockview(T::BlockSparseTensor, block::Block, ::Nothing)
  return nothing
end

blockview(T::BlockSparseTensor, block) = blockview(T, Block(block))

function blockview(T::BlockSparseTensor, bof::BlockOffset)
  blockT, offsetT = bof
  blockdimsT = blockdims(T, blockT)
  blockdimT = prod(blockdimsT)
  dataTslice = @view data(storage(T))[(offsetT + 1):(offsetT + blockdimT)]
  return tensor(Dense(dataTslice), blockdimsT)
end

view(T::BlockSparseTensor, b::Block) = blockview(T, b)

# convert to Dense
function dense(T::TensorT) where {TensorT<:BlockSparseTensor}
  R = zeros(dense(TensorT), inds(T))
  for block in keys(blockoffsets(T))
    # TODO: make sure this assignment is efficient
    R[blockindices(T, block)] = blockview(T, block)
  end
  return R
end

#
# Operations
#

# TODO: extend to case with different block structures
function +(T1::BlockSparseTensor{<:Number,N}, T2::BlockSparseTensor{<:Number,N}) where {N}
  inds(T1) ≠ inds(T2) &&
    error("Cannot add block sparse tensors with different block structure")
  R = copy(T1)
  return permutedims!!(R, T2, ntuple(identity, Val(N)), +)
end

function permutedims(T::BlockSparseTensor{<:Number,N}, perm::NTuple{N,Int}) where {N}
  blockoffsetsR, indsR = permutedims(blockoffsets(T), inds(T), perm)
  R = NDTensors.similar(T, blockoffsetsR, indsR)
  permutedims!(R, T, perm)
  return R
end

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
  is,
  combdim::Int,
  blockperm::Vector{Int},
  blockcomb::Vector{Int},
) where {ElT<:Number,N}
  ind_uncomb_perm = ⊗(setdiff(is, inds(T))...)
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
  is,
  combdim::Int,
  blockperm::Vector{Int},
  blockcomb::Vector{Int},
) where {NT}
  NR = length(is)
  R = uncombine_output(T, is, combdim, blockperm, blockcomb)
  invblockperm = invperm(blockperm)

  # This is needed for reshaping the block
  # It is already calculated in uncombine_output, use it from there
  ind_uncomb_perm = ⊗(setdiff(is, inds(T))...)
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

function copyto!(R::BlockSparseTensor, T::BlockSparseTensor)
  for bof in pairs(blockoffsets(T))
    copyto!(blockview(R, nzblock(bof)), blockview(T, bof))
  end
  return R
end

# TODO: handle case where:
# f(zero(ElR),zero(ElT)) != promote_type(ElR,ElT)
function permutedims!!(
  R::BlockSparseTensor{ElR,N},
  T::BlockSparseTensor{ElT,N},
  perm::NTuple{N,Int},
  f::Function=(r, t) -> t,
) where {ElR,ElT,N}
  RR = convert(promote_type(typeof(R), typeof(T)), R)
  #@timeit_debug timer "block sparse permutedims!!" begin
  bofsRR = blockoffsets(RR)
  bofsT = blockoffsets(T)

  # Determine if bofsRR has been copied
  copy_bofsRR = false

  new_nnz = nnz(RR)
  for (blockT, offsetT) in pairs(bofsT)
    blockTperm = permute(blockT, perm)
    if !isassigned(bofsRR, blockTperm)
      if !copy_bofsRR
        bofsRR = deepcopy(bofsRR)
        copy_bofsRR = true
      end
      insert!(bofsRR, blockTperm, new_nnz)
      new_nnz += blockdim(T, blockT)
    end
  end

  ## RR = BlockSparseTensor(promote_type(ElR,ElT), undef,
  ##                        bofsRR, inds(R))
  ## # Directly copy the data since it is the same blocks
  ## # and offsets
  ## copyto!(data(RR), data(R))

  if new_nnz > nnz(RR)
    dataRR = append!(data(RR), zeros(new_nnz - nnz(RR)))
    RR = Tensor(BlockSparse(dataRR, bofsRR), inds(RR))
  end

  permutedims!(RR, T, perm, f)
  return RR
  #end
end

# <fermions>
scale_blocks!(T, compute_fac::Function=(b) -> 1) = T

# <fermions>
function scale_blocks!(
  T::BlockSparseTensor{<:Number,N}, compute_fac::Function=(b) -> 1
) where {N}
  for blockT in keys(blockoffsets(T))
    fac = compute_fac(blockT)
    if fac != 1
      Tblock = blockview(T, blockT)
      scale!(Tblock, fac)
    end
  end
  return T
end

# <fermions>
permfactor(perm, block, inds) = 1

# Version where it is known that R has the same blocks
# as T
function permutedims!(
  R::BlockSparseTensor{<:Number,N},
  T::BlockSparseTensor{<:Number,N},
  perm::NTuple{N,Int},
  f::Function=(r, t) -> t,
) where {N}
  for blockT in keys(blockoffsets(T))
    # Loop over non-zero blocks of T/R
    Tblock = blockview(T, blockT)
    Rblock = blockview(R, permute(blockT, perm))

    # <fermions>
    pfac = permfactor(perm, blockT, inds(T))
    if pfac == 1
      permutedims!(Rblock, Tblock, perm, f)
    else
      fac_f = (r, t) -> f(r, pfac * t)
      permutedims!(Rblock, Tblock, perm, fac_f)
    end
  end
  return R
end

#
# Contraction
#

# TODO: complete this function: determine the output blocks from the input blocks
# Also, save the contraction list (which block-offsets contract with which),
# may not be generic with other contraction functions!
function contraction_output(
  T1::TensorT1, T2::TensorT2, indsR
) where {TensorT1<:BlockSparseTensor,TensorT2<:BlockSparseTensor}
  TensorR = contraction_output_type(TensorT1, TensorT2, indsR)
  return NDTensors.similar(TensorR, blockoffsetsR, indsR)
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

function contract_labels(labels1, labels2, labelsR)
  labels1_to_labels2 = find_matching_positions(labels1, labels2)
  labels1_to_labelsR = find_matching_positions(labels1, labelsR)
  labels2_to_labelsR = find_matching_positions(labels2, labelsR)
  return labels1_to_labels2, labels1_to_labelsR, labels2_to_labelsR
end

function are_blocks_contracted(
  block1::Block{N1}, block2::Block{N2}, labels1_to_labels2::NTuple{N1,Int}
) where {N1,N2}
  t1 = Tuple(block1)
  t2 = Tuple(block2)
  for i1 in 1:N1
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

function contract_blocks(
  block1::Block{N1}, labels1_to_labelsR, block2::Block{N2}, labels2_to_labelsR, ::Val{NR}
) where {N1,N2,NR}
  blockR = ntuple(_ -> UInt(0), Val(NR))
  t1 = Tuple(block1)
  t2 = Tuple(block2)
  for i1 in 1:N1
    iR = @inbounds labels1_to_labelsR[i1]
    if iR > 0
      blockR = @inbounds setindex(blockR, t1[i1], iR)
    end
  end
  for i2 in 1:N2
    iR = @inbounds labels2_to_labelsR[i2]
    if iR > 0
      blockR = @inbounds setindex(blockR, t2[i2], iR)
    end
  end
  return Block{NR}(blockR)
end

function _contract_blockoffsets(
  boffs1::BlockOffsets{N1},
  inds1,
  labels1,
  boffs2::BlockOffsets{N2},
  inds2,
  labels2,
  indsR,
  labelsR,
) where {N1,N2}
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

function _threaded_contract_blockoffsets(
  boffs1::BlockOffsets{N1},
  inds1,
  labels1,
  boffs2::BlockOffsets{N2},
  inds2,
  labels2,
  indsR,
  labelsR,
) where {N1,N2}
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

function contract_blockoffsets(args...)
  if using_threaded_blocksparse() && nthreads() > 1
    return _threaded_contract_blockoffsets(args...)
  end
  return _contract_blockoffsets(args...)
end

function contraction_output(
  T1::TensorT1, labelsT1, T2::TensorT2, labelsT2, labelsR
) where {TensorT1<:BlockSparseTensor,TensorT2<:BlockSparseTensor}
  indsR = contract_inds(inds(T1), labelsT1, inds(T2), labelsT2, labelsR)
  TensorR = contraction_output_type(TensorT1, TensorT2, indsR)
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
  R = NDTensors.similar(TensorR, blockoffsetsR, indsR)
  return R, contraction_plan
end

function contract(
  T1::BlockSparseTensor{<:Any,N1},
  labelsT1,
  T2::BlockSparseTensor{<:Any,N2},
  labelsT2,
  labelsR=contract_labels(labelsT1, labelsT2),
) where {N1,N2}
  #@timeit_debug timer "Block sparse contract" begin
  R, contraction_plan = contraction_output(T1, labelsT1, T2, labelsT2, labelsR)
  R = contract!(R, labelsR, T1, labelsT1, T2, labelsT2, contraction_plan)
  return R
  #end
end

# <fermions>
function compute_alpha(
  ElR, labelsR, blockR, indsR, labelsT1, blockT1, indsT1, labelsT2, blockT2, indsT2
)
  return one(ElR)
end

# XXX: this is not thread safe, divide into groups of
# contractions that contract into the same block
function _threaded_contract!(
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

function contract!(
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
    _threaded_contract!(R, labelsR, T1, labelsT1, T2, labelsT2, contraction_plan)
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

const IntTuple = NTuple{N,Int} where {N}
const IntOrIntTuple = Union{Int,IntTuple}

function permute_combine(inds::IndsT, pos::Vararg{IntOrIntTuple,N}) where {IndsT,N}
  IndT = eltype(IndsT)
  # Using SizedVector since setindex! doesn't
  # work for MVector when eltype not isbitstype
  newinds = SizedVector{N,IndT}(undef)
  for i in 1:N
    pos_i = pos[i]
    newind_i = inds[pos_i[1]]
    for p in 2:length(pos_i)
      newind_i = newind_i ⊗ inds[pos_i[p]]
    end
    newinds[i] = newind_i
  end
  IndsR = similartype(IndsT, Val{N})
  indsR = IndsR(Tuple(newinds))
  return indsR
end

"""
Indices are combined according to the grouping of the input,
for example (1,2),3 will combine the first two indices.
"""
function combine(inds::IndsT, com::Vararg{IntOrIntTuple,N}) where {IndsT,N}
  IndT = eltype(IndsT)
  # Using SizedVector since setindex! doesn't
  # work for MVector when eltype not isbitstype
  newinds = SizedVector{N,IndT}(undef)
  i_orig = 1
  for i in 1:N
    newind_i = inds[i_orig]
    i_orig += 1
    for p in 2:length(com[i])
      newind_i = newind_i ⊗ inds[i_orig]
      i_orig += 1
    end
    newinds[i] = newind_i
  end
  IndsR = similartype(IndsT, Val{N})
  indsR = IndsR(Tuple(newinds))
  return indsR
end

function permute_combine(
  boffs::BlockOffsets, inds::IndsT, pos::Vararg{IntOrIntTuple,N}
) where {IndsT,N}
  perm = flatten(pos...)
  boffsp, indsp = permutedims(boffs, inds, perm)
  indsR = combine(indsp, pos...)
  boffsR = reshape(boffsp, indsp, indsR)
  return boffsR, indsR
end

function reshape(boffsT::BlockOffsets{NT}, indsT, indsR) where {NT}
  NR = length(indsR)
  boffsR = BlockOffsets{NR}()
  nblocksT = nblocks(indsT)
  nblocksR = nblocks(indsR)
  for (blockT, offsetT) in pairs(boffsT)
    blockR = Block(
      CartesianIndices(nblocksR)[LinearIndices(nblocksT)[CartesianIndex(blockT)]]
    )
    insert!(boffsR, blockR, offsetT)
  end
  return boffsR
end

function reshape(boffsT::BlockOffsets{NT}, blocksR::Vector{Block{NR}}) where {NR,NT}
  boffsR = BlockOffsets{NR}()
  # TODO: check blocksR is ordered and are properly reshaped
  # versions of the blocks of boffsT
  for (i, (blockT, offsetT)) in enumerate(boffsT)
    blockR = blocksR[i]
    boffsR[blockR] = offsetT
  end
  return boffsR
end

reshape(T::BlockSparse, boffsR::BlockOffsets) = BlockSparse(data(T), boffsR)

function reshape(T::BlockSparseTensor, boffsR::BlockOffsets, indsR)
  storeR = reshape(storage(T), boffsR)
  return tensor(storeR, indsR)
end

function reshape(T::BlockSparseTensor, indsR)
  # TODO: add some checks that the block dimensions
  # are consistent (e.g. nnzblocks(T) == nnzblocks(R), etc.)
  boffsR = reshape(blockoffsets(T), inds(T), indsR)
  R = reshape(T, boffsR, indsR)
  return R
end

function permute_combine(
  T::BlockSparseTensor{ElT,NT,IndsT}, pos::Vararg{IntOrIntTuple,NR}
) where {ElT,NT,IndsT,NR}
  boffsR, indsR = permute_combine(blockoffsets(T), inds(T), pos...)

  perm = flatten(pos...)

  length(perm) ≠ NT && error("Index positions must add up to order of Tensor ($NT)")
  isperm(perm) || error("Index positions must be a permutation")

  if !is_trivial_permutation(perm)
    Tp = permutedims(T, perm)
  else
    Tp = copy(T)
  end
  NR == NT && return Tp
  R = reshape(Tp, boffsR, indsR)
  return R
end

#
# Print block sparse tensors
#

#function summary(io::IO,
#                 T::BlockSparseTensor{ElT,N}) where {ElT,N}
#  println(io,Base.dims2string(dims(T))," ",typeof(T))
#  for (dim,ind) in enumerate(inds(T))
#    println(io,"Dim $dim: ",ind)
#  end
#  println(io,"Number of nonzero blocks: ",nnzblocks(T))
#end

#function summary(io::IO,
#                 T::BlockSparseTensor{ElT,N}) where {ElT,N}
#  println(io,typeof(T))
#  println(io,Base.dims2string(dims(T))," ",typeof(T))
#  for (dim,ind) in enumerate(inds(T))
#    println(io,"Dim $dim: ",ind)
#  end
#  println("Number of nonzero blocks: ",nnzblocks(T))
#end

function _range2string(rangestart::NTuple{N,Int}, rangeend::NTuple{N,Int}) where {N}
  s = ""
  for n in 1:N
    s = string(s, rangestart[n], ":", rangeend[n])
    if n < N
      s = string(s, ", ")
    end
  end
  return s
end

function show(io::IO, mime::MIME"text/plain", T::BlockSparseTensor)
  summary(io, T)
  for (n, block) in enumerate(keys(blockoffsets(T)))
    blockdimsT = blockdims(T, block)
    println(io, block)
    println(io, " [", _range2string(blockstart(T, block), blockend(T, block)), "]")
    print_tensor(io, blockview(T, block))
    n < nnzblocks(T) && print(io, "\n\n")
  end
end

show(io::IO, T::BlockSparseTensor) = show(io, MIME("text/plain"), T)
