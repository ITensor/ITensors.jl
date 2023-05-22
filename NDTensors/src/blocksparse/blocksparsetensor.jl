#
# BlockSparseTensor (Tensor using BlockSparse storage)
#

const BlockSparseTensor{ElT,N,StoreT,IndsT} =
  Tensor{ElT,N,StoreT,IndsT} where {StoreT<:BlockSparse}

  ## Constructors


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

function randomBlockSparseTensor(
  ::Type{ElT}, blocks::Vector{<:BlockT}, inds
) where {ElT,BlockT<:Union{Block{N},NTuple{N,<:Integer}}} where {N}
  return randomBlockSparseTensor(Random.default_rng(), ElT, blocks, inds)
end

function randomBlockSparseTensor(
  rng::AbstractRNG, ::Type{ElT}, blocks::Vector{<:BlockT}, inds
) where {ElT,BlockT<:Union{Block{N},NTuple{N,<:Integer}}} where {N}
  return randn(rng, BlockSparseTensor{ElT,N}, blocks, inds)
end

function randomBlockSparseTensor(blocks::Vector, inds)
  return randomBlockSparseTensor(Random.default_rng(), blocks, inds)
end

function randomBlockSparseTensor(rng::AbstractRNG, blocks::Vector, inds)
  return randomBlockSparseTensor(rng, Float64, blocks, inds)
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

## End constructors

# Index and block operators

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

  
# End Index and block operators

nonzeros(T::Tensor) = data(T)

#complex(::Type{BlockSparseTensor{ElT,N,StoreT,IndsT}}) where {ElT<:Number,N,StoreT<:BlockSparse
#  = Tensor{ElT,N,StoreT,IndsT} where {StoreT<:BlockSparse}

hasblock(T::Tensor, block::Block) = isassigned(blockoffsets(T), block)

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
