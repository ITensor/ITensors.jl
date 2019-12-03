export BlockSparse,
       BlockSparseTensor,
       Block,
       block,
       BlockOffset,
       BlockOffsets,
       blockoffsets,
       blockview,
       nnzblocks,
       nnz,
       findblock,
       isblocknz

#
# BlockSparse storage
#

const Block{N} = NTuple{N,Int}
const BlockOffset{N} = Pair{Block{N},Int}
const BlockOffsets{N} = Vector{BlockOffset{N}}

block(bof::BlockOffset) = bof.first
offset(bof::BlockOffset) = bof.second
block(block::Block) = block

nnzblocks(bofs::BlockOffsets) = length(bofs)

# define block ordering with reverse lexographical order
function isblockless(b1::Block{N},
                     b2::Block{N}) where {N}
  return CartesianIndex(b1) < CartesianIndex(b2)
end

function isblockless(bof1::BlockOffset{N},
                     bof2::BlockOffset{N}) where {N}
  return isblockless(block(bof1),block(bof2))
end

function isblockless(bof1::BlockOffset{N},
                     b2::Block{N}) where {N}
  return isblockless(block(bof1),b2)
end

function isblockless(b1::Block{N},
                     bof2::BlockOffset{N}) where {N}
  return isblockless(b1,block(bof2))
end

function check_blocks_sorted(blockoffsets::BlockOffsets)
  for jj in 1:length(blockoffsets)-1
    block_jj = block(blockoffsets[jj])
    block_jj1 = block(blockoffsets[jj+1])
    if !isblockless(block_jj,block_jj1)
      error("Blocks in BlockOffsets not ordered")
    end
  end
  return
end

struct BlockSparse{ElT,VecT,N} <: TensorStorage{ElT}
  data::VecT
  blockoffsets::BlockOffsets{N}  # Block number-offset pairs
  function BlockSparse(data::VecT,
                       blockoffsets::BlockOffsets{N}) where {VecT<:AbstractVector{ElT},N} where {ElT}
    # TODO: make this a debug check?
    check_blocks_sorted(blockoffsets)
    new{ElT,VecT,N}(data,blockoffsets)
  end
end

function BlockSparse(::Type{ElT},
                     blockoffsets::BlockOffsets,
                     dim::Integer) where {ElT<:Number}
  return BlockSparse(zeros(ElT,dim),blockoffsets)
end

function BlockSparse(::Type{ElT},
                     ::UndefInitializer,
                     blockoffsets::BlockOffsets,
                     dim::Integer) where {ElT<:Number}
  return BlockSparse(Vector{Float64}(undef,dim),blockoffsets)
end

BlockSparse(blockoffsets::BlockOffsets,
            dim::Integer) = BlockSparse(Float64,blockoffsets,dim)

BlockSparse(::UndefInitializer,
            blockoffsets::BlockOffsets,
            dim::Integer) = BlockSparse(Float64,undef,blockoffsets,dim)

#function BlockSparse{ElR}(data::VecT,offsets) where {ElR,VecT<:AbstractVector{ElT}} where {ElT}
#  ElT == ElR ? BlockSparse(data,offsets) : BlockSparse(ElR.(data),offsets)
#end
#BlockSparse{ElT}() where {ElT} = BlockSparse(ElT[],BlockOffsets())

blockoffsets(D::BlockSparse) = D.blockoffsets
nnzblocks(D::BlockSparse) = length(blockoffsets(D))
nnz(D::BlockSparse) = length(data(D))

function Base.similar(D::BlockSparse{ElT}) where {ElT}
  return BlockSparse{ElT}(similar(data(D)),blockoffsets(D))
end

# TODO: should this accept offsets and length?
#Base.similar(D::BlockSparse{T},dims) where {T} = BlockSparse{T}(similar(data(D),dim(dims)))

# TODO: should this accept offsets and length?
#Base.similar(::Type{BlockSparse{T}},dims) where {T} = BlockSparse{T}(similar(Vector{T},dim(dims)))

Base.similar(D::BlockSparse,
             ::Type{ElT}) where {ElT} = BlockSparse{T}(similar(data(D),T),
                                                       blockoffsets(D))
Base.copy(D::BlockSparse{T}) where {T} = BlockSparse{T}(copy(data(D)),
                                                        blockoffsets(D))

# TODO: check the offsets are the same?
function Base.copyto!(D1::BlockSparse,D2::BlockSparse)
  blockoffsets(D1) ≠ blockoffsets(D1) && error("Cannot copy between BlockSparse storages with different offsets")
  copyto!(data(D1),data(D2))
  return D1
end

#Base.zeros(::Type{BlockSparse{T}},dim::Int) where {T} = BlockSparse{T}(zeros(T,dim))

# convert to complex
# TODO: this could be a generic TensorStorage function
Base.complex(D::BlockSparse{T}) where {T} = BlockSparse{complex(T)}(complex(data(D)),
                                                                    blockoffsets(D))

Base.eltype(::BlockSparse{T}) where {T} = eltype(T)
# This is necessary since for some reason inference doesn't work
# with the more general definition (eltype(Nothing) === Any)
Base.eltype(::BlockSparse{Nothing}) = Nothing
Base.eltype(::Type{BlockSparse{T}}) where {T} = eltype(T)

function Base.promote_rule(::Type{BlockSparse{T1}},
                           ::Type{BlockSparse{T2}}) where {T1,T2}
  return BlockSparse{promote_type(T1,T2)}
end

function Base.convert(::Type{BlockSparse{R}},
                      D::BlockSparse) where {R}
  return BlockSparse{R}(convert(Vector{R},data(D)),
                        blockoffsets(D))
end

function Base.:*(D::BlockSparse,x::Number)
  return BlockSparse(x*data(D),blockoffsets(D))
end
Base.:*(x::Number,D::BlockSparse) = D*x

function Base.:+(D1::BlockSparse,D2::BlockSparse)
  blockoffsets(D1) ≠ blockoffsets(D2) && error("Cannot add BlockSparse storage with different sparsity structure")
  return BlockSparse(data(D1)+data(D2),blockoffsets(D1))
end

#
# BlockSparseTensor (Tensor using BlockSparse storage)
#

const BlockSparseTensor{ElT,N,StoreT,IndsT} = Tensor{ElT,N,StoreT,IndsT} where {StoreT<:BlockSparse}

blockoffsets(T::BlockSparseTensor) = blockoffsets(store(T))
nnzblocks(T::BlockSparseTensor) = nnzblocks(store(T))
nnz(T::BlockSparseTensor) = nnz(store(T))

function nnz(bofs::BlockOffsets,inds)
  nnzblocks(bofs) == 0 && return 0
  lastblock,lastoffset = bofs[end]
  return lastoffset + blockdim(inds,lastblock)
end

nblocks(T::BlockSparseTensor) = nblocks(inds(T))
blockdims(T::BlockSparseTensor{ElT,N},
          block::Block{N}) where {ElT,N} = blockdims(inds(T),block)
blockdim(T::BlockSparseTensor{ElT,N},
         block::Block{N}) where {ElT,N} = blockdim(inds(T),block)

"""
findblock(::BlockOffsets,::Block)

Output the index of the specified block in the block-offsets
list.
If not found, return nothing.
Searches assuming the blocks are sorted.
"""
function findblock(bofs::BlockOffsets{N},
                   block::Block{N}) where {N}
  r = searchsorted(bofs,block;lt=isblockless)
  length(r)>1 && error("In findblock, more than one block found")   
  length(r)==0 && return nothing
  return r[]
end

findblock(T::BlockSparseTensor{ElT,N},
          block::Block{N}) where {ElT,N} = findblock(blockoffsets(T),block)

"""
isblocknz(T::BlockSparseTensor,
          block::Block)

Check if the specified block is non-zero
"""
function isblocknz(T::BlockSparseTensor{ElT,N},
                   block::Block{N}) where {ElT,N}
  isnothing(findblock(T,block)) && return false
  return true
end

function get_blockoffsets(blocks::Vector{Block{N}},
                          inds) where {N}
  blocks = sort(blocks;lt=isblockless)
  blockoffsets = BlockOffsets{N}(undef,length(blocks))
  offset_total = 0
  for (i,block) in enumerate(blocks)
    blockoffsets[i] = block=>offset_total
    current_block_dim = blockdim(inds,block)
    offset_total += current_block_dim
  end
  return blockoffsets,offset_total
end

function BlockSparseTensor(::Type{ElT},
                           ::UndefInitializer,
                           blockoffsets::BlockOffsets{N},
                           inds) where {ElT<:Number,N}
  nnz_tot = nnz(blockoffsets,inds)
  storage = BlockSparse(ElT,undef,blockoffsets,nnz_tot)
  return Tensor(storage,inds)
end

function BlockSparseTensor(::UndefInitializer,
                           blockoffsets::BlockOffsets{N},
                           inds) where {N}
  return BlockSparseTensor(Float64,undef,blockoffsets,inds)
end

function BlockSparseTensor(::Type{ElT},
                           blockoffsets::BlockOffsets{N},
                           inds) where {ElT<:Number,N}
  nnz_tot = nnz(blockoffsets,inds)
  storage = BlockSparse(ElT,blockoffsets,nnz_tot)
  return Tensor(storage,inds)
end

function BlockSparseTensor(blockoffsets::BlockOffsets{N},
                           inds) where {N}
  return BlockSparseTensor(Float64,blockoffsets,inds)
end

"""
BlockSparseTensor(::UndefInitializer,
                  blocks::Vector{Block{N}},
                  inds)

Construct a block sparse tensor with uninitialized memory
from indices and locations of non-zero blocks.
"""
function BlockSparseTensor(::UndefInitializer,
                           blocks::Vector{Block{N}},
                           inds) where {N}
  blockoffsets,nnz = get_blockoffsets(blocks,inds)
  storage = BlockSparse(undef,blockoffsets,nnz)
  return Tensor(storage,inds)
end

function BlockSparseTensor(::UndefInitializer,
                           blocks::Vector{Block{N}},
                           inds::Vararg{DimT,N}) where {DimT,N}
  return BlockSparseTensor(undef,blocks,inds)
end

"""
BlockSparseTensor(inds)

Construct a block sparse tensor with no blocks.
"""
function BlockSparseTensor(inds)
  return BlockSparseTensor(BlockOffsets{length(inds)}(),inds)
end

"""
BlockSparseTensor(inds)

Construct a block sparse tensor with no blocks.
"""
function BlockSparseTensor(inds::Vararg{DimT,N}) where {DimT,N}
  return BlockSparseTensor(BlockOffsets{N}(),inds)
end

"""
BlockSparseTensor(blocks::Vector{Block{N}},
                  inds)

Construct a block sparse tensor with the specified blocks.
Defaults to setting structurally non-zero blocks to zero.
"""
function BlockSparseTensor(blocks::Vector{Block{N}},
                           inds) where {N}
  blockoffsets,offset_total = get_blockoffsets(blocks,inds)
  storage = BlockSparse(blockoffsets,offset_total)
  return Tensor(storage,inds)
end

"""
BlockSparseTensor(blocks::Vector{Block{N}},
                  inds...)

Construct a block sparse tensor with the specified blocks.
Defaults to setting structurally non-zero blocks to zero.
"""
function BlockSparseTensor(blocks::Vector{Block{N}},
                           inds::Vararg{DimT,N}) where {DimT,N}
  return BlockSparseTensor(blocks,inds)
end

function Base.similar(::BlockSparseTensor{ElT,N},
                      blockoffsets::BlockOffsets{N},
                      inds) where {ElT,N}
  return BlockSparseTensor(ElT,undef,blockoffsets,inds)
end

# Basic functionality for AbstractArray interface
Base.IndexStyle(::Type{<:BlockSparseTensor}) = IndexCartesian()

# Get the nth offset
offset(bofs::BlockOffsets,n::Int) = offset(bofs[n])

function offset(bofs::BlockOffsets{N},
                block::Block{N}) where {N}
  block_pos = findblock(bofs,block)
  isnothing(block_pos) && return nothing
  return offset(bofs,block_pos)
end

"""
offset(T::BlockSparseTensor,
       block::Block)

Get the linear offset in the data storage for the specified block.
If the specified block is not non-zero structurally, return nothing.
"""
function offset(T::BlockSparseTensor{ElT,N},
                block::Block{N}) where {ElT,N}
  return offset(blockoffsets(T),block)
end

offset(T::BlockSparseTensor,n::Int) = offset(blockoffsets(T),n)

  #block_pos = findblock(T,block)
  #isnothing(block_pos) && return nothing
  #return offset(T,block_pos)
  #for (current_block,current_offset) in blockoffsets(T)
  #  if current_block == block
  #    return current_offset
  #  end
  #end
  #return nothing

# Given a CartesianIndex in the range dims(T), get the block it is in
# and the index within that block
function blockindex(T::BlockSparseTensor{ElT,N},
                    i::Vararg{Int,N}) where {ElT,N}
  # Start in the (1,1,...,1) block
  current_block_loc = @MVector ones(Int,N)
  current_block_dims = blockdims(T,Tuple(current_block_loc))
  block_index = MVector(i)
  for dim in 1:N
    while block_index[dim] > current_block_dims[dim]
      block_index[dim] -= current_block_dims[dim]
      current_block_loc[dim] += 1
      current_block_dims = blockdims(T,Tuple(current_block_loc))
    end
  end
  return Tuple(block_index),Tuple(current_block_loc)
end

# Get the starting index of the block
function blockstart(T::BlockSparseTensor{ElT,N},
                    block::Block{N}) where {ElT,N}
  start_index = @MVector ones(Int,N)
  for j in 1:N
    ind_j = ind(T,j)
    for block_j in 1:block[j]-1
      start_index[j] += blockdim(ind_j,block_j)
    end
  end
  return CartesianIndex(Tuple(start_index))
end

# Get the ending index of the block
function blockend(T::BlockSparseTensor{ElT,N},
                  block) where {ElT,N}
  end_index = @MVector zeros(Int,N)
  for j in 1:N
    ind_j = ind(T,j)
    for block_j in 1:block[j]
      end_index[j] += blockdim(ind_j,block_j)
    end
  end
  return CartesianIndex(Tuple(end_index))
end

# Get the CartesianIndices for the range of indices
# of the specified
function blockindices(T::BlockSparseTensor{ElT,N},
                      block) where {ElT,N}
  return blockstart(T,block):blockend(T,block)
end

Base.@propagate_inbounds function Base.getindex(T::BlockSparseTensor{ElT,N},
                                                i::Vararg{Int,N}) where {ElT,N}
  # TODO: Add a checkbounds
  block_index,block_loc = blockindex(T,i...)
  block_dims = blockdims(T,block_loc)
  linear_block_index = LinearIndices(block_dims)[CartesianIndex(block_index)]
  # TODO: replace with a sorted search
  for (block,offset) in blockoffsets(T)
    if block_loc == block
      return store(T)[linear_block_index+offset]
    end
  end
  return zero(ElT)
end
# These may not be valid if the Tensor has no blocks
#Base.@propagate_inbounds Base.getindex(T::BlockSparseTensor{<:Number,1},ind::Int) = store(T)[ind]
#Base.@propagate_inbounds Base.getindex(T::BlockSparseTensor{<:Number,0}) = store(T)[1]

Base.@propagate_inbounds function Base.setindex!(T::BlockSparseTensor{ElT,N},
                                                 val,
                                                 i::Vararg{Int,N}) where {ElT,N}
  # TODO: Add a checkbounds
  block_index,block_loc = blockindex(T,i...)
  block_dims = blockdims(T,block_loc)
  linear_block_index = LinearIndices(block_dims)[CartesianIndex(block_index)]
  # TODO: turn into sorted search
  for (block,offset) in blockoffsets(T)
    if block_loc == block
      store(T)[linear_block_index+offset] = val
      return T
    end
  end
  println("Index lies in a block that is not structurally non-zero, adding block")
  return T
end

# Given a specified block, return a Dense Tensor that is a view to the data
# in that block
function blockview(T::BlockSparseTensor{ElT,N},
                   block) where {ElT,N}
  !isblocknz(T,block) && error("Block must be structurally non-zero to get a view")
  blockoffsetT = offset(T,block)
  blockdimsT = blockdims(T,block)
  dataTslice = @view data(store(T))[blockoffsetT+1:blockoffsetT+prod(blockdimsT)]
  return Tensor(Dense(dataTslice),blockdimsT)
end

dense(::Type{<:BlockSparse{ElT,VecT}}) where {ElT,VecT} = Dense{ElT,VecT}

# convert to Dense
function dense(T::TensorT) where {TensorT<:BlockSparseTensor}
  R = zeros(dense(TensorT),dense(inds(T)))
  for (block,offset) in blockoffsets(T)
    # TODO: make sure this assignment is efficient
    R[blockindices(T,block)] = blockview(T,block)
  end
  return R
end

#
# Operations
#

function Base.:+(T1::BlockSparseTensor,T2::BlockSparseTensor)
  inds(T1) ≠ inds(T2) && error("Cannot add block sparse tensors with different block structure")  
  return Tensor(store(T1)+store(T2),inds(T1))
end

function similar_permuted(T::BlockSparseTensor{ElT,N},
                          perm::NTuple{N,Int}) where {ElT,N}
  blocksR = Vector{Block{N}}(undef,nnzblocks(T))
  for (i,(block,offset)) in enumerate(blockoffsets(T))
    blocksR[i] = permute(block,perm)
  end
  indsR = permute(inds(T),perm)
  return BlockSparseTensor(ElT,undef,blocksR,indsR)
end

# Permute the blockoffsets and indices
function permute(blockoffsets::BlockOffsets{N},
                 inds,
                 perm::NTuple{N,Int}) where {N}
  blocksR = Vector{Block{N}}(undef,nnzblocks(blockoffsets))
  for (i,(block,offset)) in enumerate(blockoffsets)
    blocksR[i] = permute(block,perm)
  end
  indsR = permute(inds,perm)
  blockoffsetsR,_ = get_blockoffsets(blocksR,indsR)
  return blockoffsetsR,indsR
end

function Base.permutedims(T::BlockSparseTensor{<:Number,N},
                          perm::NTuple{N,Int}) where {N}
  #R = similar_permuted(T,perm)
  blockoffsetsR,indsR = permute(blockoffsets(T),inds(T),perm)
  R = similar(T,blockoffsetsR,indsR)
  R = permutedims!!(R,T,perm)
  return R
end

function permutedims!!(R::BlockSparseTensor{<:Number,N},
                       T::BlockSparseTensor{<:Number,N},
                       perm::NTuple{N,Int}) where {N}
  R = permutedims!(R,T,perm)
  return R
end

function Base.permutedims!(R::BlockSparseTensor{<:Number,N},
                           T::BlockSparseTensor{<:Number,N},
                           perm::NTuple{N,Int}) where {N}
  for (blockT,_) in blockoffsets(T)
    # Loop over non-zero blocks of T/R
    Tblock = blockview(T,blockT)
    Rblock = blockview(R,permute(blockT,perm))
    permutedims!(Rblock,Tblock,perm)
  end
  return R
end

#
# Contraction
#

function contraction_output(T1::TensorT1,
                            T2::TensorT2,
                            indsR::IndsR) where {TensorT1<:BlockSparseTensor,
                                                 TensorT2<:BlockSparseTensor,
                                                 IndsR}
  TensorR = contraction_output_type(TensorT1,TensorT2,IndsR)
  #error("In contraction_output(::BlockSparseTensor,BlockSparseTensor,::IndsR), need to determine output blocks")
  return similar(TensorR,indsR,offsetsR)
end

#
# Print block sparse tensors
#

function Base.summary(io::IO,
                      T::BlockSparseTensor{ElT,N}) where {ElT,N}
  println(io,typeof(T))
  println(io," ",Base.dims2string(dims(T)))
  for (dim,ind) in enumerate(inds(T))
    println(io,"Dim $dim: ",ind)
  end
  println("Number of nonzero blocks: ",nnzblocks(T))
end

function Base.show(io::IO,
                   mime::MIME"text/plain",
                   T::BlockSparseTensor)
  summary(io,T)
  println(io)
  for (block,_) in blockoffsets(T)
    blockdimsT = blockdims(T,block)
    # Print the location of the current block
    println(io,"Block: ",block)
    # Print the dimension of the current block
    println(io," ",Base.dims2string(blockdimsT))
    # TODO: replace with a Tensor show method
    # instead of converting to array (for other
    # storage types, like CuArray)
    Tblock = array(blockview(T,block))
    Base.print_array(io,Tblock)
    println(io)
    println(io)
  end
end

Base.show(io::IO, T::BlockSparseTensor) = show(io,MIME("text/plain"),T)

