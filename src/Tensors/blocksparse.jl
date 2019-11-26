export BlockSparse,
       BlockSparseTensor,
       Block,
       BlockOffset,
       BlockOffsets,
       blockoffsets,
       blockview,
       nnzblocks,
       nnz

#
# BlockSparse storage
#

const Block{N} = NTuple{N,Int}
const BlockOffset{N} = Pair{Block{N},Int}
const BlockOffsets{N} = Vector{BlockOffset{N}}

struct BlockSparse{ElT,VecT,N} <: TensorStorage{ElT}
  data::VecT
  blockoffsets::BlockOffsets{N}  # Block number-offset pairs
  function BlockSparse(data::VecT,
                       blockoffsets::BlockOffsets{N}) where {VecT<:AbstractVector{ElT},N} where {ElT}
    for jj in 1:length(blockoffsets)-1
      block_jj,_ = blockoffsets[jj]
      block_jj1,_ = blockoffsets[jj+1]
      if CartesianIndex(block_jj) < CartesianIndex(block_jj1)
        error("When creating BlockSparse storage, blocks must be ordered")
      end
    end
    new{ElT,VecT,N}(data,blockoffsets)
  end
end

#function BlockSparse{ElR}(data::VecT,offsets) where {ElR,VecT<:AbstractVector{ElT}} where {ElT}
#  ElT == ElR ? BlockSparse(data,offsets) : BlockSparse(ElR.(data),offsets)
#end
#BlockSparse{ElT}() where {ElT} = BlockSparse(ElT[],BlockOffsets())

blockoffsets(D::BlockSparse) = D.blockoffsets
nnzblocks(D::BlockSparse) = length(blockoffsets(D))
nnz(D::BlockSparse) = length(data(D))

Base.similar(D::BlockSparse{T}) where {T} = BlockSparse{T}(similar(data(D)),blockoffsets(D))

# TODO: should this accept offsets and length?
#Base.similar(D::BlockSparse{T},dims) where {T} = BlockSparse{T}(similar(data(D),dim(dims)))

# TODO: should this accept offsets and length?
#Base.similar(::Type{BlockSparse{T}},dims) where {T} = BlockSparse{T}(similar(Vector{T},dim(dims)))

Base.similar(D::BlockSparse,::Type{T}) where {T} = BlockSparse{T}(similar(data(D),T),
                                                                  blockoffsets(D))
Base.copy(D::BlockSparse{T}) where {T} = BlockSparse{T}(copy(data(D)),blockoffsets(D))

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

Base.promote_rule(::Type{BlockSparse{T1}},
                  ::Type{BlockSparse{T2}}) where {T1,T2} = BlockSparse{promote_type(T1,T2)}

Base.convert(::Type{BlockSparse{R}},
             D::BlockSparse) where {R} = BlockSparse{R}(convert(Vector{R},data(D)),
                                                        blockoffsets(D))

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

nblocks(T::BlockSparseTensor) = nblocks(inds(T))
blockdims(T::BlockSparseTensor,i) = blockdims(inds(T),i)
blockdim(T::BlockSparseTensor,i) = blockdim(inds(T),i)

# Check if the specified block is non-zero
function isblocknz(T::BlockSparseTensor{ElT,N},block) where {ElT,N}
  find_block = Block(block)
  for (current_block,_) in blockoffsets(T)
    current_block == find_block && return true
  end
  return false
end

# Construct a block sparse tensor from indices and locations
# of non-zero blocks
# TODO: sort the offsets if they are not ordered (currently
# if they are not ordered the BlockSparse storage constructor 
# throws an error)
function BlockSparseTensor(blocks::Vector{Block{N}},
                           inds::BlockDims{N}) where {N}
  sort!(blocks; by=CartesianIndex, rev=true)
  blockoffsets = BlockOffsets{N}(undef,length(blocks))
  offset_total = 0
  for (i,block) in enumerate(blocks)
    blockoffsets[i] = block=>offset_total
    current_block_dim = blockdim(inds,block)
    offset_total += current_block_dim
  end
  storage = BlockSparse(Vector{Float64}(undef,offset_total),blockoffsets)
  return Tensor(storage,inds)
end

# Basic functionality for AbstractArray interface
Base.IndexStyle(::Type{<:BlockSparseTensor}) = IndexCartesian()

# Get the linear offset in the data storage for the specified block.
# If the specified block is not non-zero structurally, return nothing.
function offset(T::BlockSparseTensor{ElT,N},
                block) where {ElT,N}
  find_block = Block(block)
  for (current_block,current_offset) in blockoffsets(T)
    if current_block == find_block
      return current_offset
    end
  end
  return nothing
end

# Given a CartesianIndex in the range dims(T), get the block it is in
# and the index within that block
function blockindex(T::BlockSparseTensor{ElT,N},
                    i::Vararg{Int,N}) where {ElT,N}
  # Start in the (1,1,...,1) block
  current_block_loc = @MVector ones(Int,N)
  current_block_dims = blockdims(T,current_block_loc)
  block_index = MVector(i)
  for dim in 1:N
    while block_index[dim] > current_block_dims[dim]
      block_index[dim] -= current_block_dims[dim]
      current_block_loc[dim] += 1
      current_block_dims = blockdims(T,current_block_loc)
    end
  end
  return Tuple(block_index),Tuple(current_block_loc)
end

# Get the starting index of the block
function blockstart(T::BlockSparseTensor{ElT,N},
                    block) where {ElT,N}
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

# TODO: implement using a `getoffset(inds::BlockInds,i::Int...)::Int` function
# in order to share with setindex!
Base.@propagate_inbounds function Base.getindex(T::BlockSparseTensor{ElT,N},
                                                i::Vararg{Int,N}) where {ElT,N}
  # TODO: Add a checkbounds
  block_index,block_loc = blockindex(T,i...)
  block_dims = blockdims(T,block_loc)
  linear_block_index = LinearIndices(block_dims)[CartesianIndex(block_index)]
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

Base.@propagate_inbounds function Base.setindex!(T::BlockSparseTensor{ElT,N},val,
                                                 i::Vararg{Int,N}) where {ElT,N}
  # TODO: Add a checkbounds
  block_index,block_loc = blockindex(T,i...)
  block_dims = blockdims(T,block_loc)
  linear_block_index = LinearIndices(block_dims)[CartesianIndex(block_index)]
  for (block,offset) in blockoffsets(T)
    if block_loc == block
      return store(T)[linear_block_index+offset] = val
    end
  end
  error("Index lies in a block that is not structurally non-zero, cannot set element")
  return zero(ElT)
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
    # Make sure this assignment is efficient
    # (should be able to move it as a slice,
    # but this may be calling generic AbstractArray code)
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

# TODO: sort the output
function similar_permuted(T::BlockSparseTensor{<:Number,N},
                          perm::NTuple{N,Int}) where {N}
  blocksR = Vector{Block{N}}(undef,nnzblocks(T))
  for (i,(block,offset)) in enumerate(blockoffsets(T))
    blocksR[i] = permute(block,perm)
  end
  indsR = permute(inds(T),perm)
  return BlockSparseTensor(blocksR,indsR)
end

function Base.permutedims(T::BlockSparseTensor{<:Number,N},
                          perm::NTuple{N,Int}) where {N}
  R = similar_permuted(T,perm)
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

    # TODO: rewrite in terms of blockview(T,block)
    #blockstart = blockoffset+1
    #blockend = blockoffset+blockdim(T,block)
    #dataTslice = @view data(store(T))[blockstart:blockend]
    #Tblock = reshape(dataTslice,blockdimsT)

    Tblock = array(blockview(T,block))
    Base.print_array(io,Tblock)
    println(io)
    println(io)
  end
end

Base.show(io::IO, T::BlockSparseTensor) = show(io,MIME("text/plain"),T)

