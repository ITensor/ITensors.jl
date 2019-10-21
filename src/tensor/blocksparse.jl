export BlockSparse,
       BlockSparseTensor,
       offsets

#
# BlockSparse storage
#

const BlockOffset = Tuple{Int,Int}
const BlockOffsets = Vector{BlockOffset}

struct BlockSparse{T} <: TensorStorage
  data::Vector{T}
  offsets::BlockOffsets  # Block number-offset pairs
  BlockSparse{T}(data,offsets) where {T} = new{T}(convert(Vector{T},data),offsets)
  BlockSparse{T}() where {T} = new{T}(Vector{T}(),BlockOffsets())
end

offsets(D::BlockSparse) = D.offsets

Base.similar(D::BlockSparse{T}) where {T} = BlockSparse{T}(similar(data(D)),offsets(D))

# TODO: should this accept offsets and length?
#Base.similar(D::BlockSparse{T},dims) where {T} = BlockSparse{T}(similar(data(D),dim(dims)))

# TODO: should this accept offsets and length?
#Base.similar(::Type{BlockSparse{T}},dims) where {T} = BlockSparse{T}(similar(Vector{T},dim(dims)))

Base.similar(D::BlockSparse,::Type{T}) where {T} = BlockSparse{T}(similar(data(D),T),
                                                                  offsets(D))
Base.copy(D::BlockSparse{T}) where {T} = BlockSparse{T}(copy(data(D)),offsets(D))

# TODO: check the offsets are the same?
function Base.copyto!(D1::BlockSparse,D2::BlockSparse)
  offsets(D1) ≠ offsets(D1) && error("Cannot copy between BlockSparse storages with different offsets")
  copyto!(data(D1),data(D2))
  return D1
end

#Base.zeros(::Type{BlockSparse{T}},dim::Int) where {T} = BlockSparse{T}(zeros(T,dim))

# convert to complex
# TODO: this could be a generic TensorStorage function
Base.complex(D::BlockSparse{T}) where {T} = BlockSparse{complex(T)}(complex(data(D)),
                                                                    offsets(D))

Base.eltype(::BlockSparse{T}) where {T} = eltype(T)
# This is necessary since for some reason inference doesn't work
# with the more general definition (eltype(Nothing) === Any)
Base.eltype(::BlockSparse{Nothing}) = Nothing
Base.eltype(::Type{BlockSparse{T}}) where {T} = eltype(T)

Base.promote_rule(::Type{BlockSparse{T1}},
                  ::Type{BlockSparse{T2}}) where {T1,T2} = BlockSparse{promote_type(T1,T2)}

Base.convert(::Type{BlockSparse{R}},
             D::BlockSparse) where {R} = BlockSparse{R}(convert(Vector{R},data(D)),
                                                        offsets(D))

function Base.:*(D::BlockSparse{<:El},x::S) where {El<:Number,S<:Number}
  return BlockSparse{promote_type(El,S)}(x*data(D),offsets(D))
end

Base.:*(x::Number,D::BlockSparse) = D*x

#
# BlockSparseTensor (Tensor using BlockSparse storage)
#

const BlockSparseTensor{ElT,N,StoreT,IndsT} = Tensor{ElT,N,StoreT,IndsT} where {StoreT<:BlockSparse}

offsets(T::BlockSparseTensor) = offsets(store(T))
nblocks(T::BlockSparseTensor) = nblocks(inds(T))
nblocksnz(T::BlockSparseTensor) = length(offsets(T))
blockdims(T::BlockSparseTensor,i) = blockdims(inds(T),i)
blockdim(T::BlockSparseTensor,i) = blockdim(inds(T),i)
blockindex(T::BlockSparseTensor,i) = blockindex(inds(T),i)

# Check if the specified block is non-zero
function isblocknz(T::BlockSparseTensor,i::Int)
  for (blocknum,_) in offsets(T)
    blocknum == i && return true
  end
  return false
end

# Construct a block sparse tensor from indices and locations
# of non-zero blocks
function BlockSparseTensor(locs::Vector{NTuple{N,Int}},
                           inds::BlockDims{N}) where {N}
  nblocks_inds = nblocks(inds)
  linear_offset_total = 0
  offsets = BlockOffsets()
  for loc in locs
    # Canonical Julia way to convert CartesianIndex to LinearIndex
    linear_loc = LinearIndices(nblocks_inds)[CartesianIndex(loc)]
    push!(offsets,(linear_loc,linear_offset_total))
    current_block_length = blockdim(inds,loc)
    linear_offset_total += current_block_length
  end
  storage = BlockSparse{Float64}(Vector{Float64}(undef,linear_offset_total),offsets)
  return Tensor(storage,inds)
end

# Basic functionality for AbstractArray interface
Base.IndexStyle(::Type{<:BlockSparseTensor}) = IndexCartesian()

function getblockindex(T::BlockSparseTensor{ElT,N},i::Vararg{Int,N}) where {ElT,N}
  # Start in the (1,1,...,1) block
  current_block_loc = @MVector ones(Int64,N)
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

# TODO: implement using a `getoffset(inds::BlockInds,i::Int...)::Int` function
# in order to share with setindex!
Base.@propagate_inbounds function Base.getindex(T::BlockSparseTensor{ElT,N},
                                                i::Vararg{Int,N}) where {ElT,N}
  # TODO: Add a checkbounds

  block_index,current_block_loc = getblockindex(T,i...)

  linear_block_loc = LinearIndices(nblocks(T))[CartesianIndex(current_block_loc)]

  current_block_dims = blockdims(T,current_block_loc)
  linear_block_index = LinearIndices(current_block_dims)[CartesianIndex(block_index)]

  for (blocknum,blockoffset) in offsets(T)
    if linear_block_loc == blocknum
      return store(T)[linear_block_index+blockoffset]
    end
  end
  return zero(ElT)
end
Base.@propagate_inbounds Base.getindex(T::BlockSparseTensor{<:Number,1},ind::Int) = store(T)[ind]
Base.@propagate_inbounds Base.getindex(T::BlockSparseTensor{<:Number,0}) = store(T)[1]

function Base.summary(io::IO,
                      T::BlockSparseTensor{ElT,N}) where {ElT,N}
  println(io,typeof(T))
  println(io," ",Base.dims2string(dims(T)))
  for (dim,ind) in enumerate(inds(T))
    println(io,"Dim $dim: ",ind)
  end
  println("Number of nonzero blocks: ",nblocksnz(T))
end

function Base.show(io::IO,
                   mime::MIME"text/plain",
                   T::BlockSparseTensor)
  summary(io,T)
  println(io)
  for (blocknum,blockoffset) in offsets(T)
    blockdimsT = blockdims(T,blocknum)
    # Print the location of the current block
    println(io,"Block: ",blockindex(T,blocknum))
    # Print the dimension of the current block
    println(io," ",Base.dims2string(blockdimsT))
    blockstart = blockoffset+1
    blockend = blockoffset+blockdim(T,blocknum)
    dataTslice = @view data(store(T))[blockstart:blockend]
    Tblock = reshape(dataTslice,blockdimsT)
    Base.print_array(io,Tblock)
    println(io)
    println(io)
  end
end

