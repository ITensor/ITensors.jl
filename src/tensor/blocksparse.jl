export BlockSparse,
       BlockSparseTensor,
       offsets

#
# BlockSparse storage
#

const BlockOffset = Tuple{Int,Int}
const BlockOffsets = Vector{BlockOffset}

struct BlockSparse{ElT,VecT} <: TensorStorage{ElT}
  data::VecT
  offsets::BlockOffsets  # Block number-offset pairs
  function BlockSparse(data::VecT,offsets) where {VecT<:AbstractVector{ElT}} where {ElT}
    for jj in 1:length(offsets)-1
      block_jj,_ = offsets[jj]
      block_jj1,_ = offsets[jj+1]
      block_jj ≥ block_jj1 && error("When creating BlockSparse storage, offsets of blocks must be ordered")
    end
    new{ElT,VecT}(data,offsets)
  end
end

function BlockSparse{ElR}(data::VecT,offsets) where {ElR,VecT<:AbstractVector{ElT}} where {ElT}
  ElT == ElR ? BlockSparse(data,offsets) : BlockSparse(ElR.(data),offsets)
end
BlockSparse{ElT}() where {ElT} = BlockSparse(ElT[],BlockOffsets())

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

function Base.:*(D::BlockSparse,x::Number)
  return BlockSparse(x*data(D),offsets(D))
end
Base.:*(x::Number,D::BlockSparse) = D*x

function Base.:+(D1::BlockSparse,D2::BlockSparse)
  offsets(D1) ≠ offsets(D2) && error("Cannot add BlockSparse storage with different sparsity structure")
  return BlockSparse(data(D1)+data(D2),offsets(D1))
end

#
# BlockSparseTensor (Tensor using BlockSparse storage)
#

const BlockSparseTensor{ElT,N,StoreT,IndsT} = Tensor{ElT,N,StoreT,IndsT} where {StoreT<:BlockSparse}

offsets(T::BlockSparseTensor) = offsets(store(T))
nblocks(T::BlockSparseTensor) = nblocks(inds(T))
nblocksnz(T::BlockSparseTensor) = length(offsets(T))
blockdims(T::BlockSparseTensor,i) = blockdims(inds(T),i)
blockdim(T::BlockSparseTensor,i) = blockdim(inds(T),i)
whichblock(T::BlockSparseTensor,i) = whichblock(inds(T),i)

# From a block location, get the LinearIndex version
function linear_whichblock(T::BlockSparseTensor,whichblock)
  return linear_whichblock(inds(T),whichblock)
end

# TODO: move to tensor.jl
function linear_whichblock(inds::BlockDims,whichblock)
  return LinearIndices(nblocks(inds))[CartesianIndex(whichblock)]
end

# Check if the specified block is non-zero
function isblocknz(T::BlockSparseTensor,whichblock)
  return isblocknz(T,linear_whichblock(T,whichblock))
end

function isblocknz(T::BlockSparseTensor,linear_whichblock::Int)
  for (blocknum,_) in offsets(T)
    blocknum == linear_whichblock && return true
  end
  return false
end

# Construct a block sparse tensor from indices and locations
# of non-zero blocks
# TODO: sort the offsets if they are not ordered (currently
# if they are not ordered the BlockSparse storage constructor 
# throws an error)
function BlockSparseTensor(whichblocks::Vector{NTuple{N,Int}},
                           inds::BlockDims{N}) where {N}
  nblocks_inds = nblocks(inds)
  linear_offset_total = 0
  offsets = BlockOffsets()

  # Convention is to have offsets sorted
  # TODO: seperate this into seperate function (whichblocks,inds -> offsets)
  linear_whichblocks = [linear_whichblock(inds,whichblock_i) for whichblock_i in whichblocks]
  linear_whichblocks = sort(linear_whichblocks)

  for linear_whichblock_i in linear_whichblocks
    push!(offsets,(linear_whichblock_i,linear_offset_total))
    current_block_length = blockdim(inds,linear_whichblock_i)
    linear_offset_total += current_block_length
  end
  storage = BlockSparse{Float64}(Vector{Float64}(undef,linear_offset_total),offsets)
  return Tensor(storage,inds)
end

# Basic functionality for AbstractArray interface
Base.IndexStyle(::Type{<:BlockSparseTensor}) = IndexCartesian()

# Get the linear offset in the data storage for the specified block.
# If the specified block is not non-zero structurally, return nothing.
function blockoffset(T::BlockSparseTensor{ElT,N},
                     whichblock) where {ElT,N}
  return blockoffset(T,linear_whichblock(T,whichblock))
end

function blockoffset(T::BlockSparseTensor{ElT,N},
                     linear_whichblock::Int) where {ElT,N}
  for (blocknum_i,blockoffset_i) in offsets(T)
    if blocknum_i == linear_whichblock
      return blockoffset_i
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
                    whichblock) where {ElT,N}
  index = @MVector ones(Int,N)
  for dim in 1:N
    inddim = ind(T,dim)
    for blocknum in 1:whichblock[dim]-1
      index[dim] += blocksize(inddim,blocknum)
    end
  end
  return Tuple(index)
end

function blockend(T::BlockSparseTensor{ElT,N},
                  whichblock) where {ElT,N}
  index = @MVector zeros(Int,N)
  for dim in 1:N
    inddim = ind(T,dim)
    for blocknum in 1:whichblock[dim]
      index[dim] += blocksize(inddim,blocknum)
    end
  end
  return Tuple(index)
end

function blockindices(T::BlockSparseTensor{ElT,N},
                      whichblock) where {ElT,N}
  blockstartT = blockstart(T,whichblock)
  blockendT = blockend(T,whichblock)
  return CartesianIndex(blockstartT):CartesianIndex(blockendT)
end

# TODO: implement using a `getoffset(inds::BlockInds,i::Int...)::Int` function
# in order to share with setindex!
Base.@propagate_inbounds function Base.getindex(T::BlockSparseTensor{ElT,N},
                                                i::Vararg{Int,N}) where {ElT,N}
  # TODO: Add a checkbounds

  block_index,current_block_loc = blockindex(T,i...)

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
# These may not be valid if the Tensor has no blocks
#Base.@propagate_inbounds Base.getindex(T::BlockSparseTensor{<:Number,1},ind::Int) = store(T)[ind]
#Base.@propagate_inbounds Base.getindex(T::BlockSparseTensor{<:Number,0}) = store(T)[1]

Base.@propagate_inbounds function Base.setindex!(T::BlockSparseTensor{ElT,N},val,
                                                 i::Vararg{Int,N}) where {ElT,N}
  # TODO: Add a checkbounds

  block_index,current_block_loc = blockindex(T,i...)

  linear_block_loc = linear_whichblock(T,current_block_loc)

  current_block_dims = blockdims(T,current_block_loc)
  linear_block_index = LinearIndices(current_block_dims)[CartesianIndex(block_index)]

  for (blocknum,blockoffset) in offsets(T)
    if linear_block_loc == blocknum
      return store(T)[linear_block_index+blockoffset] = val
    end
  end
  error("Index lies in a block that is not structurally non-zero, cannot set element")
  return zero(ElT)
end

# Given a specified block, return a Dense Tensor that is a view to the data
# in that block
function blockview(T::BlockSparseTensor{ElT,N},
                   whichblock::NTuple{N,Int}) where {ElT,N}
  !isblocknz(T,whichblock) && error("Block must be structurally non-zero to get a view")
  blockoffsetT = blockoffset(T,whichblock)
  blockdimsT = blockdims(T,whichblock)
  dataTslice = @view data(store(T))[blockoffsetT+1:blockoffsetT+prod(blockdimsT)]
  return Tensor(Dense(dataTslice),blockdimsT)
end
blockview(T::BlockSparseTensor,linear_whichblock::Int) = blockview(T,whichblock(T,linear_whichblock))

dense(::Type{<:BlockSparse{ElT,VecT}}) where {ElT,VecT} = Dense{ElT,VecT}

# convert to Dense
function dense(T::TensorT) where {TensorT<:BlockSparseTensor}
  R = zeros(dense(TensorT),dense(inds(T)))
  for (blocknum_i,blockoffset_i) in offsets(T)
    whichblock_i = whichblock(T,blocknum_i)
    # This is a Tensor view of the block in A
    Tblock_i = blockview(T,whichblock_i)
    # Get the CartesianIndices for the block of data
    # in the resulting Dense tensor
    blockindices_i = blockindices(T,whichblock_i)
    # Make sure this assignment is efficient
    # (should be able to do move it as a slice,
    # but this may be calling generic AbstractArray code)
    R[blockindices_i] = Tblock_i
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

function similar_permuted(T::BlockSparseTensor{<:Number,N},
                          perm::NTuple{N,Int}) where {N}
  whichblocksp = NTuple{N,Int}[]
  for (nblock_i,offset_i) in offsets(T)
    whichblock_i = whichblock(T,nblock_i)
    whichblockp_i = permute(whichblock_i,perm)
    push!(whichblocksp,whichblockp_i)
  end
  indsR = permute(inds(T),perm)
  return BlockSparseTensor(whichblocksp,indsR)
end

function Base.permutedims(T::BlockSparseTensor{<:Number,N},
                          perm::NTuple{N,Int}) where {N}
  R = similar_permuted(T,perm)
  @show R
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
  for ((linear_whichblockT,_),(linear_whichblockR,_)) in zip(offsets(T),offsets(R))
    # Loop over non-zero blocks of T/R
    Tblock = blockview(T,linear_whichblockT)
    Rblock = blockview(R,linear_whichblockR)
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
  @show TensorR
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
    println(io,"Block: ",whichblock(T,blocknum))
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

Base.show(io::IO, T::BlockSparseTensor) = show(io,MIME("text/plain"),T)

