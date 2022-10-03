"""
    BlockDim

An index for a BlockSparseTensor.
"""
const BlockDim = Vector{Int}

# Makes for generic code
dim(d::BlockDim) = sum(d)

"""
    BlockDims{N}

Dimensions used for BlockSparse NDTensors.
Each entry lists the block sizes in each dimension.
"""
const BlockDims{N} = NTuple{N,BlockDim}

Base.ndims(ds::Type{<:BlockDims{N}}) where {N} = N

similartype(::Type{<:BlockDims}, ::Type{Val{N}}) where {N} = BlockDims{N}

Base.copy(ds::BlockDims) = ds

"""
dim(::BlockDims,::Integer)

Return the total extent of the specified dimensions.
"""
function dim(ds::BlockDims{N}, i::Integer) where {N}
  return sum(ds[i])
end

"""
dims(::BlockDims)

Return the total extents of the dense space
the block dimensions live in.
"""
function dims(ds::BlockDims{N}) where {N}
  return ntuple(i -> dim(ds, i), Val(N))
end

"""
dim(::BlockDims)

Return the total extent of the dense space
the block dimensions live in.
"""
function dim(ds::BlockDims{N}) where {N}
  return prod(dims(ds))
end

"""
    nblocks(::BlockDim)

The number of blocks of the BlockDim.
"""
function nblocks(ind::BlockDim)
  return length(ind)
end

"""
    ndiagblocks(::Any)

The number of blocks along the diagonal.
"""
function ndiagblocks(x)
  return minimum(nblocks(x))
end

"""
    nblocks(::BlockDims,i::Integer)

The number of blocks in the specified dimension.
"""
function nblocks(inds::Tuple, i::Integer)
  return nblocks(inds[i])
end

"""
    nblocks(::BlockDims,is)

The number of blocks in the specified dimensions.
"""
function nblocks(inds::Tuple, is::NTuple{N,Int}) where {N}
  return ntuple(i -> nblocks(inds, is[i]), Val(N))
end

"""
    nblocks(::BlockDims)

A tuple of the number of blocks in each
dimension.
"""
function nblocks(inds::NTuple{N,<:Any}) where {N}
  return ntuple(i -> nblocks(inds, i), Val(N))
end

function eachblock(inds::Tuple)
  return (Block(b) for b in CartesianIndices(_Tuple(nblocks(inds))))
end

function eachdiagblock(inds::Tuple)
  return (Block(ntuple(_ -> i, length(inds))) for i in 1:ndiagblocks(inds))
end

"""
blockdim(::BlockDim,::Integer)

The size of the specified block in the specified
dimension.
"""
function blockdim(ind::BlockDim, i::Integer)
  return ind[i]
end

function blockdim(ind::Integer, i)
  return error(
    "`blockdim(i::Integer, b)` not currently defined for non-block index $i of type `$(typeof(i))`. In the future this may be defined for `b == Block(1)` or `b == 1` as `dim(i)` and error otherwise.",
  )
end

"""
    blockdim(::BlockDims,block,::Integer)

The size of the specified block in the specified
dimension.
"""
function blockdim(inds, block, i::Integer)
  return blockdim(inds[i], block[i])
end

"""
    blockdims(::BlockDims,block)

The size of the specified block.
"""
function blockdims(inds, block)
  return ntuple(i -> blockdim(inds, block, i), ValLength(inds))
end

"""
    blockdim(::BlockDims,block)

The total size of the specified block.
"""
function blockdim(inds, block)
  return prod(blockdims(inds, block))
end

"""
    blockdiaglength(inds::BlockDims,block)

The length of the diagonal of the specified block.
"""
function blockdiaglength(inds, block)
  return minimum(blockdims(inds, block))
end

function outer(dim1, dim2, dim3, dims...; kwargs...)
  return outer(outer(dim1, dim2), dim3, dims...; kwargs...)
end

function outer(dim1::BlockDim, dim2::BlockDim)
  dimR = BlockDim(undef, nblocks(dim1) * nblocks(dim2))
  for (i, t) in enumerate(Iterators.product(dim1, dim2))
    dimR[i] = prod(t)
  end
  return dimR
end

function permuteblocks(dim::BlockDim, perm)
  return dim[perm]
end

# Given a CartesianIndex in the range dims(T), get the block it is in
# and the index within that block
function blockindex(T, i::Vararg{Integer,N}) where {N}
  # Bounds check.
  # Do something more robust like:
  # @boundscheck Base.checkbounds_indices(Bool, map(Base.oneto, dims(T)), i) || throw_boundserror(T, i)
  @boundscheck any(iszero, i) && Base.throw_boundserror(T, i)

  # Start in the (1,1,...,1) block
  current_block_loc = @MVector ones(Int, N)
  current_block_dims = blockdims(T, Tuple(current_block_loc))
  block_index = MVector(i)
  for dim in 1:N
    while block_index[dim] > current_block_dims[dim]
      block_index[dim] -= current_block_dims[dim]
      current_block_loc[dim] += 1
      current_block_dims = blockdims(T, Tuple(current_block_loc))
    end
  end
  return Tuple(block_index), Block{N}(current_block_loc)
end

blockindex(T) = (), Block{0}()

#
# This is to help with ITensor compatibility
#

block(i::BlockDim, n::Integer) = i[n]

resize(n::Int, newdim::Int) = newdim

setblockdim!(dim1::BlockDim, newdim::Int, n::Int) = setindex!(dim1, newdim, n)

setblock!(i::BlockDim, b::Int, n::Integer) = (i[n] = b)

sim(dim::BlockDim) = copy(dim)

dir(::BlockDim) = 0

dag(dim::BlockDim) = copy(dim)
