export BlockDims,
       blockdim,
       blockdims,
       nblocks

"""
BlockDim

An index for a BlockSparseTensor.
"""
const BlockDim = Vector{Int}

"""
BlockDims{N}

Dimensions used for BlockSparse Tensors.
Each entry lists the block sizes in each dimension.
"""
const BlockDims{N} = NTuple{N,BlockDim}

Base.ndims(ds::Type{<:BlockDims{N}}) where {N} = N

StaticArrays.similar_type(::Type{<:BlockDims},
                          ::Type{Val{N}}) where {N} = BlockDims{N}

"""
dense(::BlockDims) -> Dims

Make the "dense" version of the block indices.
"""
dense(ds::BlockDims) = dims(ds)
dense(::Type{<:BlockDims{N}}) where {N} = Dims{N}

Base.copy(ds::BlockDims) = ds

"""
dim(::BlockDims,::Integer)

Return the total extent of the specified dimensions.
"""
function dim(ds::BlockDims{N},i::Integer) where {N}
  return sum(ds[i])
end

"""
dims(::BlockDims)

Return the total extents of the dense space
the block dimensions live in.
"""
function dims(ds::BlockDims{N}) where {N}
  return ntuple(i->dim(ds,i),Val(N))
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
nblocks(::BlockDims,i::Integer)

The number of blocks in the specified dimension.
"""
function nblocks(inds::BlockDims,i::Integer)
  return nblocks(inds[i])
end

"""
nblocks(::BlockDims)

A tuple of the number of blocks in each
dimension.
"""
function nblocks(inds::BlockDims{N}) where {N}
  return ntuple(i->nblocks(inds,i),Val(N))
end

"""
blockdim(::BlockDim,::Integer)

The size of the specified block in the specified
dimension.
"""
function blockdim(ind::BlockDim,
                  i::Integer)
  return ind[i]
end

"""
blockdim(::BlockDims,block,::Integer)

The size of the specified block in the specified
dimension.
"""
function blockdim(inds::BlockDims{N},
                  block,
                  i::Integer) where {N}
  return blockdim(inds[i],block[i])
end

"""
blockdims(::BlockDims,block)

The size of the specified block.
"""
function blockdims(inds::BlockDims{N},
                   block) where {N}
  return ntuple(i->blockdim(inds,block,i),Val(N))
end

"""
blockdims(::BlockDims,block)

The total size of the specified block.
"""
function blockdim(inds::BlockDims{N},
                  block) where {N}
  return prod(blockdims(inds,block))
end

