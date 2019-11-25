export BlockDims,
       blockdim,
       blockdims,
       nblocks

# Used for BlockSparse Tensors
const BlockDims{N} = NTuple{N,Vector{Int}}

Base.length(ds::Type{<:BlockDims{N}}) where {N} = N

# Make the "dense" version of the indices
# For indices with QNs, this means removing the QNs
dense(ds::BlockDims) = dims(ds)
dense(::Type{<:BlockDims{N}}) where {N} = Dims{N}

Base.copy(ds::BlockDims) = ds

function dims(ds::BlockDims{N}) where {N}
  return ntuple(i->sum(ds[i]),Val(N))
end
function dim(ds::BlockDims{N}) where {N}
  return prod(dims(ds))
end

# A tuple of the number of blocks in each
# dimension
function nblocks(inds::BlockDims{N}) where {N}
  return ntuple(dim->length(inds[dim]),Val(N))
end

function nblocks(ind)
  return length(ind)
end

function blocksize(ind,i::Int) where {N}
  return ind[i]
end

# Version taking CartestianIndex
function blockdims(inds::BlockDims{N},
                   loc) where {N}
  return ntuple(dim->inds[dim][loc[dim]],Val(N))
end

# Version taking LinearIndex
function blockdims(inds::BlockDims{N},
                   loc::Int) where {N}
  # TODO: do this without conversion to CartesianIndex?
  # That may involve division and be slow?
  cartesian_loc = CartesianIndices(nblocks(inds))[loc]
  return ntuple(dim->inds[dim][cartesian_loc[dim]],Val(N))
end

function blockdim(inds::BlockDims{N},
                  loc) where {N}
  return prod(blockdims(inds,loc))
end

StaticArrays.similar_type(::Type{<:BlockDims},
                          ::Type{Val{N}}) where {N} = BlockDims{N}


