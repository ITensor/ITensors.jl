
const QNBlock = Pair{QN,Int64}
const QNBlocks = Vector{QNBlock}

qn(qnblock::QNBlock) = qnblock.first
Tensors.blockdim(qnblock::QNBlock) = qnblock.second

Tensors.blockdim(qn::QNBlocks,b::Int) = blockdim(qn[b])

qn(qnblock::QNBlocks,b::Int) = qn(qnblock[b])

Tensors.nblocks(qns::QNBlocks) = length(qns)
function Tensors.dim(qns::QNBlocks)
  dimtot = 0
  for qn in qns
    dimtot += blockdim(qn)
  end
  return dimtot
end

const QNIndex = Index{QNBlocks}

function Index(blockdims::QNBlocks,tags=("",0))
  ts = TagSet(tags)
  return Index(rand(IDType),blockdims,Out,ts)
end


qnblocks(i::QNIndex) = i.dim

Tensors.dim(i::QNIndex) = dim(qnblocks(i))

Tensors.nblocks(i::QNIndex) = nblocks(qnblocks(i))

qn(ind::QNIndex,b::Int) = qn(qnblocks(ind),b)

Tensors.blockdim(ind::QNIndex,b::Int) = blockdim(qnblocks(ind),b)

# TODO: generic to IndexSet and BlockDims
"""
nblocks(::IndexSet,i::Integer)

The number of blocks in the specified dimension.
"""
function Tensors.nblocks(inds::IndexSet,i::Integer)
  return nblocks(inds[i])
end

# TODO: generic to IndexSet and BlockDims
"""
nblocks(::IndexSet)

A tuple of the number of blocks in each
dimension.
"""
function Tensors.nblocks(inds::IndexSet{N}) where {N}
  return ntuple(i->nblocks(inds,i),Val(N))
end

# TODO: generic to IndexSet and BlockDims
function eachblock(inds::IndexSet)
  return CartesianIndices(nblocks(inds))
end

