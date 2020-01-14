export flux

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


Tensors.dim(i::QNIndex) = dim(space(i))

Tensors.nblocks(i::QNIndex) = nblocks(space(i))

qn(ind::QNIndex,b::Int) = qn(space(ind),b)

Tensors.blockdim(ind::QNIndex,b::Int) = blockdim(space(ind),b)

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

function flux(inds::IndexSet,block)
  qntot = QN()
  for n in 1:ndims(inds)
    ind = inds[n]
    qntot += dir(ind)*qn(ind,block[n])
  end
  return qntot
end

function nzblocks(qn::QN,inds::IndexSet{N}) where {N}
  blocks = NTuple{N,Int}[]
  for block in eachblock(inds)
    if flux(inds,block) == qn
      push!(blocks,Tuple(block))
    end
  end
  return blocks
end

