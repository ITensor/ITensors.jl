export flux

const QNBlock = Pair{QN,Int64}
const QNBlocks = Vector{QNBlock}

qn(qnblock::QNBlock) = qnblock.first
Tensors.blockdim(qnblock::QNBlock) = qnblock.second

Tensors.blockdim(qnblocks::QNBlocks,b::Int) = blockdim(qnblocks[b])

qn(qnblocks::QNBlocks,b::Int) = qn(qnblocks[b])

Tensors.nblocks(qnblocks::QNBlocks) = length(qnblocks)
function Tensors.dim(qnblocks::QNBlocks)
  dimtot = 0
  for (_,blockdim) in qnblocks
    dimtot += blockdim
  end
  return dimtot
end

const QNIndex = Index{QNBlocks}

function have_same_qns(qnblocks::QNBlocks)
  qn1 = qn(qnblocks,1)
  for n in 2:nblocks(qnblocks)
    !have_same_qns(qn1,qn(qnblocks,n)) && return false
  end
  return true
end

function have_same_mods(qnblocks::QNBlocks)
  qn1 = qn(qnblocks,1)
  for n in 2:nblocks(qnblocks)
    !have_same_mods(qn1,qn(qnblocks,n)) && return false
  end
  return true
end

function Index(qnblocks::QNBlocks, dir::Arrow, tags=("",0))
  # TODO: make this a debug check?
  have_same_qns(qnblocks) || error("When creating a QN Index, the QN blocks must have the same QNs")
  have_same_mods(qnblocks) || error("When creating a QN Index, the QN blocks must have the same mods")
  ts = TagSet(tags)
  return Index(rand(IDType),qnblocks,dir,ts)
end

Index(qnblocks::QNBlocks, tags, dir::Arrow=Out) = Index(qnblocks,dir,tags)

function Index(qnblocks::QNBlocks; dir::Arrow=Out, tags=("",0))
  return Index(qnblocks,dir,tags)
end

function Index(qnblocks::QNBlock...; dir::Arrow=Out, tags=("",0))
  return Index([qnblocks...], dir, tags)
end

Tensors.dim(i::QNIndex) = dim(space(i))

Tensors.nblocks(i::QNIndex) = nblocks(space(i))

qn(ind::QNIndex,b::Int) = qn(space(ind),b)

qnblocks(ind::QNIndex) = space(ind)

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

# Get a list of the non-zero blocks given a desired flux
# TODO: make a fillqns(inds::IndexSet) function that makes all indices
# in inds have the same qns. Then, use a faster comparison:
#   ==(flux(inds,block; assume_filled=true), qn; assume_filled=true)
function nzblocks(qn::QN,inds::IndexSet{N}) where {N}
  blocks = NTuple{N,Int}[]
  for block in eachblock(inds)
    if flux(inds,block) == qn
      push!(blocks,Tuple(block))
    end
  end
  return blocks
end

#function ⊗(dim1::BlockDim,dim2::BlockDim)
#  dimR = BlockDim(undef,nblocks(dim1)*nblocks(dim2))
#  for (i,t) in enumerate(Iterators.product(dim1,dim2))
#    dimR[i] = prod(t)
#  end
#  return dimR
#end

function Base.:*(dir::Arrow, qnb::QNBlock)
  return QNBlock(dir*qn(qnb),blockdim(qnb))
end

function Base.:*(dir::Arrow, qn::QNBlocks)
  qnR = copy(qn)
  for i in 1:nblocks(qnR)
    qnR[i] = dir*qnR[i]
  end
  return qnR
end

function Base.:*(qn1::QNBlock,qn2::QNBlock)
  return QNBlock(qn(qn1)+qn(qn2),blockdim(qn1)*blockdim(qn2))
end

function Tensors.:⊗(qn1::QNBlocks,qn2::QNBlocks)
  @show qn1,qn2
  qnR = ITensors.QNBlocks(undef,nblocks(qn1)*nblocks(qn2))
  for (i,t) in enumerate(Iterators.product(qn1,qn2))
    @show i,t
    qnR[i] = prod(t)
  end
  return qnR
end

function Tensors.:⊗(i1::QNIndex,i2::QNIndex)
  @show i1,i2
  iR = Index((dir(i1)*qnblocks(i1))⊗(dir(i2)*qnblocks(i2)))
  @show iR
  return iR
end

function isless(qnb1::QNBlock, qnb2::QNBlock)
  return isless(qn(qnb1),qn(qnb2))
end

# Combine neighboring blocks that are the same
function combineqns_sorted(qns::QNBlocks)
  
end

function combineqns(qns::QNBlocks)
  perm = sortperm(qns)
  @show perm
  qnsR = qns[perm]
  return qnsR,perm
end

function replaceqns(i::QNIndex,qns::QNBlocks)
  return Index(id(i),qns,dir(i),tags(i))
end

function combineqns(i::QNIndex)
  @show qnblocks(i)
  qnsR,perm = combineqns(qnblocks(i))
  @show qnsR
  iR = replaceqns(i,qnsR)
  @show iR
  return iR,perm
end

