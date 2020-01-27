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

function Index(qnblocks::QNBlocks,tags=("",0))
  # TODO: make this a debug check?
  have_same_qns(qnblocks) || error("When creating a QN Index, the QN blocks must have the same QNs")
  have_same_mods(qnblocks) || error("When creating a QN Index, the QN blocks must have the same mods")
  ts = TagSet(tags)
  return Index(rand(IDType),qnblocks,Out,ts)
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

# Get a list of the non-zero blocks given a desired flux
function nzblocks(qn::QN,inds::IndexSet{N}) where {N}
  blocks = NTuple{N,Int}[]
  for block in eachblock(inds)
    if flux(inds,block) == qn
      push!(blocks,Tuple(block))
    end
  end
  return blocks
end

function Base.show(io::IO,
                   i::QNIndex) 
  idstr = "$(id(i) % 1000)"
  print(io,"(dim=$(dim(i)) [")
  for s in 1:length(space(i))
    print(io,"$(space(i)[s]),")
  end
  if length(tags(i)) > 0
    print(io,"]|id=$(idstr)|$(tagstring(tags(i))))$(primestring(tags(i)))")
  else
    print(io,"]|id=$(idstr))$(primestring(tags(i)))")
  end
  print(io,"<$(dir(i))>")
end
