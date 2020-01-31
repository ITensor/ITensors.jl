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

function Base.:-(qnb::QNBlock)
  return QNBlock(-qn(qnb),blockdim(qnb))
end

function Base.:+(qn1::QNBlock,qn2::QNBlock)
  qn(qn1) != qn(qn2) && error("Cannot add qn blocks with different qns")
  return QNBlock(qn(qn1),blockdim(qn1)+blockdim(qn2))
end

function Base.:-(qns::QNBlocks)
  qns_new = copy(qns)
  for i in 1:length(qns_new)
    qns_new[i] = -qns_new[i]
  end
  return qns_new
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
  return nblocks(Tuple(inds),i)
end

function Tensors.nblocks(inds::IndexSet,is)
  return nblocks(Tuple(inds),is)
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

function Tensors.nblocks(inds::NTuple{N,QNIndex}) where {N}
  return nblocks(IndexSet(inds))
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
function Tensors.nzblocks(qn::QN,inds::IndexSet{N}) where {N}
  blocks = NTuple{N,Int}[]
  for block in eachblock(inds)
    if flux(inds,block) == qn
      push!(blocks,Tuple(block))
    end
  end
  return blocks
end

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

function Tensors.outer(qn1::QNBlocks,qn2::QNBlocks)
  qnR = ITensors.QNBlocks(undef,nblocks(qn1)*nblocks(qn2))
  for (i,t) in enumerate(Iterators.product(qn1,qn2))
    qnR[i] = prod(t)
  end
  return qnR
end

function Tensors.outer(i1::QNIndex,i2::QNIndex)
  iR = Index((dir(i1)*qnblocks(i1))⊗(dir(i2)*qnblocks(i2)))
  return iR
end

function isless(qnb1::QNBlock, qnb2::QNBlock)
  return isless(qn(qnb1),qn(qnb2))
end

# Combine neighboring blocks that are the same
#function combineqns_sorted(qns::QNBlocks)
#end

function Tensors.permuteblocks(i::QNIndex,perm)
  qnblocks_perm = qnblocks(i)[perm]
  return replaceqns(i,qnblocks_perm)
end

function combineblocks(qns::QNBlocks)
  perm = sortperm(qns)
  qnsP = qns[perm]
  qnsC = [qnsP[1]]
  comb = Vector{Int}(undef,nblocks(qns))

  # Which block this is, after combining
  block_count = 1
  comb[1] = block_count
  for i in 2:nblocks(qnsP)
    if qn(qnsP[i]) == qn(qnsP[i-1])
      qnsC[block_count] += qnsP[i]
    else
      push!(qnsC,qnsP[i])
      block_count += 1
    end
    comb[i] = block_count
  end
  return qnsC,perm,comb
end

# Make a new Index with the specified qn blocks
function replaceqns(i::QNIndex,qns::QNBlocks)
  return Index(id(i),qns,dir(i),tags(i))
end

function combineblocks(i::QNIndex)
  qnsR,perm,comb = combineblocks(qnblocks(i))
  iR = replaceqns(i,qnsR)
  return iR,perm,comb
end

function Base.show(io::IO,
                   i::QNIndex)
  idstr = "$(id(i) % 1000)"
  if length(tags(i)) > 0
    print(io,"($(dim(i))|id=$(idstr)|$(tagstring(tags(i))))$(primestring(tags(i)))")
  else
    print(io,"($(dim(i))|id=$(idstr))$(primestring(tags(i)))")
  end
  println(io," <$(dir(i))>")
  for (n,qnblock) in enumerate(qnblocks(i))
    println(io," $n: $qnblock")
  end
end

