export flux,
       hasqns,
       QNIndex,
       QNIndexVal,
       qn,
       qnblockdim,
       qnblocknum

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
const QNIndexVal = IndexVal{QNIndex}

hasqns(::QNIndex) = true

QNIndex() = Index(0,Pair{QN,Int}[],Out,"",0)

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

function Index(qnblocks::QNBlocks, dir::Arrow, tags="", plev=0)
  # TODO: make this a debug check?
  #have_same_qns(qnblocks) || error("When creating a QN Index, the QN blocks must have the same QNs")
  #have_same_mods(qnblocks) || error("When creating a QN Index, the QN blocks must have the same mods")
  return Index(rand(IDType),qnblocks,dir,tags,plev)
end

Index(qnblocks::QNBlocks, tags, dir::Arrow=Out) = Index(qnblocks,dir,tags)

function Index(qnblocks::QNBlocks; dir::Arrow=Out, tags="", plev=0)
  return Index(qnblocks,dir,tags,plev)
end

function Index(qnblocks::QNBlock...; dir::Arrow=Out, tags="", plev=0)
  return Index([qnblocks...], dir, tags, plev)
end

Tensors.dim(i::QNIndex) = dim(space(i))

Tensors.nblocks(i::QNIndex) = nblocks(space(i))

qn(ind::QNIndex,b::Int) = dir(ind)*qn(space(ind),b)

qnblocks(ind::QNIndex) = space(ind)

Tensors.blockdim(ind::QNIndex,b::Int) = blockdim(space(ind),b)

function qn(iv::QNIndexVal)
  i = ind(iv)
  v = val(iv)
  tdim = 0
  for b=1:nblocks(i)
    tdim += blockdim(i,b)
    (v <= tdim) && return qn(i,b)
  end
  error("qn: QNIndexVal out of range")
  return QN()
end

"""
    qnblocknum(ind::QNIndex,q::QN)

Given a QNIndex `ind` and QN `q`, return the 
number of the block (from 1,...,nblocks(ind)) 
of the QNIndex having QN equal to `q`. Assumes 
all blocks of `ind` have a unique QN.
"""
function qnblocknum(ind::QNIndex,q::QN) 
  for b=1:nblocks(ind)
    if qn(ind,b) == q
      return b
    end
  end
  error("No block found with QN equal to $q")
  return 0
end

"""
    qnblockdim(ind::QNIndex,q::QN)

Given a QNIndex `ind` and QN `q`, return the 
dimension of the block of the QNIndex having 
QN equal to `q`. Assumes all blocks of `ind` 
have a unique QN.
"""
qnblockdim(ind::QNIndex,q::QN) = blockdim(ind,qnblocknum(ind,q))


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

function Tensors.nblocks(inds::NTuple{N,<:Index}) where {N}
  return nblocks(IndexSet(inds))
end

ndiagblocks(inds) = minimum(nblocks(inds))

# TODO: generic to IndexSet and BlockDims
function eachblock(inds::IndexSet)
  return CartesianIndices(nblocks(inds))
end

function eachdiagblock(inds::IndexSet{N}) where {N}
  return [ntuple(_->i,Val(N)) for i in 1:ndiagblocks(inds)]
end

function flux(inds::IndexSet,block)
  qntot = QN()
  for n in 1:ndims(inds)
    ind = inds[n]
    qntot += qn(ind,block[n])
  end
  return qntot
end

flux(inds::IndexSet,vals::Int...) = flux(inds,block(inds,vals...))

Tensors.block(inds::IndexSet,vals::Int...) = blockindex(inds,vals...)[2]

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

function nzdiagblocks(qn::QN,inds::IndexSet{N}) where {N}
  blocks = NTuple{N,Int}[]
  for block in eachdiagblock(inds)
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

# TODO: add a combine kwarg to choose if the QN blocks
# get sorted and combined (could do it by default?)
function Tensors.outer(i1::QNIndex,i2::QNIndex; tags="", plev=0)
  if dir(i1) == dir(i2)
    return Index(space(i1)⊗space(i2),dir(i1),tags,plev)
  else
    return Index((dir(i1)*space(i1))⊗(dir(i2)*space(i2)),Out,tags,plev)
  end
end

Tensors.outer(i::QNIndex; tags="", plev=0) = sim(i; tags=tags, plev=plev)

function isless(qnb1::QNBlock, qnb2::QNBlock)
  return isless(qn(qnb1),qn(qnb2))
end

# Combine neighboring blocks that are the same
#function combineqns_sorted(qns::QNBlocks)
#end

function Tensors.permuteblocks(i::QNIndex,perm)
  qnblocks_perm = space(i)[perm]
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
  return Index(id(i),qns,dir(i),tags(i),plev(i))
end

function Tensors.setblockdim!(i::QNIndex,newdim::Int,n::Int)
  qns = space(i)
  qns[n] = qn(qns[n]) => newdim
  return i
end

function setblockqn!(i::QNIndex,newqn::QN,n::Int)
  qns = space(i)
  qns[n] = newqn => blockdim(qns[n])
  return i
end

function Base.deleteat!(i::QNIndex,pos)
  deleteat!(space(i),pos)
  return i
end

function Base.resize!(i::QNIndex,n::Integer)
  resize!(space(i),n)
  return i
end

function combineblocks(i::QNIndex)
  qnsR,perm,comb = combineblocks(space(i))
  iR = replaceqns(i,qnsR)
  return iR,perm,comb
end

Tensors.dense(inds::QNIndex...) = dense.(inds)

Tensors.dense(i::QNIndex) = Index(id(i),dim(i),dir(i),tags(i),plev(i))

function Base.show(io::IO,
                   i::QNIndex)
  idstr = "$(id(i) % 1000)"
  if length(tags(i)) > 0
    print(io,"(dim=$(dim(i))|id=$(idstr)|\"$(tagstring(tags(i)))\")$(primestring(plev(i)))")
  else
    print(io,"(dim=$(dim(i))|id=$(idstr))$(primestring(plev(i)))")
  end
  println(io," <$(dir(i))>")
  for (n,qnblock) in enumerate(space(i))
    println(io," $n: $qnblock")
  end
end

