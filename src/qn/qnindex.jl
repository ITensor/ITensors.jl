
const QNBlock = Pair{QN,Int64}

const QNBlocks = Vector{QNBlock}

qn(qnblock::QNBlock) = qnblock.first

blockdim(qnblock::QNBlock) = qnblock.second

blockdim(qnblocks::QNBlocks, b::Integer) = blockdim(qnblocks[b])

qn(qnblocks::QNBlocks,b::Integer) = qn(qnblocks[b])

nblocks(qnblocks::QNBlocks) = length(qnblocks)

function dim(qnblocks::QNBlocks)
  dimtot = 0
  for (_,blockdim) in qnblocks
    dimtot += blockdim
  end
  return dimtot
end

function -(qnb::QNBlock)
  return QNBlock(-qn(qnb),blockdim(qnb))
end

function (qn1::QNBlock + qn2::QNBlock)
  qn(qn1) != qn(qn2) && error("Cannot add qn blocks with different qns")
  return QNBlock(qn(qn1),blockdim(qn1)+blockdim(qn2))
end

function -(qns::QNBlocks)
  qns_new = copy(qns)
  for i in 1:length(qns_new)
    qns_new[i] = -qns_new[i]
  end
  return qns_new
end

"""
A QN Index is an Index with QN block storage instead of
just an integer dimension. The QN block storage is a 
vector of pairs of QNs and block dimensions.
The total dimension of a QN Index is the sum of the
dimensions of the blocks of the Index.
"""
const QNIndex = Index{QNBlocks}

const QNIndexVal = IndexVal{QNIndex}

hasqns(::QNIndex) = true

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

"""
    Index(qnblocks::Vector{Pair{QN, Int64}}; dir::Arrow = Out,
                                             tags = "",
                                             plev::Int = 0)

Construct a QN Index from a Vector of pairs of QN and block 
dimensions.

Note: in the future, this may enforce that all blocks have the
same QNs (which would allow for some optimizations, for example
when constructing random QN ITensors).

# Example
```
Index([QN("Sz", -1) => 1, QN("Sz", 1) => 1]; tags = "i")
```
"""
function Index(qnblocks::QNBlocks; dir::Arrow = Out, tags = "", plev = 0)
  # TODO: make this a debug check?
  #have_same_qns(qnblocks) || error("When creating a QN Index, the QN blocks must have the same QNs")
  #have_same_mods(qnblocks) || error("When creating a QN Index, the QN blocks must have the same mods")
  return Index(rand(IDType), qnblocks, dir, tags, plev)
end

"""
    Index(qnblocks::Vector{Pair{QN, Int64}}, tags; dir::Arrow = Out,
                                                   plev::Int = 0)

Construct a QN Index from a Vector of pairs of QN and block 
dimensions.

# Example
```
Index([QN("Sz", -1) => 1, QN("Sz", 1) => 1], "i"; dir = In)
```
"""
Index(qnblocks::QNBlocks,
      tags;
      dir::Arrow = Out,
      plev::Int = 0) = Index(qnblocks; dir = dir,
                                       tags = tags,
                                       plev = plev)

"""
    Index(qnblocks::Pair{QN, Int64}...; dir::Arrow = Out,
                                        tags = "",
                                        plev::Int = 0)

Construct a QN Index from a list of pairs of QN and block 
dimensions.

# Example
```
Index(QN("Sz", -1) => 1, QN("Sz", 1) => 1; tags = "i")
```
"""
function Index(qnblocks::QNBlock...; dir::Arrow=Out,
                                     tags="",
                                     plev=0)
  return Index([qnblocks...]; dir = dir,
                              tags = tags,
                              plev = plev)
end

dim(i::QNIndex) = dim(space(i))

nblocks(i::QNIndex) = nblocks(space(i))

qn(ind::QNIndex, b::Integer) = dir(ind)*qn(space(ind),b)

qnblocks(ind::QNIndex) = space(ind)

blockdim(ind::QNIndex, b::Integer) = blockdim(space(ind),b)

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
    qnblockdim(ind::QNIndex, q::QN)

Given a QNIndex `ind` and QN `q`, return the 
dimension of the block of the QNIndex having 
QN equal to `q`. Assumes all blocks of `ind` 
have a unique QN.
"""
qnblockdim(ind::QNIndex, q::QN) = blockdim(ind, qnblocknum(ind,q))

function (dir::Arrow * qnb::QNBlock)
  return QNBlock(dir*qn(qnb), blockdim(qnb))
end

function (dir::Arrow * qn::QNBlocks)
  qnR = copy(qn)
  for i in 1:nblocks(qnR)
    qnR[i] = dir*qnR[i]
  end
  return qnR
end

function (qn1::QNBlock * qn2::QNBlock)
  return QNBlock(qn(qn1)+qn(qn2), blockdim(qn1)*blockdim(qn2))
end

function outer(qn1::QNBlocks, qn2::QNBlocks)
  qnR = ITensors.QNBlocks(undef,nblocks(qn1)*nblocks(qn2))
  for (i,t) in enumerate(Iterators.product(qn1,qn2))
    qnR[i] = prod(t)
  end
  return qnR
end

function outer(i1::QNIndex, i2::QNIndex;
               dir = nothing,
               tags = "",
               plev::Int = 0)
  if isnothing(dir)
    if ITensors.dir(i1) == ITensors.dir(i2)
      dir = ITensors.dir(i1)
    else
      dir = Out
    end
  end
  newspace = dir * ((ITensors.dir(i1) * space(i1)) ⊗
                    (ITensors.dir(i2) * space(i2)))
  return Index(newspace;
               dir = dir,
               tags = tags,
               plev = plev)
end

function outer(i::QNIndex;
               dir = nothing,
               tags = "",
               plev::Int = 0)
  if isnothing(dir)
    dir = ITensors.dir(i)
  end
  newspace = dir * (ITensors.dir(i) * space(i))
  return Index(newspace;
               dir = dir,
               tags = tags,
               plev = plev)
end

function isless(qnb1::QNBlock, qnb2::QNBlock)
  return isless(qn(qnb1), qn(qnb2))
end

function permuteblocks(i::QNIndex, perm)
  qnblocks_perm = space(i)[perm]
  return replaceqns(i, qnblocks_perm)
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

function setblockdim!(i::QNIndex,newdim::Int,n::Int)
  qns = space(i)
  qns[n] = qn(qns[n]) => newdim
  return i
end

function setblockqn!(i::QNIndex,newqn::QN,n::Int)
  qns = space(i)
  qns[n] = newqn => blockdim(qns[n])
  return i
end

function deleteat!(i::QNIndex,pos)
  deleteat!(space(i),pos)
  return i
end

function resize!(i::QNIndex,n::Integer)
  resize!(space(i),n)
  return i
end

function combineblocks(i::QNIndex)
  qnsR,perm,comb = combineblocks(space(i))
  iR = replaceqns(i,qnsR)
  return iR,perm,comb
end

removeqns(i::QNIndex) =
  Index(id(i), dim(i), Neither, tags(i), plev(i))

function addqns(i::Index, qns::QNBlocks; dir::Arrow = Out)
  @assert dim(i) == dim(qns)
  return Index(id(i), qns, dir, tags(i), plev(i))
end
  
function addqns(i::QNIndex, qns::QNBlocks)
  @assert dim(i) == dim(qns)
  @assert nblocks(qns) == nblocks(i)
  iqns = space(i)
  j = copy(i)
  jqn = space(j)
  for n in 1:nblocks(i)
    @assert blockdim(iqns, n) == blockdim(qns, n)
    iqn_n = qn(iqns, n)
    qn_n = qn(qns, n)
    newqn = iqn_n
    for nqv in 1:nactive(qn_n)
      qv = qn_n[nqv]
      newqn = addqnval(newqn, qv)
    end
    jqn[n] = newqn => blockdim(iqns, n)
  end
  return j
end

mutable_storage(::Type{Order{N}},
                ::Type{IndexT}) where {N, IndexT <: QNIndex} =
  SizedVector{N, IndexT}(undef)

isfermionic(i::Index) = false

isfermionic(i::QNIndex) = any(q -> isfermionic(qn(q)), space(i))

function show(io::IO, i::QNIndex)
  idstr = "$(id(i) % 1000)"
  if length(tags(i)) > 0
    print(io,"(dim=$(dim(i))|id=$(idstr)|\"$(tagstring(tags(i)))\")$(primestring(plev(i)))")
  else
    print(io,"(dim=$(dim(i))|id=$(idstr))$(primestring(plev(i)))")
  end
  println(io," <$(dir(i))>")
  for (n,qnblock) in enumerate(space(i))
    print(io," $n: $qnblock")
    n < length(space(i)) && println(io)
  end
end

function HDF5.write(parent::Union{HDF5File, HDF5Group},
                    name::AbstractString,
                    B::QNBlocks)
  g = g_create(parent, name)
  attrs(g)["type"] = "QNBlocks"
  attrs(g)["version"] = 1
  write(g,"length",length(B))
  dims = [block[2] for block in B]
  write(g,"dims",dims)
  for n=1:length(B)
    write(g,"QN[$n]",B[n][1])
  end
end

function HDF5.read(parent::Union{HDF5File,HDF5Group},
                   name::AbstractString,
                   ::Type{QNBlocks})
  g = g_open(parent,name)
  if read(attrs(g)["type"]) != "QNBlocks"
    error("HDF5 group or file does not contain QNBlocks data")
  end
  N = read(g,"length")
  dims = read(g,"dims")
  B = QNBlocks(undef,N)
  for n=1:length(B)
    B[n] = QNBlock(read(g,"QN[$n]",QN),dims[n])
  end
  return B
end
