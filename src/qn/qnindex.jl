
const QNBlock = Pair{QN,Int64}

const QNBlocks = Vector{QNBlock}

qn(qnblock::QNBlock) = qnblock.first

# Get the dimension of the specified block
blockdim(qnblock::QNBlock) = qnblock.second

# Get the dimension of the specified block
blockdim(qnblocks::QNBlocks, b::Integer) = blockdim(qnblocks[b])
blockdim(qnblocks::QNBlocks, b::Block{1}) = blockdim(qnblocks[only(b)])

# Get the QN of the specified block
qn(qnblocks::QNBlocks, b::Integer) = qn(qnblocks[b])
qn(qnblocks::QNBlocks, b::Block{1}) = qn(qnblocks[only(b)])

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
                                             plev::Integer = 0)

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
  return Index(rand(index_id_rng(), IDType), qnblocks, dir, tags, plev)
end

"""
    Index(qnblocks::Vector{Pair{QN, Int64}}, tags; dir::Arrow = Out,
                                                   plev::Integer = 0)

Construct a QN Index from a Vector of pairs of QN and block 
dimensions.

# Example
```
Index([QN("Sz", -1) => 1, QN("Sz", 1) => 1], "i"; dir = In)
```
"""
Index(qnblocks::QNBlocks, tags; dir::Arrow = Out, plev::Integer = 0) =
  Index(qnblocks; dir = dir, tags = tags, plev = plev)

"""
    Index(qnblocks::Pair{QN, Int64}...; dir::Arrow = Out,
                                        tags = "",
                                        plev::Integer = 0)

Construct a QN Index from a list of pairs of QN and block 
dimensions.

# Example
```
Index(QN("Sz", -1) => 1, QN("Sz", 1) => 1; tags = "i")
```
"""
Index(qnblocks::QNBlock...; dir::Arrow = Out, tags = "", plev = 0) =
  Index([qnblocks...]; dir = dir, tags = tags, plev = plev)

dim(i::QNIndex) = dim(space(i))

nblocks(i::QNIndex) = nblocks(space(i))

# Get the Block that the index value falls in
# For example:
# qns = [QN(0,2) => 2, QN(0,2) => 2]
# block(qns, 1) == Block(1)
# block(qns, 2) == Block(1)
# block(qns, 3) == Block(2)
# block(qns, 4) == Block(2)
function block(qns::QNBlocks, n::Int)
  tdim = 0
  for b in 1:nblocks(qns)
    tdim += blockdim(qns, Block(b))
    (n <= tdim) && return Block(b)
  end
  error("qn: QNIndexVal out of range")
  return Block(0)
end

function block(iv::Union{<:IndexVal, <:Pair{<:Index, <:Integer}})
  i = ind(iv)
  v = val(iv)
  return block(space(i), v)
end

# Get the QN of the block
qn(i::QNIndex, b::Block{1}) = qn(space(i), b)

qn(ib::Pair{<:Index, Block{1}}) = qn(first(ib), last(ib))

# XXX: deprecate the Integer version
qn(i::QNIndex, b::Integer) = qn(i, Block(b))

# Get the QN of the block the IndexVal lies in
qn(iv::Union{<:IndexVal, <:Pair{<:Index, <:Integer}}) =
  qn(ind(iv), block(iv))

flux(i::QNIndex, b::Block{1}) = dir(i) * qn(i, b)

flux(ib::Pair{<:Index, Block{1}}) = flux(first(ib), last(ib))

flux(iv::Union{<:IndexVal, <:Pair{<:Index, <:Integer}}) =
  flux(ind(iv), block(iv))

qnblocks(i::QNIndex) = space(i)

# XXX: deprecate the Integer version
blockdim(i::QNIndex, b::Block) = blockdim(space(i), b)
blockdim(i::QNIndex, b::Integer) = blockdim(i, Block(b))

eachblock(i::Index) = (Block(n) for n in 1:nblocks(i))

# Return the first block of the QNIndex with the flux q
function block(::typeof(first), ind::QNIndex, q::QN)
  for b in eachblock(ind)
    if flux(ind => b) == q
      return b
    end
  end
  error("No block found with QN equal to $q")
  return Block(0)
end

# Find the first block that matches the pattern f,
# for example `f(blockind) = qn(blockind) == target_qn`.
# `f` accepts a pair of `i => Block(n)` where `n`
# runs over `nblocks(i)`.
function findfirstblock(f, i::QNIndex)
  for b in ITensors.eachblock(i)
    if f(i => b)
      return b
    end
  end
  error("No block of Index $i matching the specified pattern.")
  return Block(0)
end

# XXX: call this simply `block` and return a Block{1}
# Deprecate this
"""
    qnblocknum(ind::QNIndex, q::QN)

Given a QNIndex `ind` and QN `q`, return the 
number of the block (from 1,...,nblocks(ind)) 
of the QNIndex having QN equal to `q`. Assumes 
all blocks of `ind` have a unique QN.
"""
function qnblocknum(ind::QNIndex, q::QN)
  for b = 1:nblocks(ind)
    if flux(ind => Block(b)) == q
      return b
    end
  end
  error("No block found with QN equal to $q")
  return 0
end

blockdim(ind::QNIndex, q::QN) = blockdim(ind, block(first, ind, q))

# XXX: deprecate in favor of blockdim
"""
    qnblockdim(ind::QNIndex, q::QN)

Given a QNIndex `ind` and QN `q`, return the 
dimension of the block of the QNIndex having 
QN equal to `q`. Assumes all blocks of `ind` 
have a unique QN.
"""
qnblockdim(ind::QNIndex, q::QN) = blockdim(ind, qnblocknum(ind, q))

(dir::Arrow * qnb::QNBlock) = QNBlock(dir * qn(qnb), blockdim(qnb))

function (dir::Arrow * qn::QNBlocks)
  # XXX use:
  # dir .* qn
  qnR = copy(qn)
  for i in 1:nblocks(qnR)
    qnR[i] = dir*qnR[i]
  end
  return qnR
end

(qn1::QNBlock * qn2::QNBlock) =
  QNBlock(qn(qn1)+qn(qn2), blockdim(qn1)*blockdim(qn2))

function outer(qn1::QNBlocks, qn2::QNBlocks)
  qnR = ITensors.QNBlocks(undef,nblocks(qn1)*nblocks(qn2))
  for (i,t) in enumerate(Iterators.product(qn1,qn2))
    qnR[i] = prod(t)
  end
  return qnR
end

function outer(i1::QNIndex, i2::QNIndex; dir = nothing, tags = "", plev::Integer = 0)
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

function outer(i::QNIndex; dir = nothing, tags = "", plev::Integer = 0)
  if isnothing(dir)
    dir = ITensors.dir(i)
  end
  newspace = dir * (ITensors.dir(i) * space(i))
  return Index(newspace;
               dir = dir,
               tags = tags,
               plev = plev)
end

isless(qnb1::QNBlock, qnb2::QNBlock) = isless(qn(qnb1), qn(qnb2))

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

function splitblocks(qns::QNBlocks)
  idim = dim(qns)
  split_qns = similar(qns, idim)
  for n in 1:idim
    b = block(qns, n)
    split_qns[n] = qn(qns, b) => 1
  end
  return split_qns
end

# Make a new Index with the specified qn blocks
replaceqns(i::QNIndex, qns::QNBlocks) = setspace(i, qns)

function setblockdim!(i::QNIndex, newdim::Integer, n::Integer)
  qns = space(i)
  qns[n] = qn(qns[n]) => newdim
  return i
end

function setblockqn!(i::QNIndex, newqn::QN, n::Integer)
  qns = space(i)
  qns[n] = newqn => blockdim(qns[n])
  return i
end

function deleteat!(i::QNIndex, pos)
  deleteat!(space(i), pos)
  return i
end

function resize!(i::QNIndex, n::Integer)
  resize!(space(i), n)
  return i
end

function combineblocks(i::QNIndex)
  qnsR,perm,comb = combineblocks(space(i))
  iR = replaceqns(i,qnsR)
  return iR,perm,comb
end

removeqns(i::QNIndex) = setdir(setspace(i, dim(i)), Neither)

function addqns(i::Index, qns::QNBlocks; dir::Arrow = Out)
  @assert dim(i) == dim(qns)
  return setdir(setspace(i, qns), dir)
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

# Check that the QNs are all the same
function hassameflux(i1::QNIndex, i2::QNIndex)
  dim_i1 = dim(i1)
  dim_i1 ≠ dim(i2) && return false
  for n in 1:dim_i1
    flux(i1 => n) ≠ flux(i2 => n) && return false
  end
  return true
end

hassameflux(::QNIndex, ::Index) = false
hassameflux(::Index, ::QNIndex) = false

# Split the blocks into blocks of size 1 with the same QNs
splitblocks(i::Index) = setspace(i, splitblocks(space(i)))

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

function HDF5.write(parent::Union{HDF5.File, HDF5.Group},
                    name::AbstractString,
                    B::QNBlocks)
  g = create_group(parent, name)
  attributes(g)["type"] = "QNBlocks"
  attributes(g)["version"] = 1
  write(g,"length",length(B))
  dims = [block[2] for block in B]
  write(g,"dims",dims)
  for n=1:length(B)
    write(g,"QN[$n]",B[n][1])
  end
end

function HDF5.read(parent::Union{HDF5.File,HDF5.Group},
                   name::AbstractString,
                   ::Type{QNBlocks})
  g = open_group(parent,name)
  if read(attributes(g)["type"]) != "QNBlocks"
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
