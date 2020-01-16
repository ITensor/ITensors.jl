export IndexSet,
       hasindex,
       hasinds,
       hassameinds,
       findindex,
       findinds,
       indexposition,
       swaptags,
       swaptags!,
       swapprime,
       swapprime!,
       mapprime,
       mapprime!,
       commoninds,
       commonindex,
       uniqueinds,
       uniqueindex,
       mindim,
       maxdim,
       push,
       permute

struct IndexSet{N}
  inds::MVector{N,Index}
  IndexSet{N}(inds::MVector{N,Index}) where {N} = new{N}(inds)
  IndexSet{0}(::MVector{0}) = new{0}(())
  IndexSet{N}(inds::NTuple{N,Index}) where {N} = new{N}(inds)
  IndexSet{0}() = new{0}(())
  IndexSet{0}(::Tuple{}) = new{0}(())
end
IndexSet(inds::MVector{N,Index}) where {N} = IndexSet{N}(inds)
IndexSet(inds::NTuple{N,Index}) where {N} = IndexSet{N}(inds)

function IndexSet(vi::Vector{Index}) 
  N = length(vi)
  return IndexSet{N}(NTuple{N,Index}(vi))
end

Tensors.inds(is::IndexSet) = is.inds

# Empty constructor
IndexSet() = IndexSet{0}()
IndexSet(::Tuple{}) = IndexSet()
IndexSet(::MVector{0}) = IndexSet()

# Construct of some size
IndexSet{N}() where {N} = IndexSet{N}(ntuple(_->Index(),Val(N)))
IndexSet(::Val{N}) where {N} = IndexSet{N}()

# Construct from various sets of indices
IndexSet{N}(inds::Vararg{Index,N}) where {N} = IndexSet{N}(NTuple{N,Index}(inds))
IndexSet(inds::Vararg{Index,N}) where {N} = IndexSet{N}(inds...)

IndexSet{N}(ivs::NTuple{N,IndexVal}) where {N} = IndexSet{N}(ntuple(i->ind(ivs[i]),Val(N)))
IndexSet(ivs::NTuple{N,IndexVal}) where {N} = IndexSet{N}(ivs)
IndexSet{N}(ivs::Vararg{IndexVal,N}) where {N} = IndexSet{N}(tuple(ivs...))
IndexSet(ivs::Vararg{IndexVal,N}) where {N} = IndexSet{N}(tuple(ivs...))

# Construct from various sets of IndexSets
IndexSet(inds::IndexSet) = inds
IndexSet(inds::IndexSet,i::Index) = IndexSet(inds...,i)
IndexSet(i::Index,inds::IndexSet) = IndexSet(i,inds...)
IndexSet(is1::IndexSet,is2::IndexSet) = IndexSet(is1...,is2...)

# This is used in type promotion in the Tensor contraction code
Base.promote_rule(::Type{<:IndexSet},::Type{Val{N}}) where {N} = IndexSet{N}

Tensors.ValLength(::Type{IndexSet{N}}) where {N} = Val{N}
Tensors.ValLength(::IndexSet{N}) where {N} = Val(N)

# TODO: make a version that accepts an arbitrary set of IndexSets
# as well as mixtures of seperate Indices and Tuples of Indices.
# Look at jointuples in the DenseTensor decomposition logic.
IndexSet(inds::NTuple{2,IndexSet}) = IndexSet(inds...)

# Convert to an Index if there is only one
Index(is::IndexSet) = length(is)==1 ? is[1] : error("Number of Index in IndexSet ≠ 1")

function Base.show(io::IO, is::IndexSet)
  for i in is.inds
    print(io,i)
    print(io," ")
  end
end

Base.getindex(is::IndexSet,n::Integer) = getindex(is.inds,n)

function Base.setindex!(is::IndexSet,i::Index,n::Integer)
  setindex!(is.inds,i,n)
  return is
end

function StaticArrays.setindex(is::IndexSet,i::Index,n::Integer)
  # TODO: should this be deepcopy?
  isR = copy(is)
  setindex!(isR.inds,i,n)
  return isR
end

Base.length(is::IndexSet{N}) where {N} = N
Base.length(::Type{IndexSet{N}}) where {N} = N
order(is::IndexSet) = length(is)
Base.copy(is::IndexSet) = IndexSet(copy(is.inds))
Tensors.dims(is::IndexSet{N}) where {N} = ntuple(i->dim(is[i]),Val(N))
Base.ndims(::IndexSet{N}) where {N} = N
Base.ndims(::Type{IndexSet{N}}) where {N} = N
Tensors.dim(is::IndexSet) = prod(dim.(is))
Tensors.dim(is::IndexSet,pos::Integer) = dim(is[pos])

function Tensors.insertat(is1::IndexSet{N1},
                          is2::IndexSet{N2},
                          pos::Integer) where {N1,N2}
  return IndexSet{N1+N2-1}(insertat(tuple(is1...),tuple(is2...),pos))
end

function StaticArrays.deleteat(is::IndexSet{N},
                               pos::Integer) where {N}
  return IndexSet{N-1}(deleteat(tuple(is...),pos))
end

# Optimize this (right own function that extracts dimensions
# with a function)
Base.strides(is::IndexSet) = Base.size_to_strides(1, dims(is)...)
Base.stride(is::IndexSet,k::Integer) = strides(is)[k]

dag(is::IndexSet) = IndexSet(dag.(is.inds))

# Allow iteration
Base.iterate(is::IndexSet{N},state::Int=1) where {N} = state > N ? nothing : (is[state], state+1)

Base.eltype(is::Type{<:IndexSet}) = Index
Base.eltype(is::IndexSet) = eltype(typeof(is))

# Needed for findfirst (I think)
Base.keys(is::IndexSet{N}) where {N} = 1:N

StaticArrays.push(is::IndexSet{N},i::Index) where {N} = IndexSet{N+1}(push(is.inds,i))
StaticArrays.pushfirst(is::IndexSet{N},i::Index) where {N} = IndexSet{N+1}(pushfirst(is.inds,i))

# TODO: this assumes there is no overlap between the sets
unioninds(is1::IndexSet{N1},is2::IndexSet{N2}) where {N1,N2} = IndexSet{N1+N2}(is1...,is2...)

# This overload is used in Tensors
Tensors.tuplecat(is1::IndexSet{N1},
                 is2::IndexSet{N2}) where {N1,N2} = IndexSet{N1+N2}(is1...,is2...)

# This is to help with some generic programming in the Tensor
# code (it helps to construct an IndexSet(::NTuple{N,Index}) where the 
# only known thing for dispatch is a concrete type such
# as IndexSet{4})
StaticArrays.similar_type(::Type{<:IndexSet},::Val{N}) where {N} = IndexSet{N}
StaticArrays.similar_type(::Type{<:IndexSet},::Type{Val{N}}) where {N} = IndexSet{N}

Tensors.sim(is::IndexSet{N}) where {N} = IndexSet{N}(ntuple(i->sim(is[i]),Val(N)))

"""
mindim(is::IndexSet)

Get the minimum dimension of the indices in the index set.

Returns 1 if the IndexSet is empty.
"""
function mindim(is::IndexSet)
  length(is) == 0 && (return 1)
  md = dim(is[1])
  for n ∈ 2:length(is)
    md = min(md,dim(is[n]))
  end
  return md
end

"""
maxdim(is::IndexSet)

Get the maximum dimension of the indices in the index set.

Returns 1 if the IndexSet is empty.
"""
function maxdim(is::IndexSet)
  length(is) == 0 && (return 1)
  md = dim(is[1])
  for n ∈ 2:length(is)
    md = max(md,dim(is[n]))
  end
  return md
end

# 
# Set operations
#

# inds has the index i
function hasindex(inds,i::Index)
  is = IndexSet(inds)
  for j ∈ is
    i==j && return true
  end
  return false
end

# Binds is subset of Ainds
function hasinds(Binds,Ainds)
  Ais = IndexSet(Ainds)
  for i ∈ Ais
    !hasindex(Binds,i) && return false
  end
  return true
end
hasinds(Binds,Ainds::Index...) = hasinds(Binds,IndexSet(Ainds...))

# Set equality (order independent)
function hassameinds(Ainds,Binds)
  Ais = IndexSet(Ainds)
  Bis = IndexSet(Binds)
  return hasinds(Ais,Bis) && length(Ais) == length(Bis)
end

"""
==(is1::IndexSet, is2::IndexSet)

IndexSet quality (order dependent)
"""
function Base.:(==)(Ais::IndexSet,Bis::IndexSet)
  length(Ais) ≠ length(Bis) && return false
  for i ∈ 1:length(Ais)
    Ais[i] ≠ Bis[i] && return false
  end
  return true
end

# Helper function for uniqueinds
# Return true if the Index is not in any
# of the input sets of indices
function _is_unique_index(j::Index,inds::T) where {T<:Tuple}
  for I in inds
    hasindex(I,j) && return false
  end
  return true
end
# Version taking one ITensor or IndexSet
function _is_unique_index(j::Index,inds)
  hasindex(inds,j) && return false
  return true
end


"""
uniqueinds(Ais,Bis...)

Output the IndexSet with Indices in Ais but not in
the IndexSets Bis.
"""
function uniqueinds(Ainds,Binds)
  Ais = IndexSet(Ainds)
  Cis = IndexSet()
  for j ∈ Ais
    _is_unique_index(j,Binds) && (Cis = push(Cis,j))
  end
  return Cis
end

"""
uniqueindex(Ais,Bis)

Output the Index in Ais but not in the IndexSets Bis.
Otherwise, return a default constructed Index.

In the future, this may throw an error if more than 
one Index is found.
"""
function uniqueindex(Ainds,Binds)
  Ais = IndexSet(Ainds)
  for j ∈ Ais
    _is_unique_index(j,Binds) && return j
  end
  return nothing
end
# This version can check for repeats, but is a bit
# slower because of IndexSet allocation
#uniqueindex(Ais,Bis) = Index(uniqueinds(Ais,Bis)) 

Base.setdiff(Ais::IndexSet,Bis) = uniqueinds(Ais,Bis)

"""
commoninds(Ais,Bis)

Output the IndexSet in the intersection of Ais and Bis
"""
function commoninds(Ainds,Binds)
  Ais = IndexSet(Ainds)
  Cis = IndexSet()
  for i ∈ Ais
    hasindex(Binds,i) && (Cis = push(Cis,i))
  end
  return Cis
end

"""
commonindex(Ais,Bis)

Output the Index common to Ais and Bis.
If more than one Index is found, throw an error.
Otherwise, return a default constructed Index.
"""
function commonindex(Ainds,Binds)
  Ais = IndexSet(Ainds)
  for i ∈ Ais
    hasindex(Binds,i) && return i
  end
  return nothing
end
# This version checks if there are more than one indices
#commonindex(Ais,Bis) = Index(commoninds(Ais,Bis))

"""
findinds(inds,tags)

Output the IndexSet containing the subset of indices
of inds containing the tags in the input tagset.
"""
function findinds(inds,tags)
  is = IndexSet(inds)
  ts = TagSet(tags)
  found_inds = IndexSet()
  for i ∈ is
    if hastags(i,ts)
      found_inds = push(found_inds,i)
    end
  end
  return found_inds
end

"""
findindex(inds,tags)

Output the Index containing the tags in the input tagset.
If more than one Index is found, throw an error.
Otherwise, return a default constructed Index.
"""
function findindex(inds,tags)
  is = IndexSet(inds)
  ts = TagSet(tags)
  for i ∈ is
    if hastags(i,ts)
      return i
    end
  end
  # TODO: should this return `nothing` if no Index is found?
  return nothing
end
# This version checks if there are more than one indices
#findindex(inds, tags) = Index(findinds(inds,tags))

# TODO: Should this return `nothing` like `findfirst`?
# Should this just use `findfirst`?
function indexposition(is::IndexSet,
                       i::Index)
  for (n,j) in enumerate(is)
    if i==j
      return n
    end
  end
  return nothing
end

# From a tag set or index set, find the positions
# of the matching indices as a vector of integers
indexpositions(inds) = collect(1:length(inds))
indexpositions(inds, match::Nothing) = collect(1:length(inds))
#indexpositions(inds, match::Tuple{}) = collect(1:length(inds))
# Version for matching a tag set
function indexpositions(inds, match::T) where {T<:Union{AbstractString,
                                                        Tuple{<:AbstractString,<:Integer},
                                                        TagSet}}
  is = IndexSet(inds)
  tsmatch = TagSet(match)
  pos = Int[]
  for (j,I) ∈ enumerate(is)
    hastags(I,tsmatch) && push!(pos,j)
  end
  return pos
end

# Version for matching a collection of indices
function indexpositions(inds, match)
  is = IndexSet(inds)
  ismatch = IndexSet(match)
  pos = Int[]
  for (j,I) ∈ enumerate(is)
    hasindex(ismatch,I) && push!(pos,j)
  end
  return pos
end
# Version for matching a list of indices
indexpositions(inds, match_inds::Index...) = indexpositions(inds, IndexSet(match_inds...))

#
# Tagging functions
#

function prime!(is::IndexSet, plinc::Integer, match = nothing)
  pos = indexpositions(is, match)
  for jj ∈ pos
    is[jj] = prime(is[jj],plinc)
  end
  return is
end
prime!(is::IndexSet,match=nothing) = prime!(is,1,match)
prime(is::IndexSet, vargs...) = prime!(copy(is), vargs...)
# For is' notation
Base.adjoint(is::IndexSet) = prime(is)

function setprime!(is::IndexSet, plev::Integer, match = nothing)
  pos = indexpositions(is, match)
  for jj ∈ pos
    is[jj] = setprime(is[jj],plev)
  end
  return is
end
setprime(is::IndexSet, vargs...) = setprime!(copy(is), vargs...)

noprime!(is::IndexSet, match = nothing) = setprime!(is, 0, match)
noprime(is::IndexSet, vargs...) = noprime!(copy(is), vargs...)

function swapprime!(is::IndexSet, 
                    pl1::Int,
                    pl2::Int,
                    vargs...) 
  pos = indexpositions(is,vargs...)
  for n in pos
    if plev(is[n])==pl1
      is[n] = setprime(is[n],pl2)
    elseif plev(is[n])==pl2
      is[n] = setprime(is[n],pl1)
    end
  end
  return is
end

swapprime(is::IndexSet,pl1::Int,pl2::Int,vargs...) = swapprime!(copy(is),pl1,pl2,vargs...)

function mapprime!(is::IndexSet,
                   plold::Integer,
                   plnew::Integer,
                   match = nothing)
  pos = indexpositions(is,match)
  for n in pos
    if plev(is[n])==plold 
      is[n] = setprime(is[n],plnew)
    end
  end
  return is
end

function mapprime(is::IndexSet,
                  plold::Integer,
                  plnew::Integer,
                  match=nothing)
  return mapprime!(copy(is),plold,plnew,match)
end


function addtags!(is::IndexSet,
                  tags,
                  match = nothing)
  pos = indexpositions(is, match)
  for jj ∈ pos
    is[jj] = addtags(is[jj],tags)
  end
  return is
end
addtags(is, vargs...) = addtags!(copy(is), vargs...)

function settags!(is::IndexSet,
                  ts,
                  match = nothing)
  pos = indexpositions(is, match)
  for jj ∈ pos
    is[jj] = settags(is[jj],ts)
  end
  return is
end
settags(is, vargs...) = settags!(copy(is), vargs...)

function removetags!(is::IndexSet,
                     tags,
                     match = nothing)
  pos = indexpositions(is, match)
  for jj ∈ pos
    is[jj] = removetags(is[jj],tags)
  end
  return is
end
removetags(is, vargs...) = removetags!(copy(is), vargs...)

function replacetags!(is::IndexSet,
                      tags_old, tags_new,
                      match = nothing)
  pos = indexpositions(is, match)
  for jj ∈ pos
    is[jj] = replacetags(is[jj],tags_old,tags_new)
  end
  return is
end
replacetags(is, vargs...) = replacetags!(copy(is), vargs...)

# TODO: write more efficient version in terms
# of indexpositions like swapprime!
function swaptags!(is::IndexSet,
                   tags1, tags2,
                   match = nothing)
  ts1 = TagSet(tags1)
  ts2 = TagSet(tags2)
  # TODO: add debug check that this "random" tag
  # doesn't clash with ts1 or ts2
  tstemp = TagSet("e43efds")
  plev(ts1) ≥ 0 && (tstemp = setprime(tstemp,431534))
  replacetags!(is, ts1, tstemp, match)
  replacetags!(is, ts2, ts1, match)
  replacetags!(is, tstemp, ts2, match)
  return is
end
swaptags(is, vargs...) = swaptags!(copy(is), vargs...)

#
# Helper functions for contracting ITensors
#

function compute_contraction_labels(Ai::IndexSet{N1},
                                    Bi::IndexSet{N2}) where {N1,N2}
  rA = length(Ai)
  rB = length(Bi)
  Aind = MVector{N1,Int}(ntuple(_->0,Val(N1)))
  Bind = MVector{N2,Int}(ntuple(_->0,Val(N2)))

  ncont = 0
  for i = 1:rA, j = 1:rB
    if Ai[i]==Bi[j]
      Aind[i] = Bind[j] = -(1+ncont)
      ncont += 1
    end
  end

  u = ncont
  for i = 1:rA
    if(Aind[i]==0) Aind[i] = (u+=1) end
  end
  for j = 1:rB
    if(Bind[j]==0) Bind[j] = (u+=1) end
  end

  return (NTuple{N1,Int}(Aind),NTuple{N2,Int}(Bind))
end

function readcpp(io::IO,::Type{IndexSet};kwargs...)
  format = get(kwargs,:format,"v3")
  is = IndexSet()
  if format=="v3"
    size = read(io,Int)
    function readind(io,n)
      i = readcpp(io,Index;kwargs...)
      stride = read(io,UInt64)
      return i
    end
    is = IndexSet(ntuple(n->readind(io,n),size))
  else
    throw(ArgumentError("read IndexSet: format=$format not supported"))
  end
  return is
end

function HDF5.write(parent::Union{HDF5File,HDF5Group},
                    name::AbstractString,
                    is::IndexSet)
  g = g_create(parent,name)
  attrs(g)["type"] = "IndexSet"
  attrs(g)["version"] = 1
  N = length(is)
  write(g,"length",N)
  for n=1:N
    write(g,"index_$n",is[n])
  end
end

function HDF5.read(parent::Union{HDF5File,HDF5Group},
                   name::AbstractString,
                   ::Type{IndexSet})
  g = g_open(parent,name)
  if read(attrs(g)["type"]) != "IndexSet"
    error("HDF5 group or file does not contain IndexSet data")
  end
  N = read(g,"length")
  it = ntuple(n->read(g,"index_$n",Index),N)
  return IndexSet(it)
end
