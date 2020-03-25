export IndexSet,
       hasindex,
       hasinds,
       hassameinds,
       findindex,
       findinds,
       indexposition,
       not,
       swaptags,
       swapprime,
       mapprime,
       commoninds,
       commonindex,
       uniqueinds,
       uniqueindex,
       mindim,
       maxdim,
       push,
       permute,
       hasqns

struct IndexSet{N,IndexT<:Index}
  store::SVector{N,IndexT}
  IndexSet{N}(inds::SVector{N,IndexT}) where {N,IndexT<:Index} = new{N,IndexT}(inds)
  IndexSet{N}(inds::MVector{N,IndexT}) where {N,IndexT<:Index} = new{N,IndexT}(inds)
  IndexSet{0}(::MVector{0}) = new{0,Index}(())
  IndexSet{N}(inds::NTuple{N,IndexT}) where {N,IndexT<:Index} = new{N,IndexT}(inds)
  IndexSet{0}() = new{0,Index}(())
  IndexSet{0}(::Tuple{}) = new{0,Index}(())
end
IndexSet(inds::SizedVector{N,<:Index}) where {N} = IndexSet{N}(inds)
IndexSet(inds::SVector{N,<:Index}) where {N} = IndexSet{N}(inds)
IndexSet(inds::MVector{N,<:Index}) where {N} = IndexSet{N}(inds)
IndexSet(inds::NTuple{N,<:Index}) where {N} = IndexSet{N}(inds)

IndexSet(inds::Vector{<:Index}) = IndexSet(inds...)
IndexSet{N}(inds::Vector{<:Index}) where {N} = IndexSet{N}(inds...)

IndexSet{N,IndexT}(inds::NTuple{N,IndexT}) where {N,IndexT<:Index} = IndexSet(inds)
IndexSet{N,IndexT}(inds::Vararg{IndexT,N}) where {N,IndexT<:Index} = IndexSet(inds)

# TODO: what is this used for? Should we have this?
# It is not type stable.
function IndexSet(vi::Vector{Index}) 
  N = length(vi)
  return IndexSet{N}(NTuple{N,Index}(vi))
end

Tensors.store(is::IndexSet) = is.store

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

Tensors.ValLength(::Type{<:IndexSet{N}}) where {N} = Val{N}
Tensors.ValLength(::IndexSet{N}) where {N} = Val(N)

StaticArrays.popfirst(is::IndexSet) = IndexSet(popfirst(store(is))) 

# TODO: make a version that accepts an arbitrary set of IndexSets
# as well as mixtures of seperate Indices and Tuples of Indices.
# Look at jointuples in the DenseTensor decomposition logic.
IndexSet(inds::NTuple{2,IndexSet}) = IndexSet(inds...)

# Convert to an Index if there is only one
Index(is::IndexSet) = length(is)==1 ? is[1] : error("Number of Index in IndexSet ≠ 1")

function Base.show(io::IO, is::IndexSet)
  for i in store(is)
    print(io,i)
    print(io," ")
  end
end

Base.getindex(is::IndexSet,n::Integer) = getindex(store(is),n)
Base.getindex(is::IndexSet,v::AbstractVector) = IndexSet(getindex(Tuple(is),v))

function StaticArrays.setindex(is::IndexSet,i::Index,n::Integer)
  return IndexSet(setindex(store(is),i,n))
end

Base.length(is::IndexSet{N}) where {N} = N
Base.length(::Type{<:IndexSet{N}}) where {N} = N
order(is::IndexSet) = length(is)
Base.copy(is::IndexSet) = IndexSet(copy(store(is)))
Tensors.dims(is::IndexSet{N}) where {N} = ntuple(i->dim(is[i]),Val(N))
Base.ndims(::IndexSet{N}) where {N} = N
Base.ndims(::Type{<:IndexSet{N}}) where {N} = N
Tensors.dim(is::IndexSet) = prod(dim.(is))
Tensors.dim(is::IndexSet{0}) = 1
Tensors.dim(is::IndexSet,pos::Integer) = dim(is[pos])

# To help with generic code in Tensors
Base.ndims(::NTuple{N,IndT}) where {N,IndT<:Index} = N
Base.ndims(::Type{<:NTuple{N,IndT}}) where {N,IndT<:Index} = N

function Tensors.insertat(is1::IndexSet,
                          is2,
                          pos::Integer)
  return IndexSet(insertat(Tuple(is1),Tuple(IndexSet(is2)),pos))
end

function Tensors.insertafter(is::IndexSet,I...)
  return IndexSet(insertafter(Tuple(is),I...))
end

function StaticArrays.deleteat(is::IndexSet,I...)
  return IndexSet(deleteat(Tuple(is),I...))
end

function Tensors.getindices(is::IndexSet,I...)
  return IndexSet(getindices(Tuple(is),I...))
end

# Optimize this (right own function that extracts dimensions
# with a function)
Base.strides(is::IndexSet) = Base.size_to_strides(1, dims(is)...)
Base.stride(is::IndexSet,k::Integer) = strides(is)[k]

Tensors.dag(is::IndexSet) = IndexSet(dag.(store(is)))

# Allow iteration
Base.iterate(is::IndexSet{N},state::Int=1) where {N} = state > N ? nothing : (is[state], state+1)

Base.eltype(is::Type{IndexSet{N,IndexT}}) where {N,IndexT} = IndexT
Base.eltype(is::IndexSet{N,IndexT}) where {N,IndexT} = IndexT

# Needed for findfirst (I think)
Base.keys(is::IndexSet{N}) where {N} = 1:N

StaticArrays.push(is::IndexSet{N},i::Index) where {N} = IndexSet{N+1}(push(store(is),i))
StaticArrays.pushfirst(is::IndexSet{N},i::Index) where {N} = IndexSet{N+1}(pushfirst(store(is),i))

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

function indexmatch(i::Index; tags=nothing, plev=nothing)
  return (isnothing(plev) || ITensors.plev(i)==plev) &&
         (isnothing(tags) || hastags(i,tags))
end

function indexmatch(i::Index,j::Index; kwargs...)
  return indexmatch(i; kwargs...) && i==j
end

# inds has the index i
function hasindex(inds,i::Index; kwargs...)
  return indexmatch(i; kwargs...) && any(==(i),IndexSet(inds))
end

# Binds is subset of Ainds
function hasinds(Binds,Ainds; kwargs...)
  for i ∈ Ainds
    !hasindex(Binds,i; kwargs...) && return false
  end
  return true
end
hasinds(Binds,
        Ainds::Index...; kwargs...) = hasinds(Binds,
                                              IndexSet(Ainds...); kwargs...)

# Set equality (order independent)
function hassameinds(Ainds,Binds)
  return hasinds(Ainds,Binds) && length(Ainds) == length(Binds)
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

"""
uniqueinds(Ais,Bis...)

Output the IndexSet with Indices in Ais but not in
the IndexSets Bis.
"""
function uniqueinds(Ainds,Binds; kwargs...)
  Ais = IndexSet(Ainds)
  Cis = IndexSet{0,eltype(Ais)}()
  for j ∈ Ais
    indexmatch(j; kwargs...) &&
      !hasindex(Binds,j) && (Cis = push(Cis,j))
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
function uniqueindex(Ainds,Binds...; kwargs...)
  Ais = IndexSet(Ainds)
  for j ∈ Ais
    indexmatch(j; kwargs...) &&
      all(x->!hasindex(x,j),Binds) && return j
  end
  return nothing
end

Base.setdiff(Ais::IndexSet,Bis; kwargs...) = uniqueinds(Ais,Bis; kwargs...)

"""
commoninds(Ais,Bis)

Output the IndexSet in the intersection of Ais and Bis
"""
function commoninds(Ainds,Binds; kwargs...)
  Ais = IndexSet(Ainds)
  Cis = IndexSet{0,eltype(Ais)}()
  for i ∈ Ais
    hasindex(Binds,i; kwargs...) && (Cis = push(Cis,i))
  end
  return Cis
end

"""
commonindex(Ais,Bis)

Output the Index common to Ais and Bis.
If more than one Index is found, throw an error.
Otherwise, return a default constructed Index.
"""
function commonindex(Ainds,Binds; kwargs...)
  Ais = IndexSet(Ainds)
  for i ∈ Ais
    hasindex(Binds,i; kwargs...) && return i
  end
  return nothing
end

"""
findinds(inds,tags)

Output the Vector of indices containing the subset of indices
of inds containing the tags in the input tagset.
"""
function findinds(inds, args...; kwargs...)
  ns = indexpositions(inds, args...; kwargs...)
  return IndexSet(inds)[ns]
end

"""
findindex(inds,tags)

Output the Index containing the tags in the input tagset.
If more than one Index is found, throw an error.
Otherwise, return a default constructed Index.
"""
function findindex(inds, args...; kwargs...)
  n = indexposition(inds, args...; kwargs...)
  isnothing(n) && return nothing
  return IndexSet(inds)[n]
end

function indexposition(inds, args...; kwargs...)
  ns = indexpositions(inds, args...; kwargs...)
  length(ns) == 0 && return nothing
  return ns[1]
end

# From a tag set or index set, find the positions
# of the matching indices as a vector of integers
#indexpositions(inds, match::Nothing) = collect(1:length(inds))
# Version for matching a tag set
function indexpositions(inds, tags::Union{AbstractString,TagSet};
                        plev=nothing)
  return indexpositions(inds; tags=tags, plev=plev)
end

function indexpositions(inds; tags=nothing, 
                              plev=nothing)
  isnothing(tags) && isnothing(plev) && return collect(1:length(inds))
  is = IndexSet(inds)
  pos = Int[]
  if isnothing(plev)
    for (j,I) ∈ enumerate(is)
        hastags(I,tags) && push!(pos,j)
    end
  elseif isnothing(tags)
    for (j,I) ∈ enumerate(is)
      ITensors.plev(I)==plev && push!(pos,j)
    end
  else
    for (j,I) ∈ enumerate(is)
      ITensors.plev(I)==plev && hastags(I,tags) && push!(pos,j)
    end
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
# not syntax (to prime or tag the compliment 
# of the specified indices)
#

struct Not{T}
  pattern::T
  Not(p::T) where {T} = new{T}(p)
end

not(ts::Union{AbstractString,TagSet}) = Not(TagSet(ts))

not(is::IndexSet) = Not(is)
not(inds::Index...) = not(IndexSet(inds...))
not(inds::NTuple{<:Any,<:Index}) = not(IndexSet(inds))

function indexpositions(inds, match::Not{TagSet})
  is = IndexSet(inds)
  pos = Int[]
  for (j,I) ∈ enumerate(is)
    !hastags(I,match.pattern) && push!(pos,j)
  end
  return pos
end

function indexpositions(inds, match::Not{<:IndexSet})
  is = IndexSet(inds)
  pos = Int[]
  for (j,I) ∈ enumerate(is)
    !hasindex(match.pattern,I) && push!(pos,j)
  end
  return pos
end

#
# Tagging functions
#

function prime(is::IndexSet, plinc::Integer, args...; kwargs...)
  pos = indexpositions(is, args...; kwargs...)
  for jj ∈ pos
    is = setindex(is,prime(is[jj],plinc),jj)
  end
  return is
end
prime(is::IndexSet,vargs...; kwargs...) = prime(is,1,vargs...; kwargs...)
# For is' notation
Base.adjoint(is::IndexSet) = prime(is)

function setprime(is::IndexSet, plev::Integer, args...; kwargs...)
  pos = indexpositions(is, args...; kwargs...)
  for jj ∈ pos
    is = setindex(is,setprime(is[jj],plev),jj)
  end
  return is
end

noprime(is::IndexSet, args...; kwargs...) = setprime(is, 0, args...; kwargs...)

function swapprime(is::IndexSet, 
                   pl1::Int,
                   pl2::Int,
                   args...; kwargs...) 
  pos = indexpositions(is,args...; kwargs...)
  for n in pos
    if plev(is[n])==pl1
      is = setindex(is,setprime(is[n],pl2),n)
    elseif plev(is[n])==pl2
      is = setindex(is,setprime(is[n],pl1),n)
    end
  end
  return is
end

function mapprime(is::IndexSet,
                  plold::Integer,
                  plnew::Integer,
                  args...; kwargs...)
  pos = indexpositions(is, args...; kwargs...)
  for n in pos
    if plev(is[n])==plold 
      is = setindex(is,setprime(is[n],plnew),n)
    end
  end
  return is
end

function addtags(is::IndexSet,
                 tags,
                 args...; kwargs...)
  pos = indexpositions(is, args...; kwargs...)
  for jj ∈ pos
    is = setindex(is,addtags(is[jj],tags),jj)
  end
  return is
end

function settags(is::IndexSet,
                 ts,
                 args...; kwargs...)
  pos = indexpositions(is, args...; kwargs...)
  for jj ∈ pos
    is = setindex(is,settags(is[jj],ts),jj)
  end
  return is
end

function removetags(is::IndexSet,
                    tags,
                    args...; kwargs...)
  pos = indexpositions(is, args...; kwargs...)
  for jj ∈ pos
    is = setindex(is,removetags(is[jj],tags),jj)
  end
  return is
end

function replacetags(is::IndexSet,
                     tags_old, tags_new,
                     args...; kwargs...)
  pos = indexpositions(is, args...; kwargs...)
  for jj ∈ pos
    is = setindex(is,replacetags(is[jj],tags_old,tags_new),jj)
  end
  return is
end

# TODO: write more efficient version in terms
# of indexpositions like swapprime
function swaptags(is::IndexSet,
                  tags1, tags2,
                  args...; kwargs...)
  ts1 = TagSet(tags1)
  ts2 = TagSet(tags2)
  # TODO: add debug check that this "random" tag
  # doesn't clash with ts1 or ts2
  tstemp = TagSet("e43efds")
  is = replacetags(is, ts1, tstemp, args...; kwargs...)
  is = replacetags(is, ts2, ts1, args...; kwargs...)
  is = replacetags(is, tstemp, ts2, args...; kwargs...)
  return is
end

Tensors.dense(::Type{<:IndexSet}) = IndexSet

Tensors.dense(is::IndexSet) = IndexSet(dense(is...))

Tensors.dense(inds::Index...) = inds

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

#
# QN functions
#

hasqns(is::IndexSet) = any(hasqns,is)

function readcpp(io::IO,::Type{<:IndexSet};kwargs...)
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
                   ::Type{<:IndexSet})
  g = g_open(parent,name)
  if read(attrs(g)["type"]) != "IndexSet"
    error("HDF5 group or file does not contain IndexSet data")
  end
  N = read(g,"length")
  it = ntuple(n->read(g,"index_$n",Index),N)
  return IndexSet(it)
end
