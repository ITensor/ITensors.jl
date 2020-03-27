export IndexSet,
       swaptags,
       swaptags!,
       swapprime,
       swapprime!,
       mapprime,
       mapprime!,
       getfirst,
       firstintersect,
       firstsetdiff,
       mindim,
       maxdim,
       push,
       permute

struct IndexSet{N,IndexT<:Index}
  inds::SizedVector{N,IndexT}
  IndexSet{N}(inds::SizedVector{N,IndexT}) where {N,IndexT<:Index} = new{N,IndexT}(inds)
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

# This is not defined to discourage it's use,
# since it is not type stable
#IndexSet(inds::Vector{<:Index}) = IndexSet(inds...)
IndexSet{N}(inds::Vector{<:Index}) where {N} = IndexSet{N}(inds...)

IndexSet{N,IndexT}(inds::NTuple{N,IndexT}) where {N,IndexT<:Index} = IndexSet(inds)
IndexSet{N,IndexT}(inds::Vararg{IndexT,N}) where {N,IndexT<:Index} = IndexSet(inds)

# TODO: what is this used for? Should we have this?
# It is not type stable.
function IndexSet(vi::Vector{Index}) 
  N = length(vi)
  return IndexSet{N}(NTuple{N,Index}(vi))
end

not(is::IndexSet) = Not(is)
not(inds::Index...) = not(IndexSet(inds...))
not(inds::NTuple{<:Any,<:Index}) = not(IndexSet(inds))

Tensors.store(is::IndexSet) = is.inds

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
  for i in is.inds
    print(io,i)
    print(io," ")
  end
end

Base.getindex(is::IndexSet,n::Integer) = getindex(is.inds,n)
Base.getindex(is::IndexSet,v::AbstractVector) = IndexSet(getindex(Tuple(is),v))

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
Base.length(::Type{<:IndexSet{N}}) where {N} = N
order(is::IndexSet) = length(is)
Base.copy(is::IndexSet) = IndexSet(copy(is.inds))
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

Tensors.dag(is::IndexSet) = IndexSet(dag.(is.inds))

# Allow iteration
Base.iterate(is::IndexSet{N},state::Int=1) where {N} = state > N ? nothing : (is[state], state+1)

Base.eltype(is::Type{IndexSet{N,IndexT}}) where {N,IndexT} = IndexT
Base.eltype(is::IndexSet{N,IndexT}) where {N,IndexT} = IndexT

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

"""
==(is1::IndexSet, is2::IndexSet)

IndexSet quality (order dependent)
"""
function Base.:(==)(A::IndexSet,B::IndexSet)
  length(A) ≠ length(B) && return false
  for (a,b) in zip(A,B)
    a ≠ b && return false
  end
  return true
end

fmatch(n::Not) = !fmatch(parent(n))

fmatch(::Nothing) = _->true

fmatch(is::IndexSet) = in(is)
fmatch(is::Tuple{Vararg{<:Index}}) = fmatch(IndexSet(is))
fmatch(is::Index...) = fmatch(IndexSet(is...))

fmatch(pl::Int) = hasplev(pl)

fmatch(tags::TagSet) = hastags(tags)
fmatch(tags::AbstractString) = fmatch(TagSet(tags))

fmatch(id::IDType) = hasid(id)

"""
fmatch

Return a function that accepts an Index that checks if the
Index matches the provided conditions.
"""
function fmatch(; tags=nothing,
                  plev=nothing,
                  id=nothing)
  return i -> fmatch(plev)(i) && fmatch(id)(i) && fmatch(tags)(i)
end

"""
indmatch

Checks if the Index matches the provided conditions.
"""
indmatch(i::Index; kwargs...) = fmatch(; kwargs...)(i)

const IndexCollection{IndexT<:Index} = Union{IndexSet{<:Any,IndexT},
                                             Tuple{Vararg{IndexT}},
                                             Vector{IndexT},
                                             SVector{<:Any,IndexT},
                                             MVector{<:Any,IndexT}}

function Base.setdiff(f::Function, A, Bs...)
  R = eltype(A)[]
  for a ∈ A
    f(a) && all(B -> a ∉ B, Bs) && push!(R, a)
  end
  return R
end

"""
setdiff(A,B...)

Output the IndexSet with Indices in Ais but not in
the IndexSets Bis.
"""
Base.setdiff(A::IndexCollection,
             Bs::IndexCollection...;
             kwargs...) = setdiff(fmatch(; kwargs...), A, Bs...)

function firstsetdiff(f::Function, A, Bs...)
  for a in A
    f(a) && all(B -> a ∉ B, Bs) && return a
  end
  return nothing
end

"""
firstsetdiff(A,B)

Output the Index in Ais but not in the IndexSets Bis.
Otherwise, return a default constructed Index.

In the future, this may throw an error if more than 
one Index is found.
"""
firstsetdiff(A, Bs...;
             kwargs...) = firstsetdiff(fmatch(; kwargs...), A, Bs...)

function Base.intersect(f::Function, A, B)
  R = eltype(A)[]
  for a in A
    f(a) && a ∈ B && push!(R,a)
  end
  return R
end

"""
intersect(A,B)

Output the IndexSet in the intersection of A and B
"""
Base.intersect(A::IndexCollection,
               B::IndexCollection;
               kwargs...) = intersect(fmatch(; kwargs...), A, B)

function firstintersect(f::Function, A, B)
  for a in A
    f(a) && a ∈ B && return a
  end
  return nothing
end

"""
firstintersect(Ais,Bis)

Output the Index common to Ais and Bis.
If more than one Index is found, throw an error.
Otherwise, return a default constructed Index.
"""
firstintersect(A, B;
               kwargs...) = firstintersect(fmatch(; kwargs...), A, B)

"""
filter(f::Function,inds::IndexSet)

Filter the IndexSet by the given function (output a new
IndexSet with indices `i` for which `f(i)` returns true).
"""
Base.filter(f::Function, is::IndexSet) = IndexSet(filter(f,Tuple(is)))

Base.filter(is::IndexCollection,
            args...; kwargs...) = filter(fmatch(args...;
                                                kwargs...),is)

# To fix ambiguity error with Base function
Base.filter(is::IndexCollection,
            tags::String; kwargs...) = filter(fmatch(tags; kwargs...),is)

"""
Like first, but if the length is 0 return nothing
"""
function getfirst(is)
  length(is) == 0 && return nothing
  return first(is)
end

"""
Get the first value matching the pattern function,
return nothing if not found.
"""
function getfirst(f::Function, is)
  for i in is
    f(i) && return i
  end
  return nothing
end

getfirst(is,
         args...; kwargs...) = getfirst(fmatch(args...;
                                               kwargs...),is)

Base.findall(is::IndexCollection,
             args...; kwargs...) = findall(fmatch(args...;
                                                  kwargs...), is)

Base.findfirst(is::IndexCollection,
               args...; kwargs...) = findfirst(fmatch(args...;
                                                      kwargs...), is)

#
# Tagging functions
#

function prime!(is::IndexSet, plinc::Integer, args...; kwargs...)
  pos = findall(is, args...; kwargs...)
  for jj ∈ pos
    is[jj] = prime(is[jj],plinc)
  end
  return is
end
prime!(is::IndexSet,vargs...; kwargs...) = prime!(is,1,vargs...; kwargs...)
prime(is::IndexSet,vargs...; kwargs...) = prime!(copy(is),vargs...; kwargs...)
# For is' notation
Base.adjoint(is::IndexSet) = prime(is)

function setprime!(is::IndexSet, plev::Integer, args...; kwargs...)
  pos = findall(is, args...; kwargs...)
  for jj ∈ pos
    is[jj] = setprime(is[jj],plev)
  end
  return is
end
setprime(is::IndexSet, vargs...; kwargs...) = setprime!(copy(is), vargs...; kwargs...)

noprime!(is::IndexSet, args...; kwargs...) = setprime!(is, 0, args...; kwargs...)
noprime(is::IndexSet, args...; kwargs...) = noprime!(copy(is), args...; kwargs...)

function swapprime!(is::IndexSet, 
                    pl1::Int,
                    pl2::Int,
                    args...; kwargs...) 
  pos = findall(is,args...; kwargs...)
  for n in pos
    if plev(is[n])==pl1
      is[n] = setprime(is[n],pl2)
    elseif plev(is[n])==pl2
      is[n] = setprime(is[n],pl1)
    end
  end
  return is
end

swapprime(is::IndexSet,pl1::Int,pl2::Int,args...; kwargs...) = swapprime!(copy(is),pl1,pl2,args...; kwargs...)

function mapprime!(is::IndexSet,
                   plold::Integer,
                   plnew::Integer,
                   args...; kwargs...)
  pos = findall(is, args...; kwargs...)
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
                  args...; kwargs...)
  return mapprime!(copy(is),plold,plnew,args...;kwargs...)
end


function addtags!(is::IndexSet,
                  tags,
                  args...; kwargs...)
  pos = findall(is, args...; kwargs...)
  for jj ∈ pos
    is[jj] = addtags(is[jj],tags)
  end
  return is
end
addtags(is, args...; kwargs...) = addtags!(copy(is), args...; kwargs...)

function settags!(is::IndexSet,
                  ts,
                  args...; kwargs...)
  pos = findall(is, args...; kwargs...)
  for jj ∈ pos
    is[jj] = settags(is[jj],ts)
  end
  return is
end
settags(is, args...; kwargs...) = settags!(copy(is), args...; kwargs...)

function removetags!(is::IndexSet,
                     tags,
                     args...; kwargs...)
  pos = findall(is, args...; kwargs...)
  for jj ∈ pos
    is[jj] = removetags(is[jj],tags)
  end
  return is
end
removetags(is, args...; kwargs...) = removetags!(copy(is), args...; kwargs...)

function replacetags!(is::IndexSet,
                      tags_old, tags_new,
                      args...; kwargs...)
  pos = findall(is, args...; kwargs...)
  for jj ∈ pos
    is[jj] = replacetags(is[jj],tags_old,tags_new)
  end
  return is
end
replacetags(is, args...; kwargs...) = replacetags!(copy(is), args...; kwargs...)

# TODO: write more efficient version in terms
# of findall like swapprime!
function swaptags!(is::IndexSet,
                   tags1, tags2,
                   args...; kwargs...)
  ts1 = TagSet(tags1)
  ts2 = TagSet(tags2)
  # TODO: add debug check that this "random" tag
  # doesn't clash with ts1 or ts2
  tstemp = TagSet("e43efds")
  replacetags!(is, ts1, tstemp, args...; kwargs...)
  replacetags!(is, ts2, ts1, args...; kwargs...)
  replacetags!(is, tstemp, ts2, args...; kwargs...)
  return is
end
swaptags(is, args...; kwargs...) = swaptags!(copy(is), args...; kwargs...)

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
