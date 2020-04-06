export IndexSet,
       swaptags,
       swapprime,
       mapprime,
       mapprime!,
       getfirst,
       firstintersect,
       firstsetdiff,
       replaceind,
       replaceinds,
       mindim,
       maxdim,
       push,
       permute

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

"""
IndexSet(inds::Vector{<:Index})

Convert a Vector of indices to an IndexSet.

Note that this is not type stable, since a Vector
is dynamically sized and an IndexSet is statically sized.
"""
IndexSet(inds::Vector{<:Index}) = IndexSet(inds...)

"""
IndexSet{N}(inds::Vector{<:Index})

Convert a Vector of indices to an IndexSet of size N.

Type stable conversion of a Vector of indices to an IndexSet
(in contrast to `IndexSet(::Vector{<:Index})`).
"""
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

fmatch(::Nothing) = _ -> true

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

"""
map(f, is::IndexSet)

Apply the function to the elements of the IndexSet,
returning a new IndexSet.
"""
Base.map(f::Function, is::IndexSet) = IndexSet(map(f, store(is)))

#
# Tagging functions
#

function prime(f::Function,
               is::IndexSet,
               args...)
  return map(i -> f(i) ? prime(i,args...) : i, is)
end

"""
prime(A, plinc, ...)

Increase the prime level of the indices by the specified amount.
Filter which indices are primed using keyword arguments
tags, plev and id.
"""
prime(is::IndexSet,
      plinc::Integer,
      args...; kwargs...) = prime(fmatch(args...; kwargs...),
                                  is, plinc)

prime(f::Function,
      is::IndexSet) = prime(f, is, 1)

prime(is::IndexSet,
      args...;
      kwargs...) = prime(is, 1, args...; kwargs...)

"""
adjoint(is::IndexSet)

For is' notation.
"""
Base.adjoint(is::IndexSet) = prime(is)

function setprime(f::Function,
                  is::IndexSet,
                  args...)
  return map(i -> f(i) ? setprime(i, args...) : i, is)
end

setprime(is::IndexSet,
         plev::Integer,
         args...; kwargs...) = setprime(fmatch(args...; kwargs...),
                                        is, plev)

noprime(f::Function,
        is::IndexSet,
        args...) = setprime(is, 0, args...; kwargs...)

noprime(is::IndexSet,
        args...;
        kwargs...) = setprime(is, 0, args...; kwargs...)

function _swapprime(f::Function,
                    i::Index,
                    pl1::Int,
                    pl2::Int)
  if f(i)
    if hasplev(i, pl1)
      return setprime(i, pl2)
    elseif hasplev(i, pl2)
      return setprime(i, pl1)
    end
    return i
  end
  return i
end

function swapprime(f::Function,
                   is::IndexSet, 
                   pl1::Int,
                   pl2::Int)
  return map(i -> _swapprime(f, i, pl1, pl2), is)
end

swapprime(is::IndexSet, 
          pl1::Int,
          pl2::Int,
          args...; kwargs...) = swapprime(fmatch(args...; kwargs...),
                                          is, pl1, pl2)

function mapprime(f::Function,
                  is::IndexSet, 
                  pl1::Int,
                  pl2::Int)
  return map(i -> f(i) && hasplev(i, pl1) ? 
                  setprime(i, pl2) : i, is)
end

mapprime(is::IndexSet,
         pl1::Int,
         pl2::Int,
         args...; kwargs...) = mapprime(fmatch(args...; kwargs...),
                                        is, pl1, pl2)

function addtags(f::Function,
                 is::IndexSet,
                 args...)
  return map(i -> f(i) ? addtags(i, args...) : i, is)
end

addtags(is::IndexSet,
        tags,
        args...; kwargs...) = addtags(fmatch(args...; kwargs...),
                                      is, tags)

function settags(f::Function,
                 is::IndexSet,
                 args...)
  return map(i -> f(i) ? settags(i, args...) : i, is)
end

settags(is::IndexSet,
        tags,
        args...;
        kwargs...) = settags(fmatch(args...; kwargs...),
                             is, tags)

function removetags(f::Function,
                    is::IndexSet,
                    args...)
  return map(i -> f(i) ? removetags(i, args...) : i, is)
end

removetags(is::IndexSet,
           tags,
           args...;
           kwargs...) = removetags(fmatch(args...; kwargs...),
                                   is, tags)

function replacetags(f::Function,
                     is::IndexSet,
                     args...)
  return map(i -> f(i) ? replacetags(i, args...) : i, is)
end

replacetags(is::IndexSet,
            tags1,
            tags2,
            args...;
            kwargs...) = replacetags(fmatch(args...; kwargs...),
                                     is, tags1, tags2)

function _swaptags(f::Function,
                   i::Index,
                   tags1,
                   tags2)
  if f(i)
    if hastags(i, tags1)
      return replacetags(i, tags1, tags2)
    elseif hasteags(i, tags2)
      return replacetags(i, tags2, tags1)
    end
    return i
  end
  return i
end

function swaptags(f::Function,
                  is::IndexSet, 
                  tags1,
                  tags2)
  return map(i -> _swaptags(f, i, tags1, tags2), is)
end

swaptags(is::IndexSet,
         tags1,
         tags2,
         args...;
         kwargs...) = swaptags(fmatch(args...; kwargs...),
                               is, tags1, tags2)

function replaceind(is::IndexSet, i1::Index, i2::Index)
  space(i1) != space(i2) && error("Indices must have the same spaces to be replaced")
  pos = findfirst(is, i1)
  isnothing(pos) && error("Index not found")
  i2 = setdir(i2, dir(is[pos]))
  return setindex(is, i2, pos)
end

function replaceinds(is::IndexSet, is1, is2)
  poss = findall(is,is1)
  for (j,pos) ∈ enumerate(poss)
    i1 = is[pos]
    i2 = is2[j]
    i2 = setdir(i2, dir(i1))
    is = setindex(is, i2, pos)
  end
  return is
end

Tensors.dense(::Type{<:IndexSet}) = IndexSet

Tensors.dense(is::IndexSet) = IndexSet(dense(is...))

Tensors.dense(inds::Index...) = inds

#
# Helper functions for contracting ITensors
#

function compute_contraction_labels(Ais::IndexSet{NA},
                                    Bis::IndexSet{NB}) where {NA,NB}
  Alabels = MVector{NA,Int}(ntuple(_->0,Val(NA)))
  Blabels = MVector{NB,Int}(ntuple(_->0,Val(NB)))

  ncont = 0
  for i = 1:NA, j = 1:NB
    if Ais[i] == Bis[j]
      Alabels[i] = Blabels[j] = -(1+ncont)
      ncont += 1
    end
  end

  u = ncont
  for i = 1:NA
    if(Alabels[i]==0) Alabels[i] = (u+=1) end
  end
  for j = 1:NB
    if(Blabels[j]==0) Blabels[j] = (u+=1) end
  end

  return (Tuple(Alabels),Tuple(Blabels))
end

function compute_contraction_labels(Cis::IndexSet{NC},
                                    Ais::IndexSet{NA},
                                    Bis::IndexSet{NB}) where {NC,NA,NB}
  Alabels,Blabels = compute_contraction_labels(Ais, Bis)
  Clabels = MVector{NC,Int}(ntuple(_->0,Val(NC)))
  for i = 1:NC
    locA = findfirst(==(Cis[i]), Ais)
    if !isnothing(locA)
      Clabels[i] = Alabels[locA]
    else
      locB = findfirst(==(Cis[i]), Bis)
      Clabels[i] = Blabels[locB]
    end
  end
  return (Tuple(Clabels),Alabels,Blabels)
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
