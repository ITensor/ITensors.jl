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
  store::NTuple{N,IndexT}
  IndexSet{N,IndexT}(inds) where {N,IndexT} = new{N,IndexT}(inds)
end

"""
IndexSet{N,IndexT}(inds)
IndexSet{N,IndexT}(inds::Index...)

Construct an IndexSet of order N and element type IndexT
from a collection of indices (any collection that is convertable to a Tuple).
"""
IndexSet{N,IndexT}(inds::Index...) where {N,IndexT} = IndexSet{N,IndexT}(inds)

"""
IndexSet{N}(inds)
IndexSet{N}(inds::Index...)

Construct an IndexSet of order N from a collection of indices
(any collection that is convertable to a Tuple).
"""
IndexSet{N}(inds) where {N} = IndexSet{N,eltype(inds)}(inds)

IndexSet{N}(inds::Index...) where {N} = IndexSet{N}(inds)

IndexSet{N}(is::IndexSet{N}) where {N} = is

"""
IndexSet(inds)
IndexSet(inds::Index...)

Construct an IndexSet from a collection of indices
(any collection that is convertable to a Tuple).
"""
IndexSet(inds) = IndexSet{length(inds)}(inds)

IndexSet(inds::Index...) = IndexSet(inds)

IndexSet(is::IndexSet) = is

"""
convert(::Type{IndexSet}, t)

Convert the collection t to an IndexSet,
as long as it can be converted to an SVector.
"""
Base.convert(::Type{IndexSet}, t) = IndexSet(t)

Base.convert(::Type{IndexSet}, is::IndexSet) = is

Base.convert(::Type{IndexSet{N}}, t) where {N} = IndexSet{N}(t)

Base.convert(::Type{IndexSet{N}}, is::IndexSet{N}) where {N} = is

Base.convert(::Type{IndexSet{N,IndexT}}, t) where {N,IndexT} = IndexSet{N,IndexT}(t)

Base.convert(::Type{IndexSet{N,IndexT}}, is::IndexSet{N,IndexT}) where {N,IndexT} = is

Base.Tuple(is::IndexSet) = Tuple(store(is))

"""
IndexSet(inds::Vector{<:Index})

Convert a Vector of indices to an IndexSet.

Warning: this is not type stable, since a Vector
is dynamically sized and an IndexSet is statically sized.
Consider using the constructor IndexSet{N}(inds::Vector).
"""
IndexSet(inds::Vector{<:Index}) = IndexSet(inds...)

"""
IndexSet{N}(inds::Vector{<:Index})

Convert a Vector of indices to an IndexSet of size N.

Type stable conversion of a Vector of indices to an IndexSet
(in contrast to `IndexSet(::Vector{<:Index})`).
"""
IndexSet{N}(inds::Vector{<:Index}) where {N} = IndexSet{N}(inds...)

"""
not(inds::IndexSet)
not(inds::Index...)
not(inds::Tuple{Vararg{<:Index}})

Represents the set of indices not in the specified
IndexSet, for use in pattern matching (i.e. when
searching for an index or priming/tagging specified
indices).
"""
not(is::IndexSet) = Not(is)
not(inds::Index...) = not(IndexSet(inds...))
not(inds::Tuple{Vararg{<:Index}}) = not(IndexSet(inds))

"""
store(is::IndexSet)

Return the raw storage data for the indices.
Currently the storage is an SVector (a statically
sized immutable vector, similar to a Tuple).

This is mostly for internal usage.
"""
Tensors.store(is::IndexSet) = is.store

# This is used in type promotion in the Tensor contraction code
Base.promote_rule(::Type{<:IndexSet},::Type{Val{N}}) where {N} = IndexSet{N}

Tensors.ValLength(::Type{<:IndexSet{N}}) where {N} = Val{N}

Tensors.ValLength(::IndexSet{N}) where {N} = Val(N)

"""
pop(is::IndexSet)

Return a new IndexSet with the last Index removed.
"""
Tensors.pop(is::IndexSet) = IndexSet(pop(store(is))) 

"""
popfirst(is::IndexSet)

Return a new IndexSet with the first Index removed.
"""
Tensors.popfirst(is::IndexSet) = IndexSet(popfirst(store(is))) 

# Convert to an Index if there is only one
Index(is::IndexSet) = length(is)==1 ? is[1] : error("Number of Index in IndexSet ≠ 1")

function Base.show(io::IO, is::IndexSet)
  for i in store(is)
    print(io,i)
    print(io," ")
  end
end

"""
getindex(is::IndexSet, n::Int)

Get the Index of the IndexSet in the dimension n.
"""
Base.getindex(is::IndexSet,
              n::Int) = getindex(store(is),n)

"""
getindex(is::IndexSet, v::AbstractVector)

Get the indices of the IndexSet in the dimensions
specified in the collection v, returning an IndexSet.

Warning: this function is not type stable.
"""
Base.getindex(is::IndexSet,
              v::AbstractVector) = IndexSet(getindex(Tuple(is),v))

function setindex(is::IndexSet,
                  i::Index,
                  n::Integer)
  return IndexSet(setindex(store(is),i,n))
end

Base.length(is::IndexSet{N}) where {N} = N

Base.length(::Type{<:IndexSet{N}}) where {N} = N

order(is::IndexSet) = length(is)

Tensors.dims(is::IndexSet{N}) where {N} = ntuple(i->dim(is[i]),Val(N))

Base.ndims(::IndexSet{N}) where {N} = N

Base.ndims(::Type{<:IndexSet{N}}) where {N} = N

"""
dim(is::IndexSet)

Get the product of the dimensions of the indices
of the IndexSet (the total dimension of the space).
"""
Tensors.dim(is::IndexSet) = prod(dims(is))

Tensors.dim(is::IndexSet{0}) = 1

"""
dim(is::IndexSet, n::Int)

Get the dimension of the Index n of the IndexSet.
"""
Tensors.dim(is::IndexSet, pos::Int) = dim(is[pos])

# To help with generic code in Tensors
Base.ndims(::NTuple{N,<:Index}) where {N} = N

Base.ndims(::Type{<:NTuple{N,<:Index}}) where {N} = N

#Tensors.dim(::Tuple{Vararg{<:IndexT}})

"""
instertat(is1::IndexSet, is2, pos)

Remove the index at pos and insert the indices
is2 starting at that position.
"""
function Tensors.insertat(is1::IndexSet,
                          is2,
                          pos::Integer)
  return IndexSet(insertat(Tuple(is1), Tuple(IndexSet(is2)), pos))
end

"""
instertafter(is1::IndexSet, is2, pos)

Insert the indices is2 after position pos.
"""
function insertafter(is::IndexSet, I...)
  return IndexSet(insertafter(Tuple(is), I...))
end

function deleteat(is::IndexSet, I...)
  return IndexSet(deleteat(Tuple(is),I...))
end

function getindices(is::IndexSet, I...)
  return IndexSet(getindices(Tuple(is), I...))
end

"""
strides(is::IndexSet)

Get the strides of the IndexSet.
"""
Tensors.strides(is::IndexSet) = Base.size_to_strides(1, dims(is)...)

"""
stride(is::IndexSet. i::Int)

Get the stride of the IndexSet in the dimension i.
"""
Tensors.stride(is::IndexSet, k::Int) = strides(is)[k]

"""
dag(is::IndexSet)

Return a new IndexSet with the indices daggered (flip
all of the arrow directions).
"""
dag(is::IndexSet) = map(i -> dag(i), is)

"""
iterate(is::IndexSet[, state])

Iterate over the indices of an IndexSet.
"""
Base.iterate(is::IndexSet, state) = iterate(store(is), state)

Base.iterate(is::IndexSet) = iterate(store(is))

Base.eltype(is::Type{IndexSet{N,IndexT}}) where {N,IndexT} = IndexT

"""
eltype(::IndexSet)

Get the element type of the IndexSet.
"""
Base.eltype(is::IndexSet{N,IndexT}) where {N,IndexT} = IndexT

# Needed for findfirst (I think)
Base.keys(is::IndexSet{N}) where {N} = 1:N

"""
push(is::IndexSet, i::Index)

Make a new IndexSet with the Index i inserted
at the end.
"""
push(is::IndexSet,
     i::Index) = IndexSet(push(store(is), i))

push(is::IndexSet{0},
     i::Index) = IndexSet(i)

"""
pushfirst(is::IndexSet, i::Index)

Make a new IndexSet with the Index i inserted
at the beginning.
"""
pushfirst(is::IndexSet,
          i::Index) = IndexSet(pushfirst(store(is), i))

pushfirst(is::IndexSet{0},
          i::Index) = IndexSet(i)

# This is to help with some generic programming in the Tensor
# code (it helps to construct an IndexSet(::NTuple{N,Index}) where the 
# only known thing for dispatch is a concrete type such
# as IndexSet{4})
Tensors.similar_type(::Type{<:IndexSet},
                     ::Val{N}) where {N} = IndexSet{N}

Tensors.similar_type(::Type{<:IndexSet},
                     ::Type{Val{N}}) where {N} = IndexSet{N}

"""
sim(is::IndexSet)

Make a new IndexSet with similar indices.
"""
sim(is::IndexSet) = map(i -> sim(i), is)

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

"""
fmatch(pattern) -> ::Function

fmatch is an internal function that
creates a function that accepts an Index.
The function returns true if the Index matches
the provided pattern, and false otherwise.

For example:
```
i = Index(2, "s")
fmatch("s")(i) == true
```
"""
fmatch(is::IndexSet) = in(is)
fmatch(is::Tuple{Vararg{<:Index}}) = fmatch(IndexSet(is))
fmatch(is::Index...) = fmatch(IndexSet(is...))

fmatch(pl::Int) = hasplev(pl)

fmatch(tags::TagSet) = hastags(tags)
fmatch(tags::AbstractString) = fmatch(TagSet(tags))

fmatch(id::IDType) = hasid(id)

fmatch(n::Not) = !fmatch(parent(n))

fmatch(::Nothing) = _ -> true

"""
fmatch(; tags=nothing,
         plev=nothing,
         id=nothing) -> ::Function

An internal function that returns a function 
that accepts an Index that checks if the
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
