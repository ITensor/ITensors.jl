
# TODO: extend type restriction to `IndexT <: Union{<: Index, <: IndexVal}`
struct IndexSet{N, IndexT <: Index, DataT <: Tuple}
  data::DataT

  function IndexSet{N, IndexT, DataT}(data) where {N, IndexT, DataT <: NTuple{N, IndexT}}
    return new{N, IndexT, DataT}(data)
  end

  """
      IndexSet{Any}()

  Create a special "empty" IndexSet with data `Tuple{}` and `Any` number of indices.

  This is used as the IndexSet of an `emptyITensor()`, an ITensor with `NDTensors.Empty` storage and any number of indices.
  """
  IndexSet{Any}() = new{Any, Union{}, Tuple{}}()
end

# Definition to help with generic code
const Indices{N, IndexT} = Union{IndexSet{N,
                                          IndexT,
                                          NTuple{N, IndexT}},
                                 NTuple{N, IndexT}}

function IndexSet{N, IndexT}(inds) where {N, IndexT}
  data = NTuple{N, IndexT}(inds)
  return IndexSet{N, IndexT, typeof(data)}(data)
end

"""
    IndexSet{N, IndexT}(inds)
    IndexSet{N, IndexT}(inds::Index...)

Construct an IndexSet of order N and element type IndexT
from a collection of indices (any collection that is convertable to a Tuple).
"""
function IndexSet{N, IndexT}(inds::Index...) where {N, IndexT}
  return IndexSet{N, IndexT}(inds)
end

"""
    IndexSet{N}(inds)
    IndexSet{N}(inds::Index...)

Construct an IndexSet of order `N` from a collection of indices
(any collection that is convertable to a Tuple).
"""
IndexSet{N}(inds) where {N} = IndexSet{N, eltype(inds)}(inds)

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

Convert the collection to an IndexSet,
as long as it can be converted to a Tuple.
"""
Base.convert(::Type{IndexSet}, t) = IndexSet(t)

Base.convert(::Type{IndexSet}, is::IndexSet) = is

Base.convert(::Type{IndexSet{N}}, t) where {N} = IndexSet{N}(t)

Base.convert(::Type{IndexSet{N}}, is::IndexSet{N}) where {N} = is

Base.convert(::Type{IndexSet{N,IndexT}}, t) where {N,IndexT} = IndexSet{N,IndexT}(t)

Base.convert(::Type{IndexSet{N,IndexT}}, is::IndexSet{N,IndexT}) where {N,IndexT} = is

Base.Tuple(is::IndexSet) = Tuple(data(is))

"""
    IndexSet(inds::Vector{<:Index})

Convert a Vector of indices to an IndexSet.

Warning: this is not type stable, since a Vector
is dynamically sized and an IndexSet is statically sized.
Consider using the constructor `IndexSet{N}(inds::Vector)`.
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
    NDTensors.data(is::IndexSet)

Return the raw storage data for the indices.
Currently the storage is a Tuple.

This is mostly for internal usage, please
contact us if there is functionality you want
availabe for IndexSet.
"""
data(is::IndexSet) = is.data

# This is used in type promotion in the Tensor contraction code
Base.promote_rule(::Type{<:IndexSet},
                  ::Type{Val{N}}) where {N} = IndexSet{N}

NDTensors.ValLength(::Type{<:IndexSet{N}}) where {N} = Val{N}

NDTensors.ValLength(::IndexSet{N}) where {N} = Val(N)

# Convert to an Index if there is only one
function Index(is::IndexSet)
  length(is) != 1 && error("Number of Index in IndexSet ≠ 1")
  return is[1]
end

"""
    getindex(is::IndexSet, n::Int)

Get the Index of the IndexSet in the dimension n.
"""
Base.getindex(is::IndexSet, n) = getindex(data(is), n)

"""
    getindex(is::IndexSet, v::AbstractVector)

Get the indices of the IndexSet in the dimensions
specified in the collection v, returning an IndexSet.

Warning: this function is not type stable.
"""
Base.getindex(is::IndexSet,
              v::AbstractVector) = IndexSet(getindex(data(is), v))

"""
    setindex(is::IndexSet, i::Index, n::Int)

Set the Index of the IndexSet in the dimension n.

This function is mostly for internal use, if you want to
replace the indices of an IndexSet, use the `replaceind`,
`replaceind!`, `replaceinds`, and `replaceinds!` functions,
which check for compatibility of the indices and ensure
the proper Arrow directions.
"""
function Base.setindex(is::IndexSet,
                       i::Index,
                       n::Int)
  return IndexSet(setindex(data(is), i, n))
end

"""
    length(is::IndexSet)

The number of indices in the IndexSet.
"""
Base.length(::IndexSet{N}) where {N} = N

"""
length(::Type{<:IndexSet})

The number of indices in the IndexSet type.
"""
Base.length(::Type{<:IndexSet{N}}) where {N} = N

"""
    size(is::IndexSet)

The size of the IndexSet, the same as `(length(is),)`.

Mostly for internal use for compatability with Base methods,
like for broadcasting.
"""
Base.size(is::IndexSet) = size(data(is))

"""
    axes(is::IndexSet)

The axes of the IndexSet, the same as `(Base.OneTo(length(is)),)`.

Mostly for internal use for compatability with Base methods,
like for broadcasting.
"""
Base.axes(is::IndexSet) = axes(data(is))

NDTensors.dims(is::IndexSet{N}) where {N} = dims(Tuple(is))

NDTensors.dims(is::NTuple{N,<:Index}) where {N} = ntuple(i->dim(is[i]),Val(N))

"""
    dim(is::IndexSet)

Get the product of the dimensions of the indices
of the IndexSet (the total dimension of the space).
"""
NDTensors.dim(is::IndexSet) = dim(Tuple(is))

NDTensors.dim(is::IndexSet{0}) = 1

NDTensors.dim(is::Tuple{Vararg{<:Index}}) = prod(dims(is))

"""
    dim(is::IndexSet, n::Int)

Get the dimension of the Index n of the IndexSet.
"""
NDTensors.dim(is::IndexSet, pos::Int) = dim(is[pos])

"""
    NDTensors.strides(is::IndexSet)

Get the strides of the n-dimensional tensor if it had
this IndexSet.
"""
NDTensors.strides(is::IndexSet) = Base.size_to_strides(1, dims(is)...)

"""
    NDTensors.stride(is::IndexSet. i::Int)

Get the stride of the n-dimensional tensor if it had
this IndexSet in the dimension `i`.
"""
NDTensors.stride(is::IndexSet, k::Int) = NDTensors.strides(is)[k]

"""
    dag(is::IndexSet)

Return a new IndexSet with the indices daggered (flip
all of the arrow directions).
"""
dag(is::IndexSet) = map(i -> dag(i), is)

"""
    iterate(is::IndexSet[, state])

Iterate over the indices of an IndexSet.

# Example
```jldoctest
julia> using ITensors;

julia> i = Index(2);

julia> is = IndexSet(i,i');

julia> for j in is
         println(plev(j))
       end
0
1
```
"""
Base.iterate(is::IndexSet, state) = iterate(data(is), state)

Base.iterate(is::IndexSet) = iterate(data(is))

# To allow for the syntax is[end]
Base.firstindex(::IndexSet) = 1

# To allow for the syntax is[begin]
Base.lastindex(is::IndexSet) = length(is)

"""
    eltype(::IndexSet)

Get the element type of the IndexSet.
"""
Base.eltype(is::IndexSet{<: Any, IndexT}) where {IndexT} = IndexT

Base.eltype(::Type{<: IndexSet{<: Any,
                               IndexT}}) where {IndexT} = IndexT

# Needed for findfirst (I think)
Base.keys(is::IndexSet{N}) where {N} = 1:N

# This is to help with some generic programming in the Tensor
# code (it helps to construct an IndexSet(::NTuple{N,Index}) where the 
# only known thing for dispatch is a concrete type such
# as IndexSet{4})
NDTensors.similar_type(::Type{<:IndexSet},
                       ::Val{N}) where {N} = IndexSet{N}

NDTensors.similar_type(::Type{<:IndexSet},
                       ::Type{Val{N}}) where {N} = IndexSet{N}

"""
    sim(is::IndexSet)

Make a new IndexSet with similar indices.

You can also use the broadcast version `sim.(is)`.
"""
sim(is::IndexSet) = map(i -> sim(i), is)

# For generic code in NDTensors
NDTensors.sim(is::IndexSet) = sim(is)

import .NDTensors: mindim
"""
    mindim(is::IndexSet)

Get the minimum dimension of the indices in the index set.

Returns 1 if the IndexSet is empty.
"""
function mindim(is::IndexSet)
  length(is) == 0 && (return 1)
  md = dim(is[1])
  for n in 2:length(is)
    md = min(md, dim(is[n]))
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

IndexSet equality (order dependent). For order
independent equality use `issetequal` or
`hassameinds`.
"""
function Base.:(==)(A::IndexSet, B::IndexSet)
  length(A) ≠ length(B) && return false
  for (a,b) in zip(A,B)
    a ≠ b && return false
  end
  return true
end

"""
    ITensors.fmatch(pattern) -> ::Function

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
    ITensors.fmatch(; tags=nothing,
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

function Base.setdiff(f::Function,
                      A::IndexSet,
                      Bs::IndexSet...)
  R = eltype(A)[]
  for a ∈ A
    f(a) && all(B -> a ∉ B, Bs) && push!(R, a)
  end
  return R
end

"""
    setdiff(A::IndexSet, Bs::IndexSet...)

Output the IndexSet with Indices in `A` but not in
the IndexSets `Bs`.
"""
Base.setdiff(A::IndexSet,
             Bs::IndexSet...;
             kwargs...) = setdiff(fmatch(; kwargs...), A, Bs...)

function firstsetdiff(f::Function,
                      A::IndexSet,
                      Bs::IndexSet...)
  for a in A
    f(a) && all(B -> a ∉ B, Bs) && return a
  end
  return nothing
end

"""
    firstsetdiff(A::IndexSet, Bs::IndexSet...)

Output the first Index in `A` that is not in the IndexSets `Bs`.
Otherwise, return a default constructed Index.
"""
firstsetdiff(A::IndexSet,
             Bs::IndexSet...;
             kwargs...) = firstsetdiff(fmatch(; kwargs...), A, Bs...)

"""
    intersect(f::Function, A::IndexSet, B::IndexSet)

Output the IndexSet in the intersection of `A` and `B`,
optionally filtering by the function `f`.
"""
function Base.intersect(f::Function, A::IndexSet, B::IndexSet)
  R = eltype(A)[]
  for a in A
    f(a) && a ∈ B && push!(R,a)
  end
  return R
end

"""
    intersect(A::IndexSet, B::IndexSet; kwargs...)

Output the IndexSet in the intersection of `A` and `B`,
optionally filtering by tags, prime level, etc.
"""
Base.intersect(A::IndexSet,
               B::IndexSet;
               kwargs...) = intersect(fmatch(; kwargs...), A, B)

function firstintersect(f::Function, A::IndexSet, B::IndexSet)
  for a in A
    f(a) && a ∈ B && return a
  end
  return nothing
end

"""
    firstintersect(A::IndexSet, B::IndexSet; kwargs...)

Output the Index common to `A` and `B`, optionally
filtering by tags, prime level, etc.
If more than one Index is found, throw an error.
Otherwise, return a default constructed Index.
"""
firstintersect(A::IndexSet,
               B::IndexSet;
               kwargs...) = firstintersect(fmatch(; kwargs...), A, B)

"""
    filter(f::Function, inds::IndexSet)

Filter the IndexSet by the given function (output a new
IndexSet with indices `i` for which `f(i)` returns true).

Note that this function is not type stable, since the number
of output indices is not known at compile time.
"""
Base.filter(f::Function,
            is::IndexSet) = IndexSet(filter(f, Tuple(is)))

Base.filter(is::IndexSet,
            args...; kwargs...) = filter(fmatch(args...;
                                                kwargs...), is)

# To fix ambiguity error with Base function
Base.filter(is::IndexSet,
            tags::String;
            kwargs...) = filter(fmatch(tags; kwargs...),is)

"""
    getfirst(is::IndexSet)

Return the first Index in the IndexSet. If the IndexSet
is empty, return `nothing`.
"""
function getfirst(is::IndexSet)
  length(is) == 0 && return nothing
  return first(is)
end

"""
    getfirst(f::Function, is::IndexSet)

Get the first Index matching the pattern function,
return `nothing` if not found.
"""
function getfirst(f::Function, is::IndexSet)
  for i in is
    f(i) && return i
  end
  return nothing
end

getfirst(is::IndexSet,
         args...; kwargs...) = getfirst(fmatch(args...;
                                               kwargs...),is)

Base.findall(is::IndexSet,
             args...; kwargs...) = findall(fmatch(args...;
                                                  kwargs...), is)

"""
    indexin(ais::IndexSet, bis::IndexSet)

For collections of Indices, returns the first location in 
`bis` for each value in `ais`, as a Tuple.
"""
function Base.indexin(ais::IndexSet,
                      bis::IndexSet)
  return ntuple(i -> findfirst(bis, ais[i]), Val(length(ais)))
end

Base.findfirst(is::IndexSet,
               args...; kwargs...) = findfirst(fmatch(args...;
                                                      kwargs...), is)

"""
    map(f, is::IndexSet)

Apply the function to the elements of the IndexSet,
returning a new IndexSet.
"""
Base.map(f::Function, is::IndexSet) = IndexSet(map(f, data(is)))

#
# Tagging functions
#

function prime(f::Function,
               is::IndexSet,
               args...)
  return map(i -> f(i) ? prime(i,args...) : i, is)
end

"""
    prime(A::IndexSet, plinc, ...)

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

function replaceinds(is::IndexSet, inds1, inds2)
  is1 = IndexSet(inds1)
  poss = indexin(is1, is)
  for (j, pos) in enumerate(poss)
    i1 = is[pos]
    i2 = inds2[j]
    space(i1) != space(i2) && error("Indices must have the same spaces to be replaced")
    i2 = setdir(i2, dir(i1))
    is = setindex(is, i2, pos)
  end
  return is
end

replaceind(is::IndexSet, i1::Index, i2::Index) = replaceinds(is, (i1,), (i2,))

function swapinds(is::IndexSet, inds1, inds2)
  return replaceinds(is, (inds1..., inds2...), (inds2..., inds1...))
end

swapind(is::IndexSet, i1::Index, i2::Index) = swapinds(is, (i1,), (i2,))

removeqns(is::IndexSet) = is

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
# TupleTools
#

"""
    pop(is::IndexSet)

Return a new IndexSet with the last Index removed.
"""
pop(is::IndexSet) = IndexSet(NDTensors.pop(Tuple(is))) 

# Overload the unexported NDTensors version
NDTensors.pop(is::IndexSet) = pop(is)

"""
    popfirst(is::IndexSet)

Return a new IndexSet with the first Index removed.
"""
popfirst(is::IndexSet) = IndexSet(NDTensors.popfirst(Tuple(is))) 

# Overload the unexported NDTensors version
NDTensors.popfirst(is::IndexSet) = popfirst(is)

"""
    push(is::IndexSet, i::Index)

Make a new IndexSet with the Index i inserted
at the end.
"""
push(is::IndexSet,
     i::Index) = IndexSet(NDTensors.push(data(is), i))

# Overload the unexported NDTensors version
NDTensors.push(is::IndexSet,
             i::Index) = push(is, i)

push(is::IndexSet{0},
     i::Index) = IndexSet(i)

# Overload the unexported NDTensors version
NDTensors.push(is::IndexSet{0},
             i::Index) = push(is, i)

"""
    pushfirst(is::IndexSet, i::Index)

Make a new IndexSet with the Index i inserted
at the beginning.
"""
pushfirst(is::IndexSet,
          i::Index) = IndexSet(NDTensors.pushfirst(data(is), i))

# Overload the unexported NDTensors version
NDTensors.pushfirst(is::IndexSet,
                  i::Index) = pushfirst(is, i)

pushfirst(is::IndexSet{0},
          i::Index) = IndexSet(i)

# Overload the unexported NDTensors version
NDTensors.pushfirst(is::IndexSet{0},
                  i::Index) = pushfirst(is, i)

"""
    instertat(is1::IndexSet, is2, pos::Int)

Remove the index at pos and insert the indices
is2 starting at that position.
"""
function insertat(is1::IndexSet,
                  is2,
                  pos::Int)
  return IndexSet(NDTensors.insertat(Tuple(is1),
                                   Tuple(IndexSet(is2)),
                                   pos))
end

# Overload the unexported NDTensors version
NDTensors.insertat(is1::IndexSet,
                   is2,
                   pos::Int) = insertat(is1, is2, pos)

"""
    instertafter(is1::IndexSet, is2, pos)

Insert the indices is2 after position pos.
"""
function insertafter(is::IndexSet, I...)
  return IndexSet(NDTensors.insertafter(Tuple(is), I...))
end

# Overload the unexported NDTensors version
NDTensors.insertafter(is::IndexSet,
                    I...) = insertafter(is, I...)

function deleteat(is::IndexSet, I...)
  return IndexSet(NDTensors.deleteat(Tuple(is),I...))
end

# Overload the unexported NDTensors version
NDTensors.deleteat(is::IndexSet,
                   I...) = deleteat(is, I...)

function getindices(is::IndexSet, I...)
  return IndexSet(NDTensors.getindices(Tuple(is), I...))
end

NDTensors.getindices(is::IndexSet,
                     I...) = getindices(is, I...)

#
# QN functions
#

"""
    setdirs(is::IndexSet, dirs::Arrow...)

Return a new IndexSet with indices `setdir(is[i], dirs[i])`.
"""
function setdirs(is::IndexSet{N}, dirs) where {N}
  return IndexSet(ntuple(i -> setdir(is[i], dirs[i]), Val(N)))
end

"""
    dir(is::IndexSet, i::Index)

Return the direction of the Index `i` in the IndexSet `is`.
"""
function dir(is1::IndexSet, i::Index)
  return dir(getfirst(is1, i))
end

"""
    dirs(is::IndexSet, inds)

Return a tuple of the directions of the indices `inds` in 
the IndexSet `is`.
"""
function dirs(is1::IndexSet, inds)
  return ntuple(i -> dir(is1, inds[i]), Val(length(inds)))
end

hasqns(is::IndexSet) = any(hasqns,is)

"""
    nblocks(::IndexSet, i::Int)

The number of blocks in the specified dimension.
"""
function NDTensors.nblocks(inds::IndexSet, i::Int)
  return nblocks(Tuple(inds),i)
end

function NDTensors.nblocks(inds::IndexSet, is)
  return nblocks(Tuple(inds),is)
end

"""
    nblocks(::IndexSet)

A tuple of the number of blocks in each
dimension.
"""
function NDTensors.nblocks(inds::IndexSet{N}) where {N}
  return ntuple(i->nblocks(inds,i),Val(N))
end

function NDTensors.nblocks(inds::NTuple{N,<:Index}) where {N}
  return nblocks(IndexSet(inds))
end

ndiagblocks(inds) = minimum(nblocks(inds))

# TODO: generic to IndexSet and BlockDims
function eachblock(inds::IndexSet)
  return CartesianIndices(nblocks(inds))
end

# TODO: turn this into an iterator instead
# of returning a Vector
function eachdiagblock(inds::IndexSet{N}) where {N}
  return [ntuple(_->i,Val(N)) for i in 1:ndiagblocks(inds)]
end

"""
    flux(inds::IndexSet, block::Tuple{Vararg{Int}})

Get the flux of the specified block, for example:
```
i = Index(QN(0)=>2, QN(1)=>2)
is = IndexSet(i, dag(i'))
flux(is, (1,1)) == QN(0)
flux(is, (2,1)) == QN(1)
flux(is, (1,2)) == QN(-1)
flux(is, (2,2)) == QN(0)
```
"""
function flux(inds::IndexSet, block)
  qntot = QN()
  for n in 1:length(inds)
    ind = inds[n]
    qntot += qn(ind, block[n])
  end
  return qntot
end

"""
    flux(inds::IndexSet, I::Int...)

Get the flux of the block that the specified
index falls in.
```
i = Index(QN(0)=>2, QN(1)=>2)
is = IndexSet(i, dag(i'))
flux(is, 3, 1) == QN(1)
flux(is, 1, 2) == QN(0)
```
"""
flux(inds::IndexSet,
     vals::Int...) = flux(inds, block(inds, vals...))

"""
    ITensors.block(inds::IndexSet, I::Int...)

Get the block that the specified index falls in.

This is mostly an internal function, and the interface
is subject to change.
```
i = Index(QN(0)=>2, QN(1)=>2)
is = IndexSet(i, dag(i'))
ITensors.block(is, 3, 1) == (2,1)
ITensors.block(is, 1, 2) == (1,1)
```
"""
block(inds::IndexSet,
      vals::Int...) = blockindex(inds, vals...)[2]

function Base.show(io::IO, is::IndexSet)
  print(io,"IndexSet{$(length(is))} ")
  for i in is
    print(io, i)
    print(io, " ")
  end
end

#
# Read and write
#

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

import .NDTensors.store
@deprecate store(is::IndexSet) data(is)

