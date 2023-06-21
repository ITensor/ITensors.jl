# Represents a static order of an ITensor
@eval struct Order{N}
  (OrderT::Type{<:Order})() = $(Expr(:new, :OrderT))
end

@doc """
   Order{N}
A value type representing the order of an ITensor.
""" Order

"""
   Order(N) = Order{N}()
Create an instance of the value type Order representing
the order of an ITensor.
"""
Order(N) = Order{N}()

# Helpful if we want code to work generically
# for other Index-like types (such as IndexRange)
const IndexSet{IndexT<:Index} = Vector{IndexT}
const IndexTuple{IndexT<:Index} = Tuple{Vararg{IndexT}}

# Definition to help with generic code
const Indices{IndexT<:Index} = Union{Vector{IndexT},Tuple{Vararg{IndexT}}}

function _narrow_eltype(v::Vector{T}; default_empty_eltype=T) where {T}
  if isempty(v)
    return default_empty_eltype[]
  end
  return convert(Vector{mapreduce(typeof, promote_type, v)}, v)
end
function narrow_eltype(v::Vector{T}; default_empty_eltype=T) where {T}
  if isconcretetype(T)
    return v
  end
  return _narrow_eltype(v; default_empty_eltype)
end

_indices() = ()
_indices(x::Index) = (x,)

# Tuples
_indices(x1::Tuple, x2::Tuple) = (x1..., x2...)
_indices(x1::Index, x2::Tuple) = (x1, x2...)
_indices(x1::Tuple, x2::Index) = (x1..., x2)
_indices(x1::Index, x2::Index) = (x1, x2)

# Vectors
_indices(x1::Vector, x2::Vector) = narrow_eltype(vcat(x1, x2); default_empty_eltype=Index)

# Mix vectors and tuples/elements
_indices(x1::Vector, x2) = _indices(x1, [x2])
_indices(x1, x2::Vector) = _indices([x1], x2)
_indices(x1::Vector, x2::Tuple) = _indices(x1, [x2...])
_indices(x1::Tuple, x2::Vector) = _indices([x1...], x2)

indices(x::Vector{Index{S}}) where {S} = x
indices(x::Vector{Index}) = narrow_eltype(x; default_empty_eltype=Index)
indices(x::Tuple) = reduce(_indices, x; init=())
indices(x::Vector) = reduce(_indices, x; init=Index[])
indices(x...) = indices(x)

# To help with backwards compatibility
IndexSet(inds::IndexSet) = inds
IndexSet(inds::Indices) = collect(inds)
IndexSet(inds::Index...) = collect(inds)
IndexSet(f::Function, N::Int) = map(f, 1:N)
IndexSet(f::Function, ::Order{N}) where {N} = IndexSet(f, N)

Tuple(is::IndexSet) = _Tuple(is)
NTuple{N}(is::IndexSet) where {N} = _NTuple(Val(N), is)

"""
    not(inds::Union{IndexSet, Tuple{Vararg{Index}}})
    not(inds::Index...)
    !(inds::Union{IndexSet, Tuple{Vararg{Index}}})
    !(inds::Index...)

Represents the set of indices not in the specified
IndexSet, for use in pattern matching (i.e. when
searching for an index or priming/tagging specified
indices).
"""
not(is::Indices) = Not(is)
not(inds::Index...) = not(inds)
!(is::Indices) = not(is)
!(inds::Index...) = not(inds...)

# Convert to an Index if there is only one
# TODO: also define the function `only`
Index(is::Indices) = is[]

NDTensors.dims(is::IndexSet) = dim.(is)

# Helps with generic code in `NDTensors`,
# for example with `NDTensors.similar`.
# Converts a set of Indices to a shape
# for allocating data.
Base.to_shape(inds::Tuple{Vararg{Index}}) = dims(inds)

"""
    dim(is::Indices)

Get the product of the dimensions of the indices
of the Indices (the total dimension of the space).
"""
NDTensors.dim(is::IndexSet) = Compat.mapreduce(dim, *, is; init=1)

"""
    dim(is::IndexSet, n::Int)

Get the dimension of the Index n of the Indices.
"""
NDTensors.dim(is::IndexSet, pos::Int) = dim(is[pos])

"""
    dag(is::Indices)

Return a new Indices with the indices daggered (flip
all of the arrow directions).
"""
function dag(is::Indices)
  return isempty(is) ? is : map(i -> dag(i), is)
end

# TODO: move to NDTensors
NDTensors.dim(is::Tuple, pos::Integer) = dim(is[pos])

# TODO: this is a weird definition, fix it
function NDTensors.similartype(
  ::Type{<:Tuple{Vararg{IndexT}}}, ::Type{Val{N}}
) where {IndexT,N}
  return NTuple{N,IndexT}
end

## # This is to help with some generic programming in the Tensor
## # code (it helps to construct an IndexSet(::NTuple{N,Index}) where the 
## # only known thing for dispatch is a concrete type such
## # as IndexSet{4})
## 
## #NDTensors.similartype(::Type{<:IndexSet},
## #                      ::Val{N}) where {N} = IndexSet
## 
## #NDTensors.similartype(::Type{<:IndexSet},
## #                      ::Type{Val{N}}) where {N} = IndexSet

"""
    sim(is::Indices)

Make a new Indices with similar indices.

You can also use the broadcast version `sim.(is)`.
"""
sim(is::Indices) = map(i -> sim(i), is)

function trivial_index(is::Indices)
  if isempty(is)
    return Index(1)
  end
  return trivial_index(first(is))
end

"""
    mindim(is::Indices)

Get the minimum dimension of the indices in the index set.

Returns 1 if the Indices is empty.
"""
function mindim(is::Indices)
  length(is) == 0 && (return 1)
  md = dim(is[1])
  for n in 2:length(is)
    md = min(md, dim(is[n]))
  end
  return md
end

"""
    maxdim(is::Indices)

Get the maximum dimension of the indices in the index set.

Returns 1 if the Indices is empty.
"""
function maxdim(is::Indices)
  length(is) == 0 && (return 1)
  md = dim(is[1])
  for n in 2:length(is)
    md = max(md, dim(is[n]))
  end
  return md
end

"""
    commontags(::Indices)

Return a TagSet of the tags that are common to all of the indices.
"""
commontags(is::Indices) = commontags(is...)

# 
# Set operations
#

"""
    ==(is1::Indices, is2::Indices)

Indices equality (order dependent). For order
independent equality use `issetequal` or
`hassameinds`.
"""
function ==(A::Indices, B::Indices)
  length(A) ≠ length(B) && return false
  for (a, b) in zip(A, B)
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
fmatch(is::Indices) = in(is)
fmatch(is::Index...) = fmatch(is)
fmatch(i::Index) = fmatch((i,))

fmatch(pl::Int) = hasplev(pl)

fmatch(tags::TagSet) = hastags(tags)
fmatch(tags::AbstractString) = fmatch(TagSet(tags))

fmatch(id::IDType) = hasid(id)

fmatch(n::Not) = !fmatch(parent(n))

# Function that always returns true
ftrue(::Any) = true

fmatch(::Nothing) = ftrue

"""
    ITensors.fmatch(; inds = nothing,
                      tags = nothing,
                      plev = nothing,
                      id = nothing) -> Function

An internal function that returns a function 
that accepts an Index that checks if the
Index matches the provided conditions.
"""
function fmatch(; inds=nothing, tags=nothing, plev=nothing, id=nothing)
  return i -> fmatch(inds)(i) && fmatch(plev)(i) && fmatch(id)(i) && fmatch(tags)(i)
end

"""
    indmatch

Checks if the Index matches the provided conditions.
"""
indmatch(i::Index; kwargs...) = fmatch(; kwargs...)(i)

"""
    getfirst(is::Indices)

Return the first Index in the Indices. If the Indices
is empty, return `nothing`.
"""
function getfirst(is::Indices)
  length(is) == 0 && return nothing
  return first(is)
end

"""
    getfirst(f::Function, is::Indices)

Get the first Index matching the pattern function,
return `nothing` if not found.
"""
function getfirst(f::Function, is::Indices)
  for i in is
    f(i) && return i
  end
  return nothing
end

getfirst(is::Indices, args...; kwargs...) = getfirst(fmatch(args...; kwargs...), is)

Base.findall(is::Indices, args...; kwargs...) = findall(fmatch(args...; kwargs...), is)

# In general this isn't defined for Tuple but is 
# defined for Vector
"""
    indexin(ais::Indices, bis::Indices)

For collections of Indices, returns the first location in
`bis` for each value in `ais`.
"""
function Base.indexin(ais::Indices, bis::Indices)
  return [findfirst(bis, ais[i]) for i in 1:length(ais)]
end

#function Base.indexin(a::Index, bis::Indices)
#  return [findfirst(bis, a)]
#end

findfirst(is::Indices, args...; kwargs...) = findfirst(fmatch(args...; kwargs...), is)

#
# Tagging functions
#

function prime(f::Function, is::Indices, args...)
  return map(i -> f(i) ? prime(i, args...) : i, is)
end

"""
    prime(A::Indices, plinc, ...)

Increase the prime level of the indices by the specified amount.
Filter which indices are primed using keyword arguments
tags, plev and id.
"""
function prime(is::Indices, plinc::Integer, args...; kwargs...)
  return prime(fmatch(args...; kwargs...), is, plinc)
end

prime(f::Function, is::Indices) = prime(f, is, 1)

prime(is::Indices, args...; kwargs...) = prime(is, 1, args...; kwargs...)

"""
    adjoint(is::Indices)

For is' notation.
"""
adjoint(is::Indices) = prime(is)

function setprime(f::Function, is::Indices, args...)
  return map(i -> f(i) ? setprime(i, args...) : i, is)
end

function setprime(is::Indices, plev::Integer, args...; kwargs...)
  return setprime(fmatch(args...; kwargs...), is, plev)
end

noprime(f::Function, is::Indices, args...) = setprime(is, 0, args...; kwargs...)

noprime(is::Indices, args...; kwargs...) = setprime(is, 0, args...; kwargs...)

function _swapprime(f::Function, i::Index, pl1pl2::Pair{Int,Int})
  pl1, pl2 = pl1pl2
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

function swapprime(f::Function, is::Indices, pl1pl2::Pair{Int,Int})
  return map(i -> _swapprime(f, i, pl1pl2), is)
end

function swapprime(f::Function, is::Indices, pl1::Int, pl2::Int)
  return swapprime(f, is::Indices, pl1 => pl2)
end

function swapprime(is::Indices, pl1pl2::Pair{Int,Int}, args...; kwargs...)
  return swapprime(fmatch(args...; kwargs...), is, pl1pl2)
end

function swapprime(is::Indices, pl1::Int, pl2::Int, args...; kwargs...)
  return swapprime(fmatch(args...; kwargs...), is, pl1 => pl2)
end

replaceprime(f::Function, is::Indices, pl1::Int, pl2::Int) = replaceprime(f, is, pl1 => pl2)

function replaceprime(is::Indices, pl1::Int, pl2::Int, args...; kwargs...)
  return replaceprime(fmatch(args...; kwargs...), is, pl1 => pl2)
end

const mapprime = replaceprime

function _replaceprime(i::Index, rep_pls::Pair{Int,Int}...)
  for (pl1, pl2) in rep_pls
    hasplev(i, pl1) && return setprime(i, pl2)
  end
  return i
end

function replaceprime(f::Function, is::Indices, rep_pls::Pair{Int,Int}...)
  return map(i -> f(i) ? _replaceprime(i, rep_pls...) : i, is)
end

function replaceprime(is::Indices, rep_pls::Pair{Int,Int}...; kwargs...)
  return replaceprime(fmatch(; kwargs...), is, rep_pls...)
end

addtags(f::Function, is::Indices, args...) = map(i -> f(i) ? addtags(i, args...) : i, is)

function addtags(is::Indices, tags, args...; kwargs...)
  return addtags(fmatch(args...; kwargs...), is, tags)
end

settags(f::Function, is::Indices, args...) = map(i -> f(i) ? settags(i, args...) : i, is)

function settags(is::Indices, tags, args...; kwargs...)
  return settags(fmatch(args...; kwargs...), is, tags)
end

"""
    CartesianIndices(is::Indices)

Create a CartesianIndices iterator for an Indices.
"""
CartesianIndices(is::Indices) = CartesianIndices(_Tuple(dims(is)))

"""
    eachval(is::Index...)
    eachval(is::Tuple{Vararg{Index}})

Create an iterator whose values correspond to a
Cartesian indexing over the dimensions
of the provided `Index` objects.
"""
eachval(is::Index...) = eachval(is)
eachval(is::Tuple{Vararg{Index}}) = CartesianIndices(dims(is))

"""
    eachindval(is::Index...)
    eachindval(is::Tuple{Vararg{Index}})

Create an iterator whose values are Index=>value pairs
corresponding to a Cartesian indexing over the dimensions
of the provided `Index` objects.

# Example

```julia
i = Index(3; tags="i")
j = Index(2; tags="j")
T = randomITensor(j, i)
for iv in eachindval(i, j)
  @show T[iv...]
end
```
"""
eachindval(is::Index...) = eachindval(is)
eachindval(is::Tuple{Vararg{Index}}) = (is .=> Tuple(ns) for ns in eachval(is))

function removetags(f::Function, is::Indices, args...)
  return map(i -> f(i) ? removetags(i, args...) : i, is)
end

function removetags(is::Indices, tags, args...; kwargs...)
  return removetags(fmatch(args...; kwargs...), is, tags)
end

function _replacetags(i::Index, rep_ts::Pair...)
  for (tags1, tags2) in rep_ts
    hastags(i, tags1) && return replacetags(i, tags1, tags2)
  end
  return i
end

# XXX new syntax
# hastags(any, is, ts)
"""
    anyhastags(is::Indices, ts::Union{String, TagSet})
    hastags(is::Indices, ts::Union{String, TagSet})

Check if any of the indices in the Indices have the specified tags.
"""
anyhastags(is::Indices, ts) = any(i -> hastags(i, ts), is)

hastags(is::Indices, ts) = anyhastags(is, ts)

# XXX new syntax
# hastags(all, is, ts)
"""
    allhastags(is::Indices, ts::Union{String, TagSet})

Check if all of the indices in the Indices have the specified tags.
"""
allhastags(is::Indices, ts::String) = all(i -> hastags(i, ts), is)

# Version taking a list of Pairs
function replacetags(f::Function, is::Indices, rep_ts::Pair...)
  return map(i -> f(i) ? _replacetags(i, rep_ts...) : i, is)
end

function replacetags(is::Indices, rep_ts::Pair...; kwargs...)
  return replacetags(fmatch(; kwargs...), is, rep_ts...)
end

# Version taking two input TagSets/Strings
replacetags(f::Function, is::Indices, tags1, tags2) = replacetags(f, is, tags1 => tags2)

function replacetags(is::Indices, tags1, tags2, args...; kwargs...)
  return replacetags(fmatch(args...; kwargs...), is, tags1 => tags2)
end

function _swaptags(f::Function, i::Index, tags1, tags2)
  if f(i)
    if hastags(i, tags1)
      return replacetags(i, tags1, tags2)
    elseif hastags(i, tags2)
      return replacetags(i, tags2, tags1)
    end
    return i
  end
  return i
end

function swaptags(f::Function, is::Indices, tags1, tags2)
  return map(i -> _swaptags(f, i, tags1, tags2), is)
end

function swaptags(is::Indices, tags1, tags2, args...; kwargs...)
  return swaptags(fmatch(args...; kwargs...), is, tags1, tags2)
end

function swaptags(is::Indices, tags12::Pair, args...; kwargs...)
  return swaptags(is, first(tags12), last(tags12), args...; kwargs...)
end

function replaceinds(is::Indices, rep_inds::Pair{<:Index,<:Index}...)
  return replaceinds(is, zip(rep_inds...)...)
end

# Handle case of empty indices being replaced
replaceinds(is::Indices) = is
replaceinds(is::Indices, rep_inds::Tuple{}) = is

function replaceinds(is::Indices, rep_inds::Vector{<:Pair{<:Index,<:Index}})
  return replaceinds(is, rep_inds...)
end

function replaceinds(is::Indices, rep_inds::Tuple{Vararg{Pair{<:Index,<:Index}}})
  return replaceinds(is, rep_inds...)
end

function replaceinds(is::Indices, rep_inds::Pair)
  return replaceinds(is, Tuple(first(rep_inds)) .=> Tuple(last(rep_inds)))
end

# Check that the QNs are all the same
hassameflux(i1::Index, i2::Index) = (dim(i1) == dim(i2))

function replaceinds_space_error(is, inds1, inds2, i1, i2)
  return error("""
               Attempting to replace the Indices

               $(inds1)

               with

               $(inds2)

               in the Index collection

               $(is).

               However, the Index

               $(i1)

               has a different space from the Index

               $(i2).

               They must have the same spaces to be replaced.
               """)
end

function replaceinds(is::Indices, inds1, inds2)
  is1 = inds1
  poss = indexin(is1, is)
  is_tuple = Tuple(is)
  for (j, pos) in enumerate(poss)
    isnothing(pos) && continue
    i1 = is_tuple[pos]
    i2 = inds2[j]
    i2 = setdir(i2, dir(i1))
    if space(i1) ≠ space(i2)
      replaceinds_space_error(is, inds1, inds2, i1, i2)
    end
    is_tuple = setindex(is_tuple, i2, pos)
  end
  return (is_tuple)
end

replaceind(is::Indices, i1::Index, i2::Index) = replaceinds(is, (i1,), (i2,))

function replaceind(is::Indices, i1::Index, i2::Indices)
  length(i2) != 1 &&
    throw(ArgumentError("cannot use replaceind with an Indices of length $(length(i2))"))
  return replaceinds(is, (i1,), i2)
end

replaceind(is::Indices, rep_i::Pair{<:Index,<:Index}) = replaceinds(is, rep_i)

function swapinds(is::Indices, inds1, inds2)
  return replaceinds(is, (inds1..., inds2...), (inds2..., inds1...))
end

function swapinds(is::Indices, inds1::Index, inds2::Index)
  return swapinds(is, (inds1,), (inds2,))
end

function swapinds(is::Indices, inds12::Pair)
  return swapinds(is, first(inds12), last(inds12))
end

swapind(is::Indices, i1::Index, i2::Index) = swapinds(is, (i1,), (i2,))

removeqns(is::Indices) = map(removeqns, is)
function removeqn(is::Indices, qn_name::String; mergeblocks=true)
  return map(i -> removeqn(i, qn_name; mergeblocks), is)
end
mergeblocks(is::Indices) = map(mergeblocks, is)

# Permute is1 to be in the order of is2
# This is helpful when is1 and is2 have different directions, and
# you want is1 to have the same directions as is2
# TODO: replace this functionality with
#
# setdirs(is1::Indices, is2::Indices)
#
function permute(is1::Indices, is2::Indices)
  length(is1) != length(is2) && throw(
    ArgumentError(
      "length of first index set, $(length(is1)) does not match length of second index set, $(length(is2))",
    ),
  )
  perm = getperm(is1, is2)
  return is1[invperm(perm)]
end

#
# Helper functions for contracting ITensors
#

function compute_contraction_labels(Ais::Tuple, Bis::Tuple)
  have_qns = hasqns(Ais) && hasqns(Bis)
  NA = length(Ais)
  NB = length(Bis)
  Alabels = MVector{NA,Int}(ntuple(_ -> 0, Val(NA)))
  Blabels = MVector{NB,Int}(ntuple(_ -> 0, Val(NB)))

  ncont = 0
  for i in 1:NA, j in 1:NB
    Ais_i = @inbounds Ais[i]
    Bis_j = @inbounds Bis[j]
    if Ais_i == Bis_j
      if have_qns && (dir(Ais_i) ≠ -dir(Bis_j))
        error(
          "Attempting to contract IndexSet:\n\n$(Ais)\n\nwith IndexSet:\n\n$(Bis)\n\nQN indices must have opposite direction to contract, but indices:\n\n$(Ais_i)\n\nand:\n\n$(Bis_j)\n\ndo not have opposite directions.",
        )
      end
      Alabels[i] = Blabels[j] = -(1 + ncont)
      ncont += 1
    end
  end

  u = ncont
  for i in 1:NA
    if (Alabels[i] == 0)
      Alabels[i] = (u += 1)
    end
  end
  for j in 1:NB
    if (Blabels[j] == 0)
      Blabels[j] = (u += 1)
    end
  end

  return (Tuple(Alabels), Tuple(Blabels))
end

function compute_contraction_labels(Cis::Tuple, Ais::Tuple, Bis::Tuple)
  NA = length(Ais)
  NB = length(Bis)
  NC = length(Cis)
  Alabels, Blabels = compute_contraction_labels(Ais, Bis)
  Clabels = MVector{NC,Int}(ntuple(_ -> 0, Val(NC)))
  for i in 1:NC
    locA = findfirst(==(Cis[i]), Ais)
    if !isnothing(locA)
      if Alabels[locA] < 0
        error(
          "The noncommon indices of $Ais and $Bis must be the same as the indices $Cis."
        )
      end
      Clabels[i] = Alabels[locA]
    else
      locB = findfirst(==(Cis[i]), Bis)
      if isnothing(locB) || Blabels[locB] < 0
        error(
          "The noncommon indices of $Ais and $Bis must be the same as the indices $Cis."
        )
      end
      Clabels[i] = Blabels[locB]
    end
  end
  return (Tuple(Clabels), Alabels, Blabels)
end

#
# TupleTools
#

"""
    pop(is::Indices)

Return a new Indices with the last Index removed.
"""
pop(is::Indices) = (NDTensors.pop(Tuple(is)))

# Overload the unexported NDTensors version
NDTensors.pop(is::Indices) = pop(is)

# TODO: don't convert to Tuple
"""
    popfirst(is::Indices)

Return a new Indices with the first Index removed.
"""
popfirst(is::IndexSet) = (NDTensors.popfirst(Tuple(is)))

# Overload the unexported NDTensors version
NDTensors.popfirst(is::IndexSet) = popfirst(is)

"""
    push(is::Indices, i::Index)

Make a new Indices with the Index i inserted
at the end.
"""
push(is::IndexSet, i::Index) = NDTensors.push(is, i)

# Overload the unexported NDTensors version
NDTensors.push(is::IndexSet, i::Index) = push(is, i)

# TODO: deprecate in favor of `filterinds` (abuse of Base notation)
filter(is::Indices, args...; kwargs...) = filter(fmatch(args...; kwargs...), is)

# For ambiguity with Base.filter
filter(is::Indices, args::String; kwargs...) = filter(fmatch(args; kwargs...), is)

#
# QN functions
#

"""
    setdirs(is::Indices, dirs::Arrow...)

Return a new Indices with indices `setdir(is[i], dirs[i])`.
"""
function setdirs(is::Indices, dirs)
  return map(i -> setdir(is[i], dirs[i]), 1:length(is))
end

"""
    dir(is::Indices, i::Index)

Return the direction of the Index `i` in the Indices `is`.
"""
function dir(is1::Indices, i::Index)
  return dir(getfirst(is1, i))
end

"""
    dirs(is::Indices, inds)

Return a tuple of the directions of the indices `inds` in
the Indices `is`, in the order they are found in `inds`.
"""
function dirs(is1::Indices, inds)
  return map(i -> dir(is1, inds[i]), 1:length(inds))
end

"""
    dirs(is::Indices)

Return a tuple of the directions of the indices `is`.
"""
dirs(is::Indices) = dir.(is)

hasqns(is::Indices) = any(hasqns, is)

"""
    getperm(col1, col2)

Get the permutation that takes collection 2 to collection 1,
such that `col2[p] .== col1`.
"""
function getperm(s1, s2)
  N = length(s1)
  r = Vector{Int}(undef, N)
  return map!(i -> findfirst(==(s1[i]), s2), r, 1:length(s1))
end

# TODO: define directly for Vector
"""
    nblocks(::Indices, i::Int)

The number of blocks in the specified dimension.
"""
function NDTensors.nblocks(inds::IndexSet, i::Int)
  return nblocks(Tuple(inds), i)
end

# TODO: don't convert to Tuple
function NDTensors.nblocks(inds::IndexSet, is)
  return nblocks(Tuple(inds), is)
end

"""
    nblocks(::Indices)

A tuple of the number of blocks in each
dimension.
"""
NDTensors.nblocks(inds::Indices) = nblocks.(inds)

# TODO: is this needed?
function NDTensors.nblocks(inds::NTuple{N,<:Index}) where {N}
  return ntuple(i -> nblocks(inds, i), Val(N))
end

ndiagblocks(inds) = minimum(nblocks(inds))

"""
    flux(inds::Indices, block::Tuple{Vararg{Int}})

Get the flux of the specified block, for example:

```
i = Index(QN(0)=>2, QN(1)=>2)
is = (i, dag(i'))
flux(is, Block(1, 1)) == QN(0)
flux(is, Block(2, 1)) == QN(1)
flux(is, Block(1, 2)) == QN(-1)
flux(is, Block(2, 2)) == QN(0)
```
"""
function flux(inds::Indices, block::Block)
  qntot = QN()
  for n in 1:length(inds)
    ind = inds[n]
    qntot += flux(ind, Block(block[n]))
  end
  return qntot
end

"""
    flux(inds::Indices, I::Integer...)

Get the flux of the block that the specified
index falls in.

```
i = Index(QN(0)=>2, QN(1)=>2)
is = (i, dag(i'))
flux(is, 3, 1) == QN(1)
flux(is, 1, 2) == QN(0)
```
"""
flux(inds::Indices, vals::Integer...) = flux(inds, block(inds, vals...))

"""
    ITensors.block(inds::Indices, I::Integer...)

Get the block that the specified index falls in.

This is mostly an internal function, and the interface
is subject to change.

# Examples

```julia
i = Index(QN(0)=>2, QN(1)=>2)
is = (i, dag(i'))
ITensors.block(is, 3, 1) == (2,1)
ITensors.block(is, 1, 2) == (1,1)
```
"""
block(inds::Indices, vals::Integer...) = blockindex(inds, vals...)[2]

#show(io::IO, is::IndexSet) = show(io, MIME"text/plain"(), is)

#
# Read and write
#

function readcpp(io::IO, ::Type{<:Indices}; kwargs...)
  format = get(kwargs, :format, "v3")
  is = IndexSet()
  if format == "v3"
    size = read(io, Int)
    function readind(io, n)
      i = readcpp(io, Index; kwargs...)
      stride = read(io, UInt64)
      return i
    end
    is = IndexSet(n -> readind(io, n), size)
  else
    throw(ArgumentError("read IndexSet: format=$format not supported"))
  end
  return is
end

function HDF5.write(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, is::Indices)
  g = create_group(parent, name)
  attributes(g)["type"] = "IndexSet"
  attributes(g)["version"] = 1
  N = length(is)
  write(g, "length", N)
  for n in 1:N
    write(g, "index_$n", is[n])
  end
end

function HDF5.read(
  parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, T::Type{<:Indices}
)
  g = open_group(parent, name)
  if read(attributes(g)["type"]) != "IndexSet"
    error("HDF5 group or file does not contain IndexSet data")
  end
  n = read(g, "length")
  return T(Index[read(g, "index_$j", Index) for j in 1:n])
end
