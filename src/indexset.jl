
@eval struct Order{N}
  (OrderT::Type{ <: Order})() = $(Expr(:new, :OrderT))
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

# Definition to help with generic code
const Indices{IndexT <: Index} = Union{Vector{IndexT}, Tuple{Vararg{IndexT}}}

# This is only for backwards compatibility, in general just use Vector
const IndexSet{IndexT <: Index} = Vector{IndexT}

# TODO: also define IndexTuple?

# To help with backwards compatibility
IndexSet(inds::IndexSet) = inds
IndexSet(inds::Indices) = collect(inds)
IndexSet(inds::Index...) = collect(inds)
IndexSet(f::Function, N::Int) = map(f, 1:N)
IndexSet(f::Function, ::Order{N}) where {N} = IndexSet(f, N)

# TODO: extend type restriction to `IndexT <: Union{<: Index, <: IndexVal}`
## struct IndexSet{IndexT <: Index} <: AbstractVector{IndexT}
##   data::Vector{IndexT}
## 
##   function IndexSet{IndexT}(data::AbstractVector{IndexT}) where {IndexT}
##     @debug_check begin
##       if !allunique(data)
##         error("Trying to create IndexSet with collection of indices $data. Indices must be unique.")
##       end
##     end
##     return new{IndexT}(data)
##   end
## 
##   """
##       IndexSet()
## 
##   Create a special "empty" `IndexSet` with data `[]` and any number of indices.
## 
##   This is used as the `IndexSet` of an `emptyITensor()`, an ITensor with `NDTensors.Empty` storage and any number of indices.
##   """
##   IndexSet()   = new{Union{}}(Any[])
## end

## function IndexSet{Union{}}(data::Vector{<:Any})
##   return IndexSet()
## end
## IndexSet{Union{}}(())                 = IndexSet{Union{}}(Any[])
## IndexSet{IndexT}(data) where {IndexT} = IndexSet{IndexT}(collect(data))

## """
##     IndexSet(::Function, ::Order{N})
## Construct an IndexSet of length N from a function that accepts
## an integer between 1:N and returns an Index.
## # Examples
## ```julia
## IndexSet(n -> Index(1, "i\$n"), Order(4))
## ```
## """
## IndexSet(f::Function, N::Int) = IndexSet(map(f, 1:N))
## 
## IndexSet(f::Function, ::Order{N}) where {N} = IndexSet(f, N)

## """
##     IndexSet{IndexT}(inds)
##     IndexSet{IndexT}(inds::Index...)
## 
## Construct an `IndexSet` of element type `IndexT`
## from a collection of indices (any collection that is convertable to a `Vector`).
## """
## function IndexSet{IndexT}(inds::Index...) where {IndexT}
##   return IndexSet{IndexT}(inds)
## end

## """
##     IndexSet(inds)
##     IndexSet(inds::Index...)
## 
## Construct an IndexSet from a collection of indices
## (any collection that is convertable to a `Vector`).
## """
## IndexSet(inds) = IndexSet{eltype(inds)}(inds)
## 
## IndexSet(inds::NTuple{N, IndexT}) where {N, IndexT} =
##     IndexSet{IndexT}(inds)
## 
## IndexSet(inds::Index...) = IndexSet(inds)
## 
## IndexSet(is::IndexSet) = is
## 
## IndexSet(::Tuple{}) = IndexSet()#IndexSet{Union{}}(Any[])
## 
## """
##     convert(::Type{IndexSet}, t)
## 
## Convert the collection to an `IndexSet`,
## as long as it can be converted to a `Tuple`.
## """
## Base.convert(::Type{IndexSet}, t) = IndexSet(t)
## 
## Base.convert(::Type{IndexSet}, is::IndexSet) = is
## 
## Base.convert(::Type{IndexSet{IndexT}}, t) where {IndexT} = IndexSet{IndexT}(t)
## 
## Base.convert(::Type{IndexSet{IndexT}}, is::IndexSet{IndexT}) where {IndexT} = is

# This is not defined on purpose because in general it won't make
# unique indices, use a Vector{<: Index} instead
#Base.similar(is::IndexSet) = IndexSet(similar(data(is)))

# TODO: define for Vector{<: Index}

# This is a cache of [Val(1), Val(2), ...]
# Hard-coded for now to only handle tensors up to order 100
const ValCache = Val[Val(n) for n in 0:100]
# Faster conversions of collection to tuple than `Tuple(::AbstractVector)`
_NTuple(::Val{N}, v::Vector{T}) where {N, T} = ntuple(n -> v[n], Val(N))
_Tuple(v::Vector{T}) where {T} = _NTuple(ValCache[length(v) + 1], v)
_Tuple(t::Tuple) = t

# TODO: define in terms of IndexSet
Base.Tuple(is::AbstractVector{<: Index}) = _Tuple(is)
Base.NTuple{N}(is::AbstractVector{<: Index}) where {N} = _NTuple(Val(N), is)

## """
##     IndexSet(inds::Vector{<:Index})
## 
## Convert a `Vector` of indices to an `IndexSet`.
## """
## IndexSet(inds::Vector{IndexT}) where {IndexT} = IndexSet{IndexT}(inds)


"""
    not(inds::Union{IndexSet, Tuple{Vararg{<:Index}}})
    not(inds::Index...)
    !(inds::Union{IndexSet, Tuple{Vararg{<:Index}}})
    !(inds::Index...)

Represents the set of indices not in the specified
IndexSet, for use in pattern matching (i.e. when
searching for an index or priming/tagging specified
indices).
"""
not(is::Indices) = Not(is)
not(inds::Index...) = not(inds)
Base.:!(is::Indices) = not(is)
Base.:!(inds::Index...) = not(inds...)

"""
    NDTensors.data(is::IndexSet)

Return the raw storage data for the indices.
Currently the storage is a `Tuple`.

This is mostly for internal usage, please
contact us if there is functionality you want
availabe for `IndexSet`.
"""
data(is::IndexSet) = is.data
data(is) = is

# This is used in type promotion in the Tensor contraction code
Base.promote_rule(::Type{<:IndexSet},
                  ::Type{Val{N}}) where {N} = IndexSet

# TODO: delete
#NDTensors.ValLength(::Type{<:IndexSet}) = Val(length(is))

# TODO: delete
#
# This is only needed for
#
# block(inds::IndexSet, vals::Int...) = blockindex(inds, vals...)[2]
#
# which should be written in a different way to avoid this.
#
NDTensors.ValLength(s::Indices) = Val(length(s))

# TODO: delete
#function NDTensors._permute(s::T, perm) where {T<:IndexSet}
#  return ntuple(i->s[perm[i]], length(s))
#end

# Convert to an Index if there is only one
# TODO: also define the function `only`
Index(is::Indices) = is[]

## """
##     getindex(is::IndexSet, n::Integer)
## 
## Get the Index of the IndexSet in the dimension n.
## """
## getindex(is::IndexSet, n::Union{Integer, CartesianIndex{1}}) = getindex(data(is), n)
## getindex(is::IndexSet) = getindex(data(is))
## 
## """
##     getindex(is::IndexSet, v)
## 
## Get the indices of the IndexSet in the dimensions
## specified in the collection v, returning an IndexSet.
## """
## getindex(is::IndexSet, v) =
##   IndexSet(getindex(data(is), v))

"""
    setindex(is::IndexSet, i::Index, n::Int)

Set the Index of the IndexSet in the dimension n.

This function is mostly for internal use, if you want to
replace the indices of an IndexSet, use the `replaceind`,
`replaceind!`, `replaceinds`, and `replaceinds!` functions,
which check for compatibility of the indices and ensure
the proper Arrow directions.
"""
function Base.setindex(is::IndexSet, i::Index, n::Int)
  return IndexSet(setindex!(copy(data(is)), i, n))
end

## """
##     length(is::IndexSet)
## 
## The number of indices in the IndexSet.
## """
## Base.length(is::IndexSet) = length(is.data)
## 
## """
##     size(is::IndexSet)
## 
## The size of the IndexSet, the same as `(length(is),)`.
## 
## Mostly for internal use for compatability with Base methods,
## like for broadcasting.
## """
## Base.size(is::IndexSet) = size(data(is))
## 
## """
##     axes(is::IndexSet)
## 
## The axes of the IndexSet, the same as `(Base.OneTo(length(is)),)`.
## 
## Mostly for internal use for compatability with Base methods,
## like for broadcasting.
## """
## Base.axes(is::IndexSet) = axes(data(is))

NDTensors.dims(is::Indices) = dim.(is)

# TODO: is this needed in NDTensors?
# Is the above generic definition just as fast?
NDTensors.dims(is::NTuple{N,<:Index}) where {N} = ntuple(i->dim(is[i]),Val(N))

"""
    dim(is::Indices)

Get the product of the dimensions of the indices
of the Indices (the total dimension of the space).
"""
NDTensors.dim(is::Indices) = prod(dim, is; init = 1)

# TODO: is this needed in NDTensors?
NDTensors.dim(is::Tuple{Vararg{<:Index}}) = prod(dims(is))

"""
    dim(is::Indices, n::Int)

Get the dimension of the Index n of the Indices.
"""
NDTensors.dim(is::Indices, pos::Int) = dim(is[pos])

# TODO: is this needed in NDTensors?
"""
    dim(is::NTuple{N, <:Index}, n::Int)

Get the dimension of the Index n of the `NTuple`.
"""
NDTensors.dim(is::NTuple{N, <:Index}, pos::Int) where {N} = dim(is[pos])


"""
    dag(is::Indices)

Return a new Indices with the indices daggered (flip
all of the arrow directions).
"""
dag(is::Indices) = map(i -> dag(i), is)

## """
##     iterate(is::Indices[, state])
## 
## Iterate over the indices of an Indices.
## 
## # Example
## ```jldoctest
## julia> using ITensors;
## 
## julia> i = Index(2);
## 
## julia> is = (i,i');
## 
## julia> for j in is
##          println(plev(j))
##        end
## 0
## 1
## ```
## """
## Base.iterate(is::IndexSet, state) = iterate(data(is), state)
## 
## Base.iterate(is::IndexSet) = iterate(data(is))
## 
## # To allow for the syntax is[end]
## Base.firstindex(::IndexSet) = 1
## 
## # To allow for the syntax is[begin]
## Base.lastindex(is::IndexSet) = length(is)
## 
## """
##     eltype(::Indices)
## 
## Get the element type of the Indices.
## """
## Base.eltype(is::IndexSet{IndexT}) where {IndexT} = IndexT
## 
## Base.eltype(::Type{<: IndexSet{IndexT}}) where {IndexT} = IndexT
## 
## # Needed for findfirst (I think)
## Base.keys(is::IndexSet) = 1:length(is)

# This is to help with some generic programming in the Tensor
# code (it helps to construct an IndexSet(::NTuple{N,Index}) where the 
# only known thing for dispatch is a concrete type such
# as IndexSet{4})

NDTensors.similar(T::NDTensors.DenseTensor,
                  inds::NTuple) = NDTensors._similar(T, inds)
NDTensors.similar_type(::Type{<:IndexSet},
                       ::Val{N}) where {N} = IndexSet

NDTensors.similar_type(::Type{<:Tuple{Vararg{<:Index}}},
                       ::Type{Val{N}}) where {N} = NTuple{N, Index}

NDTensors.similar_type(::Type{<:IndexSet},
                       ::Type{Val{N}}) where {N} = IndexSet

"""
    sim(is::Indices)

Make a new Indices with similar indices.

You can also use the broadcast version `sim.(is)`.
"""
sim(is::Indices) = map(i -> sim(i), is)

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
  for n ∈ 2:length(is)
    md = max(md,dim(is[n]))
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
function Base.:(==)(A::Indices, B::Indices)
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
fmatch(is::Indices) = in(is)
fmatch(is::Tuple{Vararg{<:Index}}) = fmatch(IndexSet(is))
fmatch(is::Index...) = fmatch(IndexSet(is...))

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
function fmatch(; inds = nothing,
                  tags = nothing,
                  plev = nothing,
                  id = nothing)
  return i -> fmatch(inds)(i) &&
              fmatch(plev)(i) &&
              fmatch(id)(i) &&
              fmatch(tags)(i)
end

"""
    indmatch

Checks if the Index matches the provided conditions.
"""
indmatch(i::Index; kwargs...) = fmatch(; kwargs...)(i)

# Set functions
#
# setdiff
# intersect
# symdiff
# union
# filter
#

function Base.setdiff!(f::Function,
                       r,
                       A::Indices,
                       Bs::Indices...)
  
  N = length(r)
  j = 1
  for a in A
    if f(a) && all(B -> a ∉ B, Bs)
      j > N && error("Too many intersects found")
      r[j] = a
      j += 1
    end
  end
  j ≤ N && error("Too few intersects found")
  return r
end

function Base.setdiff(f::Function,
                      A::Indices,
                      Bs::Indices...)
  R = eltype(A)[]
  for a ∈ A
    f(a) && all(B -> a ∉ B, Bs) && push!(R, a)
  end
  return R
end

"""
    setdiff(A::Indices, Bs::Indices...)

Output the Vector of Indices with Indices in `A` but not in
the Indicess `Bs`.
"""
Base.setdiff(A::Indices, Bs::Indices...; kwargs...) =
  setdiff(fmatch(; kwargs...), A, Bs...)

function firstsetdiff(f::Function,
                      A::Indices,
                      Bs::Indices...)
  for a in A
    f(a) && all(B -> a ∉ B, Bs) && return a
  end
  return nothing
end

# XXX: use the interface setdiff(first, A, Bs...)
"""
    firstsetdiff(A::Indices, Bs::Indices...)

Output the first Index in `A` that is not in the Indicess `Bs`.
Otherwise, return a default constructed Index.
"""
firstsetdiff(A::Indices,
             Bs::Indices...;
             kwargs...) = firstsetdiff(fmatch(; kwargs...), A, Bs...)

function Base.intersect(f::Function, A::Indices, B::Indices)
  R = eltype(A)[]
  for a in A
    f(a) && a ∈ B && push!(R,a)
  end
  return R
end

function Base.intersect!(f::Function,
                         R::AbstractVector,
                         A::Indices,
                         B::Indices)
  N = length(R)
  j = 1
  for a in A
    if f(a) && a ∈ B
      j > N && error("Too many intersects found")
      R[j] = a
      j += 1
    end
  end
  j ≤ N && error("Too few intersects found")
  return R
end

"""
    intersect(A::Indices, B::Indices; kwargs...)

    intersect(f::Function, A::Indices, B::Indices)

Output the Vector of Indices in the intersection of `A` and `B`,
optionally filtering with keyword arguments `tags`, `plev`, etc. 
or by a function `f(::Index) -> Bool`.
"""
Base.intersect(A::Indices, B::Indices; kwargs...) =
  intersect(fmatch(; kwargs...), A, B)

function firstintersect(f::Function, A::Indices, B::Indices)
  for a in A
    f(a) && a ∈ B && return a
  end
  return nothing
end

# XXX: use interface intersect(first, A, B)
"""
    firstintersect(A::Indices, B::Indices; kwargs...)

    firstintersect(f::Function, A::Indices, B::Indices)

Output the first Index common to `A` and `B`, optionally
filtering by tags, prime level, etc. or by a function
`f`.

If no common Index is found, return `nothing`.
"""
firstintersect(A::Indices, B::Indices; kwargs...) =
  firstintersect(fmatch(; kwargs...), A, B)

"""
    filter(f::Function, inds::Indices)

Filter the Indices by the given function (output a new
Indices with indices `i` for which `f(i)` returns true).

Note that this function is not type stable, since the number
of output indices is not known at compile time.
"""
Base.filter(f::Function, is::Tuple{Vararg{Index}}) = filter(f, collect(is))

# TODO: is this definition needed?
Base.filter(is::Tuple{Vararg{Index}}) = is

Base.filter(is::Indices, args...; kwargs...) =
  filter(fmatch(args...; kwargs...), is)

# To fix ambiguity error with Base function
Base.filter(is::Indices, tags::String; kwargs...) =
  filter(fmatch(tags; kwargs...),is)

function Base.filter!(f::Function,
                      r,
                      is::Indices{IndexT}) where {IndexT}
  N = length(r)
  j = 1
  for i in is
    if f(i)
      j > N && error("Too many intersects found")
      r[j] = i
      j += 1
    end
  end
  j ≤ N && error("Too few intersects found")
  return r 
end

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

getfirst(is::Indices,
         args...; kwargs...) = getfirst(fmatch(args...;
                                               kwargs...),is)

Base.findall(is::Indices,
             args...; kwargs...) = findall(fmatch(args...;
                                                  kwargs...), is)

"""
    indexin(ais::Indices, bis::Indices)

For collections of Indices, returns the first location in 
`bis` for each value in `ais`.
"""
function Base.indexin(ais::Indices,
                      bis::Indices)
  return [findfirst(bis, ais[i]) for i in 1:length(ais)]
end

Base.findfirst(is::Indices, args...; kwargs...) =
  findfirst(fmatch(args...; kwargs...), is)

#
# Tagging functions
#

## function prime(f::Function, is::Indices, args...)
##   isᵣ = similar(data(is))
##   map!(i -> f(i) ? prime(i, args...) : i, isᵣ, is)
##   return (isᵣ)
## end

function prime(f::Function, is::Indices, args...)
  return map(i -> f(i) ? prime(i, args...) : i, is)
end

"""
    prime(A::Indices, plinc, ...)

Increase the prime level of the indices by the specified amount.
Filter which indices are primed using keyword arguments
tags, plev and id.
"""
prime(is::Indices, plinc::Integer, args...; kwargs...) =
  prime(fmatch(args...; kwargs...), is, plinc)

prime(f::Function, is::Indices) = prime(f, is, 1)

prime(is::Indices, args...; kwargs...) = prime(is, 1, args...; kwargs...)

"""
    adjoint(is::Indices)

For is' notation.
"""
Base.adjoint(is::Indices) = prime(is)

function setprime(f::Function,
                  is::Indices,
                  args...)
  return map(i -> f(i) ? setprime(i, args...) : i, is)
end

setprime(is::Indices,
         plev::Integer,
         args...; kwargs...) = setprime(fmatch(args...; kwargs...),
                                        is, plev)

noprime(f::Function,
        is::Indices,
        args...) = setprime(is, 0, args...; kwargs...)

noprime(is::Indices,
        args...;
        kwargs...) = setprime(is, 0, args...; kwargs...)

function _swapprime(f::Function, i::Index, pl1pl2::Pair{Int, Int})
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

swapprime(f::Function, is::Indices, pl1pl2::Pair{Int, Int}) =
  map(i -> _swapprime(f, i, pl1pl2), is)

swapprime(f::Function, is::Indices, pl1::Int, pl2::Int) =
  swapprime(f, is::Indices, pl1 => pl2)

swapprime(is::Indices, pl1pl2::Pair{Int, Int}, args...; kwargs...) =
  swapprime(fmatch(args...; kwargs...), is, pl1pl2)

swapprime(is::Indices, pl1::Int, pl2::Int, args...; kwargs...) =
  swapprime(fmatch(args...; kwargs...), is, pl1 => pl2)

replaceprime(f::Function, is::Indices, pl1::Int, pl2::Int) =
  replaceprime(f, is, pl1 => pl2)

replaceprime(is::Indices, pl1::Int, pl2::Int, args...; kwargs...) =
  replaceprime(fmatch(args...; kwargs...), is, pl1 => pl2)

const mapprime = replaceprime

function _replaceprime(i::Index, rep_pls::Pair{Int, Int}...)
  for (pl1, pl2) in rep_pls
    hasplev(i, pl1) && return setprime(i, pl2)
  end
  return i
end

function replaceprime(f::Function, is::Indices, rep_pls::Pair{Int, Int}...)
  return map(i -> f(i) ? _replaceprime(i, rep_pls...) : i, is)
end

replaceprime(is::Indices, rep_pls::Pair{Int, Int}...; kwargs...) =
  replaceprime(fmatch(; kwargs...), is, rep_pls...)

addtags(f::Function, is::Indices, args...) =
  map(i -> f(i) ? addtags(i, args...) : i, is)

addtags(is::Indices, tags, args...; kwargs...) =
  addtags(fmatch(args...; kwargs...), is, tags)

settags(f::Function, is::Indices, args...) =
  map(i -> f(i) ? settags(i, args...) : i, is)

settags(is::Indices, tags, args...; kwargs...) =
  settags(fmatch(args...; kwargs...), is, tags)

"""
    CartesianIndices(is::Indices)

Create a CartesianIndices iterator for an Indices.
"""
CartesianIndices(is::Indices) = CartesianIndices(_Tuple(dims(is)))

removetags(f::Function, is::Indices, args...) =
  map(i -> f(i) ? removetags(i, args...) : i, is)

removetags(is::Indices, tags, args...; kwargs...) =
  removetags(fmatch(args...; kwargs...), is, tags)

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
allhastags(is::Indices, ts::String) =
  all(i -> hastags(i, ts), is)

# Version taking a list of Pairs
replacetags(f::Function, is::Indices, rep_ts::Pair...) =
  map(i -> f(i) ? _replacetags(i, rep_ts...) : i, is)

replacetags(is::Indices, rep_ts::Pair...; kwargs...) =
  replacetags(fmatch(; kwargs...), is, rep_ts...)

# Version taking two input TagSets/Strings
replacetags(f::Function, is::Indices, tags1, tags2) =
  replacetags(f, is, tags1 => tags2)

replacetags(is::Indices, tags1, tags2, args...; kwargs...) =
  replacetags(fmatch(args...; kwargs...), is, tags1 => tags2)

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

swaptags(is::Indices, tags1, tags2, args...; kwargs...) =
  swaptags(fmatch(args...; kwargs...), is, tags1, tags2)

replaceinds(is::Indices, rep_inds::Pair{<: Index, <: Index}...) =
  replaceinds(is, zip(rep_inds...)...)

replaceinds(is::Indices, rep_inds::Vector{<: Pair{<: Index, <: Index}}) =
  replaceinds(is, rep_inds...)

replaceinds(is::Indices, rep_inds::Tuple{Vararg{Pair{<: Index, <: Index}}}) =
  replaceinds(is, rep_inds...)

replaceinds(is::Indices, rep_inds::Pair) =
  replaceinds(is, Tuple(first(rep_inds)) .=> Tuple(last(rep_inds)))

# Check that the QNs are all the same
hassameflux(i1::Index, i2::Index) = (dim(i1) == dim(i2))

function replaceinds(is::Indices, inds1, inds2)
  is1 = (inds1)
  poss = indexin(is1, is)
  is_tuple = Tuple(is)
  for (j, pos) in enumerate(poss)
    isnothing(pos) && continue
    i1 = is_tuple[pos]
    i2 = inds2[j]
    i2 = setdir(i2, dir(i1))
    space(i1) ≠ space(i2) && error("Indices must have the same spaces to be replaced")
    is_tuple = setindex(is_tuple, i2, pos)
  end
  return (is_tuple)
end

replaceind(is::Indices, i1::Index, i2::Index) =
  replaceinds(is, (i1,), (i2,))

function replaceind(is::Indices, i1::Index, i2::Indices)
    length(i2) != 1 && throw(ArgumentError("cannot use replaceind with an Indices of length $(length(i2))"))
    replaceinds(is, (i1,), i2)
end

replaceind(is::Indices, rep_i::Pair{ <: Index, <: Index}) =
  replaceinds(is, rep_i)

swapinds(is::Indices, inds1, inds2) =
  replaceinds(is, (inds1..., inds2...), (inds2..., inds1...))

swapind(is::Indices, i1::Index, i2::Index) = swapinds(is, (i1,), (i2,))

removeqns(is::Indices) = is

# Permute is1 to be in the order of is2
# This is helpful when is1 and is2 have different directions, and
# you want is1 to have the same directions as is2
# TODO: replace this functionality with
#
# setdirs(is1::Indices, is2::Indices)
#
function permute(is1::Indices, is2::Indices)
  length(is1) != length(is2) && throw(ArgumentError("length of first index set, $(length(is1)) does not match length of second index set, $(length(is2))"))
  perm = getperm(is1, is2)
  return is1[perm]
end

#
# Helper functions for contracting ITensors
#

function compute_contraction_labels(Ais::Indices, Bis::Indices)
  have_qns = hasqns(Ais) && hasqns(Bis)
  NA = length(Ais)
  NB = length(Bis)
  Alabels = fill(0, NA)
  Blabels = fill(0, NB)

  ncont = 0
  for i = 1:NA, j = 1:NB
    Ais_i = @inbounds Ais[i]
    Bis_j = @inbounds Bis[j]
    if Ais_i == Bis_j
      if have_qns && (dir(Ais_i) ≠ -dir(Bis_j))
        error("Attempting to contract Indices:\n$(Ais)with Indices:\n$(Bis)QN indices must have opposite direction to contract.")
      end
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

  return Alabels, Blabels
end

function compute_contraction_labels(Cis::Indices,
                                    Ais::Indices,
                                    Bis::Indices)
  NA = length(Ais)
  NB = length(Bis)
  NC = length(Cis)
  Alabels, Blabels = compute_contraction_labels(Ais, Bis)
  Clabels = fill(0, NC)
  for i = 1:NC
    locA = findfirst(==(Cis[i]), Ais)
    if !isnothing(locA)
      if Alabels[locA] < 0
        error("The noncommon indices of $Ais and $Bis must be the same as the indices $Cis.")
      end
      Clabels[i] = Alabels[locA]
    else
      locB = findfirst(==(Cis[i]), Bis)
      if isnothing(locB) || Blabels[locB] < 0
        error("The noncommon indices of $Ais and $Bis must be the same as the indices $Cis.")
      end
      Clabels[i] = Blabels[locB]
    end
  end
  return Clabels, Alabels, Blabels
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

# Overload the unexported NDTensors version
#NDTensors.push(is::Indices{0},
#             i::Index) = push(is, i)

# TODO: define directly for Vector
"""
    pushfirst(is::Indices, i::Index)

Make a new Indices with the Index i inserted
at the beginning.
"""
pushfirst(is::IndexSet, i::Index) = NDTensors.pushfirst(Tuple(is), i)

# Overload the unexported NDTensors version
NDTensors.pushfirst(is::IndexSet, i::Index) = pushfirst(is, i)

# TODO: don't convert to Tuple
"""
    instertat(is1::Indices, is2, pos::Int)

Remove the index at pos and insert the indices
is2 starting at that position.
"""
function insertat(is1::IndexSet, is2, pos::Int)
  return NDTensors.insertat(Tuple(is1), Tuple((is2)), pos)
end

# Overload the unexported NDTensors version
NDTensors.insertat(is1::IndexSet, is2, pos::Int) = insertat(is1, is2, pos)

# TODO: don't convert to Tuple
"""
    instertafter(is1::Indices, is2, pos)

Insert the indices is2 after position pos.
"""
insertafter(is::IndexSet, I...) =
  (NDTensors.insertafter(Tuple(is), I...))

# Overload the unexported NDTensors version
NDTensors.insertafter(is::IndexSet, I...) = insertafter(is, I...)

# TODO: don't convert to Tuple here
deleteat(is::IndexSet, I...) =
  (NDTensors.deleteat(Tuple(is),I...))

# Overload the unexported NDTensors version
NDTensors.deleteat(is::IndexSet, I...) = deleteat(is, I...)

# TODO: don't convert to Tuple
getindices(is::IndexSet, I...) = NDTensors.getindices(Tuple(is), I...)

NDTensors.getindices(is::IndexSet, I...) = getindices(is, I...)

#
# QN functions
#

"""
    setdirs(is::Indices, dirs::Arrow...)

Return a new Indices with indices `setdir(is[i], dirs[i])`.
"""
function setdirs(is::Indices, dirs)
  return map(i->setdir(is[i], dirs[i]), 1:length(is))
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
  return map(i->dir(is1, inds[i]), 1:length(inds))
end

"""
    dirs(is::Indices)

Return a tuple of the directions of the indices `is`.
"""
dirs(is::Indices) = dir.(is)

hasqns(is::Indices) = any(hasqns,is)

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
  return nblocks(Tuple(inds),i)
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

# TODO: generic to Indices and BlockDims
function eachblock(inds::Indices)
  return CartesianIndices(_Tuple(nblocks(inds)))
end

# TODO: turn this into an iterator instead
# of returning a Vector
function eachdiagblock(inds::Indices)
  return [fill(i, length(inds)) for i in 1:ndiagblocks(inds)]
end

"""
    flux(inds::Indices, block::Tuple{Vararg{Int}})

Get the flux of the specified block, for example:
```
i = Index(QN(0)=>2, QN(1)=>2)
is = (i, dag(i'))
flux(is, (1,1)) == QN(0)
flux(is, (2,1)) == QN(1)
flux(is, (1,2)) == QN(-1)
flux(is, (2,2)) == QN(0)
```
"""
function flux(inds::Indices, block)
  qntot = QN()
  for n in 1:length(inds)
    ind = inds[n]
    qntot += flux(ind, Block(block[n]))
  end
  return qntot
end

"""
    flux(inds::Indices, I::Int...)

Get the flux of the block that the specified
index falls in.
```
i = Index(QN(0)=>2, QN(1)=>2)
is = (i, dag(i'))
flux(is, 3, 1) == QN(1)
flux(is, 1, 2) == QN(0)
```
"""
flux(inds, vals...) = flux(inds, block(inds, vals...))

"""
    ITensors.block(inds::Indices, I::Int...)

Get the block that the specified index falls in.

This is mostly an internal function, and the interface
is subject to change.
```
i = Index(QN(0)=>2, QN(1)=>2)
is = (i, dag(i'))
ITensors.block(is, 3, 1) == (2,1)
ITensors.block(is, 1, 2) == (1,1)
```
"""
block(inds::Indices, vals::Int...) = blockindex(inds, vals...)[2]

function Base.show(io::IO, is::IndexSet)
  print(io,"IndexSet{$(length(is))} ")
  for n in eachindex(is)
    i = is[n]
    print(io, i)
    if n < lastindex(is)
      print(io, " ")
    end
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
    is = IndexSet(n -> readind(io,n), size)
  else
    throw(ArgumentError("read IndexSet: format=$format not supported"))
  end
  return is
end

function HDF5.write(parent::Union{HDF5.File,HDF5.Group},
                    name::AbstractString,
                    is::IndexSet)
  g = create_group(parent,name)
  attributes(g)["type"] = "IndexSet"
  attributes(g)["version"] = 1
  N = length(is)
  write(g,"length",N)
  for n=1:N
    write(g,"index_$n",is[n])
  end
end

function HDF5.read(parent::Union{HDF5.File,HDF5.Group},
                   name::AbstractString,
                   ::Type{<:IndexSet})
  g = open_group(parent,name)
  if read(attributes(g)["type"]) != "IndexSet"
    error("HDF5 group or file does not contain IndexSet data")
  end
  N = read(g,"length")
  return IndexSet(n->read(g,"index_$n",Index),N)
end

