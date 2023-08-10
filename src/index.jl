
#const IDType = UInt128
const IDType = UInt64

# Custom RNG for Index id
# Vector of RNGs, one for each thread
const INDEX_ID_RNGs = MersenneTwister[]
@inline index_id_rng() = index_id_rng(Threads.threadid())
@noinline function index_id_rng(tid::Int)
  0 < tid <= length(INDEX_ID_RNGs) || _index_id_rng_length_assert()
  if @inbounds isassigned(INDEX_ID_RNGs, tid)
    @inbounds MT = INDEX_ID_RNGs[tid]
  else
    MT = MersenneTwister()
    @inbounds INDEX_ID_RNGs[tid] = MT
  end
  return MT
end
@noinline _index_id_rng_length_assert() = @assert false "0 < tid <= length(INDEX_ID_RNGs)"

"""
An `Index` represents a single tensor index with fixed dimension `dim`. Copies of an Index compare equal unless their
`tags` are different.

An Index carries a `TagSet`, a set of tags which are small strings that specify properties of the `Index` to help
distinguish it from other Indices. There is a special tag which is referred to as the integer tag or prime
level which can be incremented or decremented with special priming functions.

Internally, an `Index` has a fixed `id` number, which is how the ITensor library knows two indices are copies of a
single original `Index`. `Index` objects must have the same `id`, as well as the `tags` to compare equal.
"""
struct Index{T}
  id::IDType
  space::T
  dir::Arrow
  tags::TagSet
  plev::Int
  function Index{T}(id, space::T, dir, tags, plev) where {T}
    return new{T}(id, space, dir, tags, plev)
  end
end

#######################
# Index Constructors
#

# Used in NDTensors for generic code,
# mostly for internal usage
Index{T}(dim::T) where {T} = Index(dim)

function Index(id, space::T, dir, tags, plev) where {T}
  return Index{T}(id, space, dir, tags, plev)
end

"""
    Index(dim::Int; tags::Union{AbstractString, TagSet} = "",
                    plev::Int = 0)

Create an `Index` with a unique `id`, a TagSet given by `tags`,
and a prime level `plev`.

# Examples

```jldoctest; filter=r"id=[0-9]{1,3}"
julia> i = Index(2; tags="l", plev=1)
(dim=2|id=818|"l")'

julia> dim(i)
2

julia> plev(i)
1

julia> tags(i)
"l"
```
"""
function Index(dim::Number; tags="", plev=0, dir=Neither)
  return Index(rand(index_id_rng(), IDType), dim, dir, tags, plev)
end

"""
    Index(dim::Integer, tags::Union{AbstractString, TagSet}; plev::Int = 0)

Create an `Index` with a unique `id` and a tagset given by `tags`.

# Examples

```jldoctest; filter=r"id=[0-9]{1,3}"
julia> i = Index(2, "l,tag")
(dim=2|id=58|"l,tag")

julia> dim(i)
2

julia> plev(i)
0

julia> tags(i)
"l,tag"
```
"""
function Index(dim::Number, tags::Union{AbstractString,TagSet}; plev::Int=0)
  return Index(dim; tags=tags, plev=plev)
end

# This is so that when IndexSets are converted
# to Julia Base Sets, the hashing is done correctly
#function Base.hash(i::Index, h::UInt)
#  return hash((id(i), tags(i), plev(i)), h)
#end

"""
    copy(i::Index)

Create a copy of index `i` with identical `id`, `space`, `dir` and `tags`.
"""
copy(i::Index) = Index(id(i), copy(space(i)), dir(i), tags(i), plev(i))

"""
    sim(i::Index; tags = tags(i), plev = plev(i), dir = dir(i))

Produces an `Index` with the same properties (dimension or QN structure)
but with a new `id`.
"""
function sim(i::Index; tags=copy(tags(i)), plev=plev(i), dir=dir(i))
  return Index(rand(index_id_rng(), IDType), copy(space(i)), dir, tags, plev)
end

trivial_space(i::Index) = 1
trivial_index(i::Index) = Index(trivial_space(i))

#######################
# End Index Constructors
#

#######################
# Index properties
#

# TODO: decide if these are good definitions, using
# them for generic code in ContractionSequenceOptimization
Base.Int(i::Index) = dim(i)
length(i::Index) = 1

# Collect into a tuple
Base.Tuple(i::Index) = (i,)

# Collect into a 0-dimensional Vector
Base.collect(i::Index) = fill(i, ())

"""
    id(i::Index)

Obtain the id of an Index, which is a unique 64 digit integer.
"""
id(i::Index) = i.id

"""
    dim(i::Index)

Obtain the dimension of an Index.

For a QN Index, this is the sum of the block dimensions.
"""
NDTensors.dim(i::Index) = i.space

space(i::Index) = i.space

"""
    dir(i::Index)

Return the direction of an `Index` (`ITensors.In`, `ITensors.Out`, or `ITensors.Neither`).
"""
dir(i::Index) = i.dir

# Used for generic code in NDTensors
NDTensors.dir(i::Index) = dir(i)

# Trait to determine if an Index, Index collection, Tensor, or ITensor
# has symmetries
abstract type SymmetryStyle end

struct NonQN <: SymmetryStyle end

symmetrystyle(i::Index) = NonQN()
# Fallback definition for scalar ITensors (without any indices)
symmetrystyle() = NonQN()

"""
    tags(i::Index)

Obtain the TagSet of an Index.
"""
tags(i::Index) = i.tags

commontags(is::Index...) = commontags(tags.(is)...)
commontags(is::Index) = tags(is)
commontags() = ts""

"""
    plev(i::Index)

Obtain the prime level of an Index.
"""
plev(i::Index) = i.plev

"""
    hastags(i::Index, ts::Union{AbstractString,TagSet})

Check if an `Index` `i` has the provided tags,
which can be a string of comma-separated tags or
a TagSet object.

# Examples

```jldoctest; filter=r"id=[0-9]{1,3}"
julia> i = Index(2, "SpinHalf,Site,n=3")
(dim=2|id=861|"Site,SpinHalf,n=3")

julia> hastags(i, "SpinHalf,Site")
true

julia> hastags(i, "Link")
false
```
"""
hastags(i::Index, ts::Union{AbstractString,TagSet}) = hastags(tags(i), ts)

hastags(ts::Union{AbstractString,TagSet}) = x -> hastags(x, ts)

"""
    hasplev(i::Index, plev::Int)

Check if an `Index` `i` has the provided prime level.

# Examples

```jldoctest; filter=r"id=[0-9]{1,3}"
julia> i = Index(2; plev=2)
(dim=2|id=543)''

julia> hasplev(i, 2)
true

julia> hasplev(i, 1)
false
```
"""
hasplev(i::Index, pl::Int) = plev(i) == pl

"""
    hasplev(pl::Int)

Returns an anonymous function `x -> hasplev(x, pl)`.

Useful for passing to functions like `map`.
"""
hasplev(pl::Int) = x -> hasplev(x, pl)

"""
    hasind(i::Index)

Returns an anonymous function `x -> hasind(x, i)`.

Useful for passing to functions like `map`.
"""
hasind(s::Index) = x -> hasind(x, s)

"""
    hasid(i::Index, id::ITensors.IDType)

Check if an `Index` `i` has the provided id.

# Examples

```jldoctest; filter=r"id=[0-9]{1,3}"
julia> i = Index(2)
(dim=2|id=321)

julia> hasid(i, id(i))
true

julia> j = Index(2)
(dim=2|id=17)

julia> hasid(i, id(j))
false
```
"""
hasid(ind::Index, i::IDType) = id(ind) == i

hasid(i::IDType) = x -> hasid(x, i)

#
# QN related functions
#

hasqns(::Integer) = false

"""
    hasqns(::Index)

Checks of the Index has QNs or not.
"""
hasqns(i::Index) = hasqns(space(i))

#######################
# End Index properties
#

#######################
# Index operations
#

"""
    setdir(i::Index, dir::Arrow)

Create a copy of Index i with the specified direction.
"""
function setdir(i::Index, dir::Arrow)
  return Index(id(i), copy(space(i)), dir, copy(tags(i)), plev(i))
end

"""
    not(n::Int)

Return Not{Int}(n).
"""
not(pl::Int) = Not(pl)

"""
    not(::IDType)

Return Not{IDType}(n).
"""
not(id::IDType) = Not(id)

# Information essential to the
# identity of an Index.
# Currently only used for hashing an Index.
struct IndexID
  id::IDType
  tags::TagSet
  plev::Int
end
IndexID(i::Index) = IndexID(id(i), tags(i), plev(i))
hash(i::Index, h::UInt) = hash(IndexID(i), h)

"""
    ==(i1::Index, i1::Index)

Compare indices for equality. First the id's are compared,
then the prime levels are compared, and finally the
tags are compared.
"""
(i1::Index == i2::Index) =
  (id(i1) == id(i2)) && (plev(i1) == plev(i2)) && (tags(i1) == tags(i2))

"""
    dag(i::Index)

Copy an index `i` and reverse its direction.
"""
dag(i::Index) = Index(id(i), copy(space(i)), -dir(i), tags(i), plev(i))

# For internal use in NDTensors
NDTensors.dag(i::Index) = dag(i)

"""
    settags(i::Index, ts)

Return a copy of Index `i` with
tags replaced by the ones given
The `ts` argument can be a comma-separated
string of tags or a TagSet.

# Examples

```jldoctest; filter=r"id=[0-9]{1,3}"
julia> i = Index(2, "SpinHalf,Site,n=3")
(dim=2|id=543|"Site,SpinHalf,n=3")

julia> hastags(i, "Link")
false

julia> j = settags(i, "Link,n=4")
(dim=2|id=543|"Link,n=4")

julia> hastags(j, "Link")
true

julia> hastags(j, "n=4,Link")
true
```
"""
settags(i::Index, ts) = Index(id(i), copy(space(i)), dir(i), ts, plev(i))

setspace(i::Index, s) = Index(id(i), s, dir(i), tags(i), plev(i))

"""
    addtags(i::Index,ts)

Return a copy of Index `i` with the
specified tags added to the existing ones.
The `ts` argument can be a comma-separated
string of tags or a TagSet.
"""
addtags(i::Index, ts) = settags(i, addtags(tags(i), ts))

"""
    removetags(i::Index, ts)

Return a copy of Index `i` with the
specified tags removed. The `ts` argument
can be a comma-separated string of tags or a TagSet.
"""
removetags(i::Index, ts) = settags(i, removetags(tags(i), ts))

"""
    replacetags(i::Index, tsold, tsnew)

    replacetags(i::Index, tsold => tsnew)

If the tag set of `i` contains the tags specified by `tsold`,
replaces these with the tags specified by `tsnew`, preserving
any other tags. The arguments `tsold` and `tsnew` can be
comma-separated strings of tags, or TagSet objects.

# Examples

```jldoctest; filter=r"id=[0-9]{1,3}"
julia> i = Index(2; tags="l,x", plev=1)
(dim=2|id=83|"l,x")'

julia> replacetags(i, "l", "m")
(dim=2|id=83|"m,x")'

julia> replacetags(i, "l" => "m")
(dim=2|id=83|"m,x")'
```
"""
replacetags(i::Index, tsold, tsnew) = settags(i, replacetags(tags(i), tsold, tsnew))

replacetags(i::Index, rep_ts::Pair) = replacetags(i, rep_ts...)

"""
    prime(i::Index, plinc::Int = 1)

Return a copy of Index `i` with its
prime level incremented by the amount `plinc`
"""
prime(i::Index, plinc::Int=1) = setprime(i, plev(i) + plinc)

"""
    setprime(i::Index, plev::Int)

Return a copy of Index `i` with its
prime level set to `plev`
"""
setprime(i::Index, plev::Int) = Index(id(i), copy(space(i)), dir(i), tags(i), plev)

"""
    noprime(i::Index)

Return a copy of Index `i` with its
prime level set to zero.
"""
noprime(i::Index) = setprime(i, 0)

"""
    adjoint(i::Index)

Prime an Index using the notation `i'`.
"""
Base.adjoint(i::Index) = prime(i)

"""
    ^(i::Index, pl::Int)

Prime an Index using the notation `i^3`.
"""
Base.:^(i::Index, pl::Int) = prime(i, pl)

"""
Iterating over Index `I` gives the IndexVals `I(1)` through `I(dim(I))`.
"""
function Base.iterate(i::Index, state::Int=1)
  Base.depwarn(
    "iteration of `Index` is deprecated, use `eachindval` or `eachval` instead.", :iterate
  )
  (state > dim(i)) && return nothing
  return (i => state, state + 1)
end

# Treat Index as a scalar for the sake of broadcast.
# This allows:
#
# i = Index(2)
# ps = (n - 1 for n in 1:4)
# is = prime.(i, ps)
#
# or
#
# ts = ("i$n" for n in 1:4)
# is = settags.(i, ts)
#
Base.broadcastable(i::Index) = Ref(i)

"""
    eachval(i::Index)

Create an iterator whose values range
over the dimension of the provided `Index`.
"""
eachval(i::Index) = 1:dim(i)

"""
    eachindval(i::Index)

Create an iterator whose values are Pairs of
the form `i=>n` with `n` from `1:dim(i)`.
This iterator is useful for accessing elements of
an ITensor in a loop without needing to know
the ordering of the indices. See also
[`eachindval(is::Index...)`](@ref).
"""
eachindval(i::Index) = (i => n for n in eachval(i))

# This is a trivial definition for use in NDTensors
# XXX: rename tensorproduct with ⊗ alias
function NDTensors.outer(i::Index; dir=dir(i), tags="", plev::Int=0)
  return sim(i; tags=tags, plev=plev, dir=dir)
end

# This is for use in NDTensors
# XXX: rename tensorproduct with ⊗ alias
function NDTensors.outer(i1::Index, i2::Index; tags="")
  return Index(dim(i1) * dim(i2), tags)
end

# Non-qn Index
# TODO: add ⊕ alias
directsum(i::Index, j::Index; tags="sum") = Index(dim(i) + dim(j); tags=tags)
function directsum(i::Index, j::Index, k::Index, inds::Index...; tags="sum")
  return directsum(directsum(i, j; tags), k, inds...; tags)
end

#
# QN related functions
#

"""
    removeqns(::Index)

Removes the QNs from the Index, if it has any.
"""
removeqns(i::Index) = i

"""
    removeqn(::Index, qn_name::String)

Remove the specified QN from the Index, if it has any.
"""
removeqn(i::Index, qn_name::String) = i

"""
    mergeblocks(::Index)

Merge the contiguous QN blocks if they have the same
quantum numbers.
"""
mergeblocks(i::Index) = i

#######################
# End Index operations
#

#######################
# IndexVal functions
#

# Keep partial backwards compatibility by defining IndexVal as follows:
const IndexVal{IndexT} = Pair{IndexT,Int}

IndexVal(i::Index, n::Int) = (i => n)

function (i::Index)(n::Integer)
  Base.depwarn("Index(::Int) is deprecated, for an Index i use i=>n instead.", :Index)
  return i => n
end

NDTensors.ind(iv::Pair{<:Index}) = first(iv)

val(iv::Pair{<:Index}) = val(iv.first, iv.second)

val(i::Index, l::LastVal) = l.f(dim(i))

"""
    isindequal(i::Index, iv::IndexVal)

    isindequal(i::IndexVal, iv::Index)

    isindequal(i::IndexVal, iv::IndexVal)

Check if the Index and IndexVal have the same indices.
"""
isindequal(i::Index, iv::Pair{<:Index}) = (i == ind(iv))

isindequal(iv::Pair{<:Index}, i::Index) = isindequal(i, iv)

isindequal(iv1::Pair{<:Index}, iv2::Pair{<:Index}) = (ind(iv1) == ind(iv2))

plev(iv::Pair{<:Index}) = plev(ind(iv))

# TODO:
# Implement a macro with a general definition:
# f(iv::Pair{<:Index}, args...) = (f(ind(iv), args...) => val(iv))
prime(iv::Pair{<:Index}, inc::Integer=1) = (prime(ind(iv), inc) => val(iv))
sim(iv::Pair{<:Index}, args...) = (sim(ind(iv), args...) => val(iv))

dag(iv::Pair{<:Index}) = (dag(ind(iv)) => val(iv))

Base.adjoint(iv::Pair{<:Index}) = (prime(ind(iv)) => val(iv))

dir(iv::Pair{<:Index}) = dir(ind(iv))

#######################
# End IndexVal functions
#

#######################
# Index IO
#

function primestring(plev)
  if plev < 0
    return " (warning: prime level $plev is less than 0)"
  end
  if plev == 0
    return ""
  elseif plev > 3
    return "'$plev"
  else
    return "'"^plev
  end
end

function Base.show(io::IO, i::Index)
  idstr = "$(id(i) % 1000)"
  if length(tags(i)) > 0
    print(
      io, "(dim=$(space(i))|id=$(idstr)|\"$(tagstring(tags(i)))\")$(primestring(plev(i)))"
    )
  else
    print(io, "(dim=$(space(i))|id=$(idstr))$(primestring(plev(i)))")
  end
end

function readcpp(io::IO, ::Type{Index}; kwargs...)
  format = get(kwargs, :format, "v3")
  if format != "v3"
    throw(ArgumentError("read Index: format=$format not supported"))
  end
  tags = readcpp(io, TagSet; kwargs...)
  id = read(io, IDType)
  dim = convert(Int64, read(io, Int32))
  dir_int = read(io, Int32)
  dir = dir_int < 0 ? In : Out
  read(io, 8) # Read default IQIndexDat size, 8 bytes
  return Index(id, dim, dir, tags)
end

function HDF5.write(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, I::Index)
  g = create_group(parent, name)
  attributes(g)["type"] = "Index"
  attributes(g)["version"] = 1
  write(g, "id", id(I))
  write(g, "dim", dim(I))
  write(g, "dir", Int(dir(I)))
  write(g, "tags", tags(I))
  write(g, "plev", plev(I))
  if typeof(space(I)) == Int
    attributes(g)["space_type"] = "Int"
  elseif typeof(space(I)) == QNBlocks
    attributes(g)["space_type"] = "QNBlocks"
    write(g, "space", space(I))
  else
    error("Index space type not recognized")
  end
end

function HDF5.read(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{Index})
  g = open_group(parent, name)
  if read(attributes(g)["type"]) != "Index"
    error("HDF5 group or file does not contain Index data")
  end
  id = read(g, "id")
  dim = read(g, "dim")
  dir = Arrow(read(g, "dir"))
  tags = read(g, "tags", TagSet)
  plev = read(g, "plev")
  space_type = "Int"
  if haskey(attributes(g), "space_type")
    space_type = read(attributes(g)["space_type"])
  end
  if space_type == "Int"
    space = dim
  elseif space_type == "QNBlocks"
    space = read(g, "space", QNBlocks)
  end
  return Index(id, space, dir, tags, plev)
end
