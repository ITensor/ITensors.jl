
"""
An ITensor is a tensor whose interface is 
independent of its memory layout. Therefore
it is not necessary to know the ordering
of an ITensor's indices, only which indices
an ITensor has. Operations like contraction
and addition of ITensors automatically
handle any memory permutations.
"""
mutable struct ITensor{N}
  store::TensorStorage
  inds::IndexSet{N}

  # TODO: check that the storage is consistent with the 
  # indices (possibly only in debug mode);
  """
      ITensor{N}(is::IndexSet{N}, st::TensorStorage)

  This is an internal constructor for an ITensor where the ITensor stores a view of the `NDTensors.TensorStorage`.
  """
  ITensor{N}(is,
             st::TensorStorage) where {N} = new{N}(st, is)

  ITensor{Any}(is,
               st::Empty) = new{Any}(st, is)
end

function ITensor{Any}(is,
                      st::TensorStorage)
  error("Can only make an ITensor with Any number of indices with NDTensors.Empty storage")
end

"""
    itensor(st::TensorStorage, is)

Constructor for an ITensor from a TensorStorage
and a set of indices.
The ITensor stores a view of the TensorStorage.
"""
itensor(st::TensorStorage, is) = ITensor{length(is)}(is, st)

"""
    ITensor(st::TensorStorage, is)

Constructor for an ITensor from a TensorStorage
and a set of indices.
The TensorStorage is copied (the ITensor
owns the storage data).
"""
ITensor(st::TensorStorage, is) = itensor(copy(st), is)

"""
    inds(T::ITensor)

Return the indices of the ITensor as an IndexSet.
"""
NDTensors.inds(T::ITensor) = T.inds

"""
    ind(T::ITensor, i::Int)

Get the Index of the ITensor along dimension i.
"""
NDTensors.ind(T::ITensor, i::Int) = inds(T)[i]

# Explicit import since there are some deprecations
# involving store for other types.
import .NDTensors: store

"""
    store(T::ITensor)

Return a view of the TensorStorage of the ITensor.
"""
store(T::ITensor) = T.store

"""
    data(T::ITensor)

Return a view of the raw data of the ITensor.

This is mostly an internal ITensor function, please
let the developers of ITensors.jl know if there is
functionality for ITensors that you would like
that is not currently available.
"""
data(T::ITensor) = NDTensors.data(store(T))

Base.similar(T::ITensor) = itensor(similar(tensor(T)))

Base.similar(T::ITensor,
             ::Type{ElT}) where {ElT<:Number} = itensor(similar(tensor(T),ElT))

setinds!(T::ITensor,is) = (T.inds = is; return T)

setstore!(T::ITensor,st::TensorStorage) = (T.store = st; return T)

setinds(T::ITensor, is) = itensor(store(T),is)

setstore(T::ITensor, st) = itensor(st,inds(T))

#
# Iteration over ITensors
#

"""
    CartesianIndices(A::ITensor)

Iterate over the CartesianIndices of an ITensor.
"""
Base.CartesianIndices(A::ITensor) = CartesianIndices(dims(A))

#
# ITensor constructors
#

"""
    ITensor(::Tensor)

Convert a Tensor to an ITensor with a copy of the storage and
indices.

To make an ITensor that shares the same storage as the Tensor,
is the function `itensor(::Tensor)`.
"""
ITensor(T::Tensor{<:Number,N}) where {N} = ITensor{N}(store(T),
                                                      inds(T))

ITensor{N}(T::Tensor{<:Number,N}) where {N} = ITensor{N}(store(T),
                                                         inds(T))

"""
    itensor(::Tensor)

Make an ITensor that shares the same storage as the Tensor and
has the same indices.
"""
itensor(T::Tensor) = itensor(store(T),
                             inds(T))

"""
    tensor(::ITensor)

Convert the ITensor to a Tensor that shares the same
storage and indices as the ITensor.
"""
NDTensors.tensor(A::ITensor) = tensor(store(A),inds(A))

"""
    ITensor([::Type{ElT} = Float64, ]inds)
    ITensor([::Type{ElT} = Float64, ]inds::Index...)

Construct an ITensor filled with zeros having indices `inds` and element type `ElT`. If the element type is not specified, it defaults to `Float64`.

The storage will have `NDTensors.Dense` type.
"""
function ITensor(::Type{ElT},
                 inds::Indices) where {ElT <: Number}
  return itensor(Dense(ElT, dim(inds)), inds)
end

ITensor(::Type{ElT},
        inds::Index...) where {ElT <: Number} = ITensor(ElT,
                                                        IndexSet(inds...))

# To fix ambiguity with QN Index version
ITensor(::Type{ElT}) where {ElT <: Number} = ITensor(ElT, IndexSet())

ITensor(is::Indices) = ITensor(Float64, is)

ITensor(inds::Index...) = ITensor(Float64, IndexSet(inds...))

# To fix ambiguity with QN Index version
ITensor() = ITensor(Float64, IndexSet())

"""
    ITensor([::Type{ElT} = Float64, ]::UndefInitializer, inds)
    ITensor([::Type{ElT} = Float64, ]::UndefInitializer, inds::Index...)

Construct an ITensor filled with undefined elements having indices `inds` and element type `ElT`. If the element type is not specified, it defaults to `Float64`.

The storage will have `NDTensors.Dense` type.
"""
function ITensor(::Type{ElT},
                 ::UndefInitializer,
                 inds::Indices) where {ElT <: Number}
  return itensor(Dense(ElT, undef, dim(inds)), inds)
end

ITensor(::Type{ElT},
        ::UndefInitializer,
        inds::Index...) where {ElT} = ITensor(ElT,
                                              undef,
                                              IndexSet(inds...))

ITensor(::UndefInitializer,
        inds::Indices) = ITensor(Float64, undef, inds)

ITensor(::UndefInitializer,
        inds::Index...) = ITensor(Float64, undef, IndexSet(inds...))

"""
    ITensor(x::Number, inds)
    ITensor(x::Number, inds::Index...)

Construct an ITensor with all elements set to `float(x)` and indices `inds`.

The storage will have `NDTensors.Dense` type.
"""
function ITensor(x::Number,
                 inds::Indices)
  return itensor(Dense(float(x),dim(inds)),inds)
end

ITensor(x::Number,
        inds::Index...) = ITensor(x, IndexSet(inds...))

#
# Empty ITensor constructors
#

"""
    emptyITensor([::Type{ElT} = Float64, ]inds)
    emptyITensor([::Type{ElT} = Float64, ]inds::Index...)

Construct an ITensor with storage type `NDTensors.Empty`, indices `inds`, and element type `ElT`. If the element type is not specified, it defaults to `Float64`.
"""
function emptyITensor(::Type{ElT},
                      inds::Indices) where {ElT <: Number}
  return itensor(EmptyTensor(ElT, inds))
end

function emptyITensor(::Type{ElT},
                     inds::Index...) where {ElT <: Number}
  return emptyITensor(ElT, IndexSet(inds...))
end

emptyITensor(is::Indices) = emptyITensor(Float64, is)

emptyITensor(inds::Index...) = emptyITensor(Float64,
                                            IndexSet(inds...))

function emptyITensor(::Type{ElT}) where {ElT <: Number}
  return itensor(EmptyTensor(ElT, IndexSet()))
end

emptyITensor() = emptyITensor(Float64)

"""
    emptyITensor([::Type{ElT} = Float64, ]::Type{Any})

Construct an ITensor with empty storage and `Any` number of indices.
"""
function emptyITensor(::Type{ElT}, ::Type{Any}) where {ElT <: Number}
  return itensor(EmptyTensor(ElT, IndexSet{Any}()))
end

emptyITensor(::Type{Any}) = emptyITensor(Float64, Any)

#
# Construct from Array
#

"""
    itensor(A::Array, inds)
    itensor(A::Array, inds::Index...)

Construct an ITensor from an Array `A` and indices `inds`.
The ITensor will be the closest floating point storage to the
Array (`float(A)`), and the storage will be a view of the Array
data if possible (if the Array already has floating point elements).
"""
function itensor(A::Array{<:Number},
                 inds::Indices)
  length(A) ≠ dim(inds) && throw(DimensionMismatch("In ITensor(::Array, ::IndexSet), length of Array ($(length(A))) must match total dimension of IndexSet ($(dim(inds)))"))
  return itensor(Dense(float(vec(A))), inds)
end

itensor(A::Array{<:Number},
        inds::Index...) = itensor(A, IndexSet(inds...))

"""
    ITensor(A::Array, inds)
    ITensor(A::Array, inds::Index...)

Construct an ITensor from an Array `A` and indices `inds`.
The ITensor will be the closest floating point storage to the
Array (`float(A)`), and the storage will store a copy of the Array
data.
"""
ITensor(A::Array{<:AbstractFloat},
        inds::Indices) = itensor(copy(A), inds)

ITensor(A::Array,
        inds::Indices) = itensor(A, inds)

ITensor(A::Array, inds::Index...) = ITensor(A, IndexSet(inds...))

#
# Diag ITensor constructors
#

"""
    diagITensor([::Type{ElT} = Float64, ]inds)
    diagITensor([::Type{ElT} = Float64, ]inds::Index...)

Make a sparse ITensor of element type `ElT` with only elements
along the diagonal stored. Defaults to having `zero(T)` along 
the diagonal.

The storage will have `NDTensors.Diag` type.
"""
function diagITensor(::Type{ElT},
                     is::Indices) where {ElT}
  return itensor(Diag(ElT, mindim(is)), is)
end

diagITensor(::Type{ElT},
            inds::Index...) where {ElT} = diagITensor(ElT,
                                                      IndexSet(inds...))

diagITensor(is::Indices) = diagITensor(Float64, is)

diagITensor(inds::Index...) = diagITensor(Float64, IndexSet(inds...))

"""
    diagITensor(v::Vector{T}, inds)
    diagITensor(v::Vector{T}, inds::Index...)

Make a sparse ITensor with non-zero elements only along the diagonal. 
The diagonal elements will be set to the values stored in `v` and 
the ITensor will have element type `float(T)`.
The storage will have type `NDTensors.Diag`.
"""
function diagITensor(v::Vector{<:Number},
                     is::Indices)
  length(v) ≠ mindim(is) && error("Length of vector for diagonal must equal minimum of the dimension of the input indices")
  return itensor(Diag(float(v)), is)
end

function diagITensor(v::Vector{<:Number},
                     is::Index...)
  return diagITensor(v, IndexSet(is...))
end

"""
    diagITensor(x::Number, inds)
    diagITensor(x::Number, inds::Index...)

Make a sparse ITensor with non-zero elements only along the diagonal. 
The diagonal elements will be set to the value `float(x)` and
the ITensor will have element type `float(eltype(x))`.
The storage will have `NDTensors.Diag` type.
"""
function diagITensor(x::Number,
                     is::Indices)
  return itensor(Diag(float(x), mindim(is)), is)
end

function diagITensor(x::Number,
                     is::Index...)
  return diagITensor(x, IndexSet(is...))
end

"""
    delta([::Type{ElT} = Float64, ]inds)
    delta([::Type{ElT} = Float64, ]inds::Index...)

Make a uniform diagonal ITensor with all diagonal elements
`one(ElT)`. Only a single diagonal element is stored.

This function has an alias `δ`.
"""
function delta(::Type{T},
               is::Indices) where {T<:Number}
  return itensor(Diag(one(T)), is)
end

function delta(::Type{T},
               is::Index...) where {T<:Number}
  return delta(T, IndexSet(is...))
end

delta(is::Indices) = delta(Float64, is)

delta(is::Index...) = delta(Float64, IndexSet(is...))

const δ = delta

"""
    setelt(iv)

Create an ITensor with all zeros except the specified value,
which is set to 1.
"""
function setelt(iv::IndexValOrPairIndexInt)
  A = emptyITensor(ind(iv))
  A[val(iv)] = 1.0
  return A
end

"""
    dense(T::ITensor)

Make a new ITensor where the storage is the closest Dense storage,
avoiding allocating new data if possible.
For example, an ITensor with Diag storage will become Dense storage,
filled with zeros except for the diagonal values.
"""
function NDTensors.dense(A::ITensor)
  T = dense(tensor(A))
  return itensor(store(T), removeqns(inds(A)))
end

"""
    complex(T::ITensor)

Convert to the complex version of the storage.
"""
Base.complex(T::ITensor) = itensor(complex(tensor(T)))

Base.eltype(T::ITensor) = eltype(tensor(T))

"""
    order(A::ITensor)
    ndims(A::ITensor)

The number of indices, `length(inds(A))`.
"""
order(T::ITensor) = ndims(T)

Base.ndims(::ITensor{N}) where {N} = N

"""
    dim(A::ITensor)

The total dimension of the space the tensor lives in, `prod(dims(A))`.
"""
NDTensors.dim(T::ITensor) = dim(inds(T))

"""
    dims(A::ITensor)
    size(A::ITensor)

Tuple containing `dim(inds(A)[d]) for d in 1:ndims(A)`.
"""
NDTensors.dims(T::ITensor) = dims(inds(T))

Base.size(T::ITensor) = dims(T)

Base.size(A::ITensor,
          d::Int) = dim(inds(A), d)

Base.copy(T::ITensor) = itensor(copy(tensor(T)))

"""
    Array{ElT}(T::ITensor, i:Index...)

Given an ITensor `T` with indices `i...`, returns
an Array with a copy of the ITensor's elements. The
order in which the indices are provided indicates
the order of the data in the resulting Array.
"""
function Base.Array{ElT, N}(T::ITensor{N},
                            is::Vararg{Index, N}) where {ElT, N}
  return Array{ElT, N}(tensor(permute(T, is...)))::Array{ElT, N}
end

function Base.Array{ElT}(T::ITensor{N},
                         is::Vararg{Index, N}) where {ElT, N}
  return Array{ElT, N}(T, is...)
end

function Base.Array(T::ITensor{N},
                    is::Vararg{Index, N}) where {N}
  return Array{eltype(T), N}(T, is...)::Array{<:Number, N}
end

"""
    Matrix(T::ITensor, row_i:Index, col_i::Index)

Given an ITensor `T` with two indices `row_i` and `col_i`, returns
a Matrix with a copy of the ITensor's elements. The
order in which the indices are provided indicates
which Index is to be treated as the row index of the 
Matrix versus the column index.

"""
function Base.Matrix(T::ITensor{2},
                     row_i::Index,
                     col_i::Index)
  return Array(T,row_i,col_i)
end

function Base.Vector(T::ITensor{1},
                     i::Index)
  return Array(T,i)
end

function Base.Vector{ElT}(T::ITensor{1}) where {ElT}
  return Vector{ElT}(T,inds(T)...)
end

function Base.Vector(T::ITensor{1})
  return Vector(T,inds(T)...)
end

"""
    scalar(T::ITensor)

Extract the element of an order zero ITensor.

Same as `T[]`.
"""
scalar(T::ITensor) = T[]::Number

"""
    getindex(T::ITensor, I::Int...)

Get the specified element of the ITensor, using internal
Index ordering of the ITensor.

# Example
```julia
i = Index(2; tags = "i")
A = ITensor(2.0, i, i')
A[1, 2] # 2.0, same as: A[i => 1, i' => 2]
```
"""
function Base.getindex(T::ITensor{N},
                       I::Vararg{Int,N}) where {N}
  return tensor(T)[I...]::Number
end

# Version accepting CartesianIndex, useful when iterating over
# CartesianIndices
Base.getindex(T::ITensor{N},
              I::CartesianIndex{N}) where {N} = tensor(T)[I]::Number

"""
    getindex(T::ITensor, ivs...)

Get the specified element of the ITensor, using a list
of `IndexVal`s or `Pair{<:Index, Int}`.

# Example
```julia
i = Index(2; tags = "i")
A = ITensor(2.0, i, i')
A[i => 1, i' => 2] # 2.0, same as: A[i' => 2, i => 1]
```
"""
function Base.getindex(T::ITensor, ivs...)
  p = NDTensors.getperm(inds(T), ind.(ivs))
  vals = NDTensors.permute(val.(ivs), p)
  return T[vals...]
end

Base.getindex(T::ITensor) = tensor(T)[]

"""
    setindex!(T::ITensor, x::Number, I::Int...)

Set the specified element of the ITensor, using internal
Index ordering of the ITensor.

# Example
```julia
i = Index(2; tags = "i")
A = ITensor(i, i')
A[1, 2] = 1.0 # same as: A[i => 1, i' => 2] = 1.0
```
"""
function Base.setindex!(T::ITensor, x::Number, I::Int...)
  fluxT = flux(T)
  if !isnothing(fluxT) && fluxT != flux(T, I...)
    error("In `setindex!`, the element you are trying to set is in a block that does not have the same flux as the other blocks of the ITensor. You may be trying to create an ITensor that does not have a well defined quantum number flux.")
  end
  TR = setindex!!(tensor(T), x, I...)
  setstore!(T, store(TR))
  return T
end

"""
    setindex!(T::ITensor, x::Number, ivs...)

Set the specified element of the ITensor using a list
of `IndexVal`s or `Pair{<:Index, Int}`.

# Example
```julia
i = Index(2; tags = "i")
A = ITensor(i, i')
A[i => 1, i' => 2] = 1.0 # same as: A[i' => 2, i => 1] = 1.0
```
"""
function Base.setindex!(T::ITensor, x::Number, ivs...)
  p = NDTensors.getperm(inds(T), ind.(ivs))
  vals = NDTensors.permute(val.(ivs), p)
  T[vals...] = x
  return T
end

function Base.setindex!(::ITensor{Any}, ::Number, ivs...)
  error("Cannot set the element of an emptyITensor(). Must define indices to set elements")
end

function Base.iterate(::ITensor, args...)
  error("""Iterating ITensors is currently not supported (it will be supported in the future).

        You may be attempting to use the deprecated notation `C,c = combiner(i,j)` to grab both the combiner ITensor and combined Index.
        Note that the `combiner` constructor currently only outputs the combiner ITensor, you can extract the combined Index with `C = combiner(i,j); c = combinedind(C)`.
        """)
end

"""
    fill!(T::ITensor, x::Number)

Fill all values of the ITensor with the specified value.
"""
function Base.fill!(T::ITensor,
                    x::Number)
  # TODO: automatically switch storage type if needed?
  # Use broadcasting `T .= x`?
  fill!(tensor(T), x)
  return T
end

itensor2inds(A::ITensor) = inds(A)
itensor2inds(A) = A


# in
hasind(A,i::Index) = i ∈ itensor2inds(A)

# issubset
hasinds(A, is) = is ⊆ itensor2inds(A)
hasinds(A, is::Index...) = hasinds(A, IndexSet(is...))

# issetequal
hassameinds(A,B) = issetequal(itensor2inds(A),
                              itensor2inds(B))

# intersect
commoninds(A...; kwargs...) = IndexSet(intersect(itensor2inds.(A)...;
                                                 kwargs...)...)

# firstintersect
commonind(A...; kwargs...) = firstintersect(itensor2inds.(A)...;
                                            kwargs...)

# symdiff
noncommoninds(A...; kwargs...) = IndexSet(symdiff(itensor2inds.(A)...;
                                               kwargs...)...)

# firstsymdiff
noncommonind(A...; kwargs...) = getfirst(symdiff(itensor2inds.(A)...;
                                                 kwargs...))

# setdiff
uniqueinds(A...; kwargs...) = IndexSet(setdiff(itensor2inds.(A)...;
                                               kwargs...)...)

# firstsetdiff
uniqueind(A...; kwargs...) = firstsetdiff(itensor2inds.(A)...;
                                          kwargs...)

# union
unioninds(A...; kwargs...) = IndexSet(union(itensor2inds.(A)...;
                                            kwargs...)...)

# firstsymdiff
unionind(A...; kwargs...) = getfirst(union(itensor2inds.(A)...;
                                           kwargs...))

firstind(A...; kwargs...) = getfirst(itensor2inds.(A)...;
                                     kwargs...)

NDTensors.inds(A...; kwargs...) = filter(itensor2inds.(A)...;
                                         kwargs...)

# in-place versions of priming and tagging
for fname in (:prime,
              :setprime,
              :noprime,
              :mapprime,
              :swapprime,
              :addtags,
              :removetags,
              :replacetags,
              :settags,
              :swaptags,
              :replaceind,
              :replaceinds,
              :swapind,
              :swapinds)
  @eval begin
    $fname(f::Function,
           A::ITensor,
           args...) = setinds(A,$fname(f,
                                       inds(A),
                                       args...))

    $(Symbol(fname,:!))(f::Function,
                        A::ITensor,
                        args...) = setinds!(A,$fname(f,
                                                     inds(A),
                                                     args...))

    $fname(A::ITensor,
           args...;
           kwargs...) = setinds(A,$fname(inds(A),
                                         args...;
                                         kwargs...))

    $(Symbol(fname,:!))(A::ITensor,
                        args...;
                        kwargs...) = setinds!(A,$fname(inds(A),
                                                       args...;
                                                       kwargs...))
  end
end

priming_tagging_doc = """
Optionally, only modify the indices with the specified keyword arguments.

# Arguments
- `tags = nothing`: if specified, only modify Index `i` if `hastags(i, tags) == true`. 
- `plev = nothing`: if specified, only modify Index `i` if `hasplev(i, plev) == true`.

In both versions above, the ITensor storage is not modified or copied (so the first version returns an ITensor with a view of the original storage).
"""

@doc """
    prime(A::ITensor, plinc::Int = 1; <keyword arguments>) -> ITensor

    prime!(A::ITensor, plinc::Int = 1; <keyword arguments>)

Increase the prime level of the indices of an ITensor.

$priming_tagging_doc
""" prime(::ITensor, ::Any...)

@doc """
    setprime(A::ITensor, plev::Int; <keyword arguments>) -> ITensor

    setprime!(A::ITensor, plev::Int; <keyword arguments>)

Set the prime level of the indices of an ITensor.

$priming_tagging_doc
""" setprime(::ITensor, ::Any...)

@doc """
    noprime(A::ITensor; <keyword arguments>) -> ITensor

    noprime!(A::ITensor; <keyword arguments>)

Set the prime level of the indices of an ITensor to zero.

$priming_tagging_doc
""" noprime(::ITensor, ::Any...)

@doc """
    mapprime(A::ITensor, plold::Int, plnew::Int; <keyword arguments>) -> ITensor

    mapprime!(A::ITensor, plold::Int, plnew::Int; <keyword arguments>)

Set the prime level of the indices of an ITensor with prime level `plold` to `plnew`.

$priming_tagging_doc
""" mapprime(::ITensor, ::Any...)

@doc """
    swapprime(A::ITensor, pl1::Int, pl2::Int; <keyword arguments>) -> ITensor

    swapprime!(A::ITensor, pl1::Int, pl2::Int; <keyword arguments>)

Set the prime level of the indices of an ITensor with prime level `pl1` to `pl2`, and those with prime level `pl2` to `pl1`.

$priming_tagging_doc
""" swapprime(::ITensor, ::Any...)

@doc """
    addtags(A::ITensor, ts::String; <keyword arguments>) -> ITensor

    addtags!(A::ITensor, ts::String; <keyword arguments>)

Add the tags `ts` to the indices of an ITensor.

$priming_tagging_doc
""" addtags(::ITensor, ::Any...)

@doc """
    removetags(A::ITensor, ts::String; <keyword arguments>) -> ITensor

    removetags!(A::ITensor, ts::String; <keyword arguments>)

Remove the tags `ts` from the indices of an ITensor.

$priming_tagging_doc
""" removetags(::ITensor, ::Any...)

@doc """
    settags(A::ITensor, ts::String; <keyword arguments>) -> ITensor

    settags!(A::ITensor, ts::String; <keyword arguments>)

Set the tags of the indices of an ITensor to `ts`.

$priming_tagging_doc
""" settags(::ITensor, ::Any...)

@doc """
    replacetags(A::ITensor, tsold::String, tsnew::String; <keyword arguments>) -> ITensor

    replacetags!(A::ITensor, tsold::String, tsnew::String; <keyword arguments>)

Replace the tags `tsold` with `tsnew` for the indices of an ITensor.

$priming_tagging_doc
""" replacetags(::ITensor, ::Any...)

@doc """
    swaptags(A::ITensor, ts1::String, ts2::String; <keyword arguments>) -> ITensor

    swaptags!(A::ITensor, ts1::String, ts2::String; <keyword arguments>)

Swap the tags `ts1` with `ts2` for the indices of an ITensor.

$priming_tagging_doc
""" swaptags(::ITensor, ::Any...)

@doc """
    replaceind(A::ITensor, i1::Index, i2::Index) -> ITensor

    replaceind!(A::ITensor, i1::Index, i2::Index)

Replace the Index `i1` with the Index `i2` in the ITensor.

The indices must have the same space (i.e. the same dimension and QNs, if applicable).
""" replaceind(::ITensor, ::Any...)

@doc """
    replaceinds(A::ITensor, inds1, inds2) -> ITensor

    replaceinds!(A::ITensor, inds1, inds2)

Replace the Index `inds1[n]` with the Index `inds2[n]` in the ITensor, where `n` runs from `1` to `length(inds1) == length(inds2)`.

The indices must have the same space (i.e. the same dimension and QNs, if applicable).

The storage of the ITensor is not modified or copied (the output ITensor is a view of the input ITensor).
""" replaceinds(::ITensor, ::Any...)

@doc """
    swapind(A::ITensor, i1::Index, i2::Index) -> ITensor

    swapind!(A::ITensor, i1::Index, i2::Index)

Swap the Index `i1` with the Index `i2` in the ITensor.

The indices must have the same space (i.e. the same dimension and QNs, if applicable).
""" swapind(::ITensor, ::Any...)

@doc """
    swapinds(A::ITensor, inds1, inds2) -> ITensor

    swapinds!(A::ITensor, inds1, inds2)

Swap the Index `inds1[n]` with the Index `inds2[n]` in the ITensor, where `n` runs from `1` to `length(inds1) == length(inds2)`.

The indices must have the same space (i.e. the same dimension and QNs, if applicable).

The storage of the ITensor is not modified or copied (the output ITensor is a view of the input ITensor).
""" swapinds(::ITensor, ::Any...)

"""
    adjoint(A::ITensor)

For `A'` notation to prime an ITensor by 1.
"""
Base.adjoint(A::ITensor) = prime(A)

dirs(A::ITensor, is) = dirs(inds(A), is)

function Base.:(==)(A::ITensor, B::ITensor)
  return norm(A - B) == zero(promote_type(eltype(A),eltype(B)))
end

function Base.isapprox(A::ITensor,
                       B::ITensor;
                       kwargs...)
    B = permute(dense(B), inds(A))
    return isapprox(array(A), array(B); kwargs...)
end

function Random.randn!(T::ITensor)
  return randn!(tensor(T))
end

"""
    randomITensor([::Type{ElT <: Number} = Float64, ]inds)

    randomITensor([::Type{ElT <: Number} = Float64, ]inds::Index...)

Construct an ITensor with type `ElT` and indices `inds`, whose elements are normally distributed random numbers. If the element type is not specified, it defaults to `Float64`.
"""
function randomITensor(::Type{S},
                       inds::Indices) where {S <: Number}
  T = ITensor(S, inds)
  randn!(T)
  return T
end

function randomITensor(::Type{S},
                       inds::Index...) where {S <: Number}
  return randomITensor(S, IndexSet(inds...))
end

# To fix ambiguity errors with QN version
function randomITensor(::Type{ElT}) where {ElT <: Number}
  return randomITensor(ElT, IndexSet())
end

randomITensor(inds::Indices) = randomITensor(Float64, inds)

randomITensor(inds::Index...) = randomITensor(Float64,
                                              IndexSet(inds...))

# To fix ambiguity errors with QN version
randomITensor() = randomITensor(Float64, IndexSet())

function combiner(inds::Indices;
                  kwargs...)
  tags = get(kwargs, :tags, "CMB,Link")
  new_ind = Index(prod(dims(inds)), tags)
  new_is = IndexSet(new_ind, inds...)
  return itensor(Combiner(), new_is)
end

combiner(inds::Index...;
         kwargs...) = combiner(IndexSet(inds...); kwargs...)

# Special case when no indices are combined (useful for generic code)
function combiner(; kwargs...)
  return itensor(Combiner(), IndexSet())
end

function combinedind(T::ITensor)
  if store(T) isa Combiner
    return inds(T)[1]
  end
  return nothing
end

combinedind(T::ITensor{0}) = nothing

LinearAlgebra.norm(T::ITensor) = norm(tensor(T))

function dag(T::ITensor; always_copy=false)
  TT = conj(tensor(T); always_copy=always_copy)
  return itensor(store(TT),dag(inds(T)))
end

"""
    permute(T::ITensors, inds; always_copy::Bool = false)
    permute(T::ITensors, inds::Index...; always_copy::Bool = false)

Return a new ITensor T with indices permuted according
to the input indices inds. The storage of the ITensor
is permuted accordingly.

If `always_copy = false`, it avoids copying data if possible.
Therefore, it may return a view. Use `always_copy = true`
if you never want it to return an ITensor with a view
of the original ITensor.
"""
function permute(T::ITensor{N},
                 new_inds; always_copy::Bool = false) where {N}
  perm = NDTensors.getperm(new_inds, inds(T))
  if !always_copy && NDTensors.is_trivial_permutation(perm)
    return T
  end
  Tp = permutedims(tensor(T), perm)
  return itensor(Tp)::ITensor{N}
end

permute(T::ITensor,
        inds::Index...; vargs...) = permute(T,
                                            IndexSet(inds...); vargs...)

Base.:*(T::ITensor, x::Number) = itensor(x*tensor(T))

Base.:*(x::Number, T::ITensor) = T*x

#TODO: make a proper element-wise division
Base.:/(A::ITensor, x::Number) = A*(1.0/x)

Base.:-(A::ITensor) = itensor(-tensor(A))

function Base.:+(A::ITensor{N}, B::ITensor{N}) where {N}
  C = copy(A)
  C .+= B
  return C
end

function Base.:-(A::ITensor{N}, B::ITensor{N}) where {N}
  C = copy(A)
  C .-= B
  return C
end

Base.:+(A::ITensor{Any}, B::ITensor) = copy(B)

Base.:+(A::ITensor, B::ITensor{Any}) = B + A

Base.:+(A::ITensor, B::ITensor) = error("cannot add ITensors with different numbers of indices")
Base.:-(A::ITensor, B::ITensor) = error("cannot subtract ITensors with different numbers of indices")

"""
    *(A::ITensor, B::ITensor)

Contract ITensors A and B to obtain a new ITensor. This 
contraction `*` operator finds all matching indices common
to A and B and sums over them, such that the result will 
have only the unique indices of A and B. To prevent
indices from matching, their prime level or tags can be 
modified such that they no longer compare equal - for more
information see the documentation on Index objects.
"""
function Base.:*(A::ITensor, B::ITensor)
  (labelsA,labelsB) = compute_contraction_labels(inds(A),inds(B))
  CT = contract(tensor(A),labelsA,tensor(B),labelsB)
  C = itensor(CT)
  warnTensorOrder = GLOBAL_PARAMS["WarnTensorOrder"]
  if warnTensorOrder > 0 && order(C) >= warnTensorOrder
    @warn "Contraction resulted in ITensor with $(order(C)) indices"
  end
  return C
end

# TODO: define for contraction order optimization
#Base.:*(A1::ITensor,
#        A2::ITensor,
#        A3::ITensor, As::ITensor...)

function LinearAlgebra.mul!(C::ITensor, A::ITensor, B::ITensor,
                            α::Number, β::Number=0)
  (labelsC,labelsA,labelsB) = compute_contraction_labels(inds(C),
                                                         inds(A),
                                                         inds(B))
  CT = NDTensors.contract!!(tensor(C), labelsC,
                          tensor(A), labelsA,
                          tensor(B), labelsB,
                          α, β)
  C = itensor(CT)
  return C
end

# This is necessary for now since not all types implement contract!!
# with non-trivial α and β
function LinearAlgebra.mul!(C::ITensor, A::ITensor, B::ITensor)
  (labelsC,labelsA,labelsB) = compute_contraction_labels(inds(C),
                                                         inds(A),
                                                         inds(B))
  CT = NDTensors.contract!!(tensor(C), labelsC,
                          tensor(A), labelsA,
                          tensor(B), labelsB)
  C = itensor(CT)
  return C
end

LinearAlgebra.dot(A::ITensor, B::ITensor) = (dag(A)*B)[]

"""
    exp(A::ITensor, Lis; hermitian = false)

Compute the exponential of the tensor `A` by treating it as a matrix ``A_{lr}`` with
the left index `l` running over all indices in `Lis` and `r` running over all
indices not in `Lis`. Must have `dim(Lis) == dim(inds(A))/dim(Lis)` for the exponentiation to
be defined.
When `ishermitian=true` the exponential of `Hermitian(A_{lr})` is
computed internally.
"""
function LinearAlgebra.exp(A::ITensor,
                           Linds,
                           Rinds = prime(IndexSet(Linds));
                           ishermitian = false)
  Lis,Ris = IndexSet(Linds),IndexSet(Rinds)
  Lpos,Rpos = NDTensors.getperms(inds(A), Lis, Ris)
  expAT = exp(tensor(A), Lpos, Rpos; ishermitian=ishermitian)
  return itensor(expAT)
end

"""
    product(A::ITensor, B::ITensor)

For matrix-like ITensors (ones with pairs of primed and
unprimed indices), perform a matrix product, i.e.
```julia
mapprime(prime(A) * B, 2, 1)
```
In the future, more general ITensors with other tag or
prime conventions may be supported.
"""
function product(A::ITensor,
                 B::ITensor)
  R = prime(A) * B
  return mapprime(R,2,1)
end

#######################################################################
#
# In-place operations
#

"""
    normalize!(T::ITensor)

Normalize an ITensor in-place, such that norm(T)==1.
"""
LinearAlgebra.normalize!(T::ITensor) = (T .*= 1/norm(T))

"""
    copyto!(B::ITensor, A::ITensor)

Copy the contents of ITensor A into ITensor B.
```
B .= A
```
"""
function Base.copyto!(R::ITensor{N}, T::ITensor{N}) where {N}
  R .= T
  return R
end

function Base.map!(f::Function,
                   R::ITensor{N},
                   T1::ITensor{N},
                   T2::ITensor{N}) where {N}
  R !== T1 && error("`map!(f, R, T1, T2)` only supports `R === T1` right now")
  perm = NDTensors.getperm(inds(R),inds(T2))
  TR,TT = tensor(R),tensor(T2)

  # TODO: Include type promotion from α
  TR = convert(promote_type(typeof(TR),typeof(TT)),TR)
  TR = permutedims!!(TR,TT,perm,f)

  setstore!(R,store(TR))
  setinds!(R,inds(TR))
  return R
end

"""
    axpy!(a::Number, v::ITensor, w::ITensor)
```
w .+= a .* v
```
"""
LinearAlgebra.axpy!(a::Number,
                    v::ITensor,
                    w::ITensor) = (w .+= a .* v)

"""
axpby!(a,v,b,w)

```
w .= a .* v + b .* w
```
"""
LinearAlgebra.axpby!(a::Number,
                     v::ITensor,
                     b::Number,
                     w::ITensor) = (w .= a .* v + b .* w)

"""
    scale!(A::ITensor,x::Number) = rmul!(A,x)

Scale the ITensor A by x in-place. May also be written `rmul!`.
```
A .*= x
```
"""
NDTensors.scale!(T::ITensor, α::Number) = (T .*= α)

LinearAlgebra.rmul!(T::ITensor, α::Number) = (T .*= α)

LinearAlgebra.lmul!(T::ITensor, α::Number) = (T .= α .* T)

"""
    mul!(A::ITensor, x::Number, B::ITensor)

Scalar multiplication of ITensor B with x, and store the result in A.
Like `A .= x .* B`.
"""
LinearAlgebra.mul!(R::ITensor,
                   α::Number,
                   T::ITensor) = (R .= α .* T)

LinearAlgebra.mul!(R::ITensor,
                   T::ITensor,
                   α::Number) = (R .= T .* α)

#
# Block sparse related functions
# (Maybe create fallback definitions for dense tensors)
#

hasqns(T::ITensor) = hasqns(inds(T))

NDTensors.nnz(T::ITensor) = nnz(tensor(T))

NDTensors.nnzblocks(T::ITensor) = nnzblocks(tensor(T))

NDTensors.nzblock(T::ITensor, args...) = nzblock(tensor(T), args...)

NDTensors.nzblocks(T::ITensor) = nzblocks(tensor(T))

NDTensors.blockoffsets(T::ITensor) = blockoffsets(tensor(T))

flux(T::ITensor, args...) = flux(inds(T), args...)

function NDTensors.addblock!(T::ITensor,
                             args...)
  (!isnothing(flux(T)) && flux(T) ≠ flux(T, args...)) && 
   error("Block does not match current flux")
  TR = addblock!!(tensor(T), args...)
  setstore!(T, store(TR))
  return T
end

"""
    isempty(T::ITensor)

Returns `true` if the ITensor contains no elements.

An ITensor with `Empty` storage always returns `true`.
"""
Base.isempty(T::ITensor) = isempty(tensor(T))

"""
    flux(T::ITensor)

Returns the flux of the ITensor.

If the ITensor is empty or it has no QNs, returns `nothing`.
"""
function flux(T::ITensor)
  (!hasqns(T) || isempty(T)) && return nothing
  bofs = blockoffsets(T)
  block1 = nzblock(bofs, 1)
  return flux(T, block1)
end


#######################################################################
#
# Developer functions
#

# TODO: make versions where the element type can be specified (for type
# inference).
NDTensors.array(T::ITensor) = array(tensor(T))

"""
    matrix(T::ITensor)

Given an ITensor `T` with two indices, returns
a Matrix with a copy of the ITensor's elements,
or a view in the case the the ITensor's storage is Dense.
The ordering of the elements in the Matrix, in
terms of which Index is treated as the row versus
column, depends on the internal layout of the ITensor.
*Therefore this method is intended for developer use
only and not recommended for use in ITensor applications.*
"""
function NDTensors.matrix(T::ITensor{2})
  return array(tensor(T))
end

function NDTensors.vector(T::ITensor{1})
  return array(tensor(T))
end

#######################################################################
#
# Printing, reading and writing ITensors
#

function Base.summary(io::IO,
                      T::ITensor)
  print(io,"ITensor ord=$(order(T))")
  if hasqns(T)
    println(io)
    for i in 1:order(T)
      print(io, inds(T)[i])
      println(io)
    end
  else
    for i in 1:order(T)
      print(io, " ", inds(T)[i])
    end
    println(io)
  end
  print(io, typeof(store(T)))
end

function Base.summary(io::IO,
                      T::ITensor{Any})
  print(io,"ITensor ord=$(order(T))")
  print(io," \n", typeof(inds(T)))
  print(io," \n", typeof(store(T)))
end

# TODO: make a specialized printing from Diag
# that emphasizes the missing elements
function Base.show(io::IO,
                   T::ITensor)
  println(io,"ITensor ord=$(order(T))")
  Base.show(io, MIME"text/plain"(), tensor(T))
end

function Base.show(io::IO,
                   mime::MIME"text/plain",
                   T::ITensor)
  summary(io,T)
end

function readcpp(io::IO,
                 ::Type{Dense{ValT}};
                 kwargs...) where {ValT}
  format = get(kwargs,:format,"v3")
  if format=="v3"
    size = read(io,UInt64)
    data = Vector{ValT}(undef,size)
    for n=1:size
      data[n] = read(io,ValT)
    end
    return Dense(data)
  else
    throw(ArgumentError("read Dense: format=$format not supported"))
  end
end

function readcpp(io::IO,
                 ::Type{ITensor};
                 kwargs...)
  format = get(kwargs,:format,"v3")
  if format=="v3"
    inds = readcpp(io,IndexSet;kwargs...)
    read(io,12) # ignore scale factor by reading 12 bytes
    storage_type = read(io,Int32)
    if storage_type==0 # Null
      store = Dense{Nothing}()
    elseif storage_type==1  # DenseReal
      store = readcpp(io,Dense{Float64};kwargs...)
    elseif storage_type==2  # DenseCplx
      store = readcpp(io,Dense{ComplexF64};kwargs...)
    elseif storage_type==3  # Combiner
      store = CombinerStorage(T.inds[1])
    #elseif storage_type==4  # DiagReal
    #elseif storage_type==5  # DiagCplx
    #elseif storage_type==6  # QDenseReal
    #elseif storage_type==7  # QDenseCplx
    #elseif storage_type==8  # QCombiner
    #elseif storage_type==9  # QDiagReal
    #elseif storage_type==10 # QDiagCplx
    #elseif storage_type==11 # ScalarReal
    #elseif storage_type==12 # ScalarCplx
    else
      throw(ErrorException("C++ ITensor storage type $storage_type not yet supported"))
    end
    return itensor(store,inds)
  else
    throw(ArgumentError("read ITensor: format=$format not supported"))
  end
end

function setwarnorder!(ord::Int)
  ITensors.GLOBAL_PARAMS["WarnTensorOrder"] = ord
end

function HDF5.write(parent::Union{HDF5File,HDF5Group},
                    name::AbstractString,
                    T::ITensor)
  g = g_create(parent,name)
  attrs(g)["type"] = "ITensor"
  attrs(g)["version"] = 1
  write(g,"inds", inds(T))
  write(g,"store", store(T))
end

#function HDF5.read(parent::Union{HDF5File,HDF5Group},
#                   name::AbstractString)
#  g = g_open(parent,name)
#
#  try
#    typestr = read(attrs(g)["type"])
#    type_t = eval(Meta.parse(typestr))
#    res = read(parent,"name",type_t)
#    return res
#  end
#  return 
#end

function HDF5.read(parent::Union{HDF5File,HDF5Group},
                   name::AbstractString,
                   ::Type{ITensor})
  g = g_open(parent,name)
  if read(attrs(g)["type"]) != "ITensor"
    error("HDF5 group or file does not contain ITensor data")
  end
  inds = read(g,"inds",IndexSet)

  stypestr = read(attrs(g_open(g,"store"))["type"])
  stype = eval(Meta.parse(stypestr))

  store = read(g,"store",stype)

  return itensor(store,inds)
end

#
# Deprecations
#

@deprecate findindex(args...; kwargs...) firstind(args...; kwargs...)

@deprecate findinds(args...; kwargs...) inds(args...; kwargs...)

@deprecate commonindex(args...; kwargs...) commonind(args...; kwargs...)

@deprecate uniqueindex(args...; kwargs...) uniqueind(args...; kwargs...)

@deprecate replaceindex!(args...; kwargs...) replaceind!(args...; kwargs...)

@deprecate siteindex(args...; kwargs...) siteind(args...; kwargs...)

@deprecate linkindex(args...; kwargs...) linkind(args...; kwargs...)

@deprecate matmul(A::ITensor, B::ITensor) product(A, B)

