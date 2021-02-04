
"""
    ITensor{N}

An ITensor is a tensor whose interface is 
independent of its memory layout. Therefore
it is not necessary to know the ordering
of an ITensor's indices, only which indices
an ITensor has. Operations like contraction
and addition of ITensors automatically
handle any memory permutations.

# Examples

```julia
julia> i = Index(2, "i")
(dim=2|id=287|"i")

julia> A = randomITensor(i', i)
ITensor ord=2 (dim=2|id=287|"i")' (dim=2|id=287|"i")
NDTensors.Dense{Float64,Array{Float64,1}}

julia> @show A;
A = ITensor ord=2
Dim 1: (dim=2|id=287|"i")'
Dim 2: (dim=2|id=287|"i")
NDTensors.Dense{Float64,Array{Float64,1}}
 2×2
 0.28358594718392427   1.4342219756446355
 1.6620103556283987   -0.40952231269251566

julia> @show inds(A);
inds(A) = IndexSet{2} (dim=2|id=287|"i")' (dim=2|id=287|"i") 

julia> A[i => 1, i' => 2] = 1;

julia> @show A;
A = ITensor ord=2
Dim 1: (dim=2|id=287|"i")'
Dim 2: (dim=2|id=287|"i")
NDTensors.Dense{Float64,Array{Float64,1}}
 2×2
 0.28358594718392427   1.4342219756446355
 1.0                  -0.40952231269251566

julia> @show store(A);
store(A) = [0.28358594718392427, 1.0, 1.4342219756446355, -0.40952231269251566]

julia> B = randomITensor(i, i');

julia> @show B;
B = ITensor ord=2
Dim 1: (dim=2|id=287|"i")
Dim 2: (dim=2|id=287|"i")'
NDTensors.Dense{Float64,Array{Float64,1}}
 2×2
 -0.6510816500352691   0.2579101497658179
  0.256266641521826   -0.9464735926768166

julia> @show A + B;
A + B = ITensor ord=2
Dim 1: (dim=2|id=287|"i")'
Dim 2: (dim=2|id=287|"i")
NDTensors.Dense{Float64,Array{Float64,1}}
 2×2
 -0.3674957028513448   1.6904886171664615
  1.2579101497658178  -1.3559959053693322
```
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
  ITensor{N}(is, st::TensorStorage) where {N} = new{N}(st, is)

  ITensor{Any}(is, st::Empty) = new{Any}(st, is)
end

ITensor{Any}(is, st::TensorStorage) =
  error("Can only make an ITensor with Any number of indices with NDTensors.Empty storage")

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
inds(T::ITensor) = T.inds

"""
    ind(T::ITensor, i::Int)

Get the Index of the ITensor along dimension i.
"""
ind(T::ITensor, i::Int) = inds(T)[i]

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

similar(T::ITensor) = itensor(similar(tensor(T)))

similar(T::ITensor, ::Type{ElT}) where {ElT<:Number} =
  itensor(similar(tensor(T),ElT))

setinds!(T::ITensor,is) = (T.inds = is; return T)

setstore!(T::ITensor,st::TensorStorage) = (T.store = st; return T)

setinds(T::ITensor, is) = itensor(store(T),is)

setstore(T::ITensor, st) = itensor(st,inds(T))

#
# Iteration over ITensors
#

"""
    CartesianIndices(A::ITensor)

Create a CartesianIndices iterator for an ITensor. Helpful for
iterating over all elements of the ITensor.

julia> i = Index(2, "i")
(dim=2|id=90|"i")

julia> j = Index(3, "j")
(dim=3|id=554|"j")

julia> A = randomITensor(i, j)
ITensor ord=2 (dim=2|id=90|"i") (dim=3|id=554|"j")
Dense{Float64,Array{Float64,1}}

julia> C = CartesianIndices(A)
2×3 CartesianIndices{2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}}:
 CartesianIndex(1, 1)  CartesianIndex(1, 2)  CartesianIndex(1, 3)
 CartesianIndex(2, 1)  CartesianIndex(2, 2)  CartesianIndex(2, 3)

julia> for c in C
         @show c, A[c]
       end
(c, A[c]) = (CartesianIndex(1, 1), 0.9867887290267864)
(c, A[c]) = (CartesianIndex(2, 1), -0.5967323222288754)
(c, A[c]) = (CartesianIndex(1, 2), 0.9675791778518225)
(c, A[c]) = (CartesianIndex(2, 2), 0.2842549524334651)
(c, A[c]) = (CartesianIndex(1, 3), -0.023483276282564795)
(c, A[c]) = (CartesianIndex(2, 3), -0.4877709982071688)
"""
CartesianIndices(A::ITensor) = CartesianIndices(inds(A))

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
itensor(T::Tensor) = itensor(store(T), inds(T))

"""
    tensor(::ITensor)

Convert the ITensor to a Tensor that shares the same
storage and indices as the ITensor.
"""
tensor(A::ITensor) = tensor(store(A),inds(A))

"""
    ITensor([::Type{ElT} = Float64, ]inds)
    ITensor([::Type{ElT} = Float64, ]inds::Index...)

Construct an ITensor filled with zeros having indices `inds` and element type `ElT`. If the element type is not specified, it defaults to `Float64`.

The storage will have `NDTensors.Dense` type.
"""
ITensor(::Type{ElT}, inds::Indices) where {ElT <: Number} =
  itensor(Dense(ElT, dim(inds)), inds)

ITensor(::Type{ElT}, inds::Index...) where {ElT <: Number} =
  ITensor(ElT, IndexSet(inds...))

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
ITensor(::Type{ElT}, ::UndefInitializer,
        inds::Indices) where {ElT <: Number} =
  itensor(Dense(ElT, undef, dim(inds)), inds)

ITensor(::Type{ElT}, ::UndefInitializer,
        inds::Index...) where {ElT} =
  ITensor(ElT, undef, IndexSet(inds...))

ITensor(::UndefInitializer, inds::Indices) =
  ITensor(Float64, undef, inds)

ITensor(::UndefInitializer, inds::Index...) =
  ITensor(Float64, undef, IndexSet(inds...))

"""
    ITensor(x::Number, inds)
    ITensor(x::Number, inds::Index...)

Construct an ITensor with all elements set to `float(x)` and indices `inds`.

The storage will have `NDTensors.Dense` type.
"""
ITensor(x::Number, inds::Indices) =
  itensor(Dense(float(x), dim(inds)), inds)

ITensor(x::Number, inds::Index...) = ITensor(x, IndexSet(inds...))

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
diagITensor(::Type{ElT}, is::Indices) where {ElT} =
  itensor(Diag(ElT, mindim(is)), is)

diagITensor(::Type{ElT}, inds::Index...) where {ElT} =
  diagITensor(ElT, IndexSet(inds...))

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
    setelt(ivs...)

Create an ITensor with all zeros except the specified value,
which is set to 1.

# Examples
```julia
i = Index(2,"i")
A = setelt(i=>2)
# A[i=>2] == 1, all other elements zero

j = Index(3,"j")
B = setelt(i=>1,j=>3)
# B[i=>1,j=>3] == 1, all other element zero
```
"""
function setelt(ivs::IndexValOrPairIndexInt...)
  A = emptyITensor(ind.(ivs)...)
  A[val.(ivs)...] = 1.0
  return A
end

"""
    dense(T::ITensor)

Make a new ITensor where the storage is the closest Dense storage,
avoiding allocating new data if possible.
For example, an ITensor with Diag storage will become Dense storage,
filled with zeros except for the diagonal values.
"""
function dense(A::ITensor)
  T = dense(tensor(A))
  return itensor(store(T), removeqns(inds(A)))
end

"""
    complex(T::ITensor)

Convert to the complex version of the storage.
"""
complex(T::ITensor) = itensor(complex(tensor(T)))

function complex!(T::ITensor)
  ct = complex(tensor(T))
  setstore!(T,store(ct))
  setinds!(T,inds(ct))
  return T
end

eltype(T::ITensor) = eltype(tensor(T))

"""
    order(A::ITensor)
    ndims(A::ITensor)

The number of indices, `length(inds(A))`.
"""
order(T::ITensor) = ndims(T)

ndims(::ITensor{N}) where {N} = N

"""
    dim(A::ITensor)

The total dimension of the space the tensor lives in, `prod(dims(A))`.
"""
dim(T::ITensor) = dim(inds(T))

"""
    maxdim(A::ITensor)

The maximum dimension of the tensor indices.
"""
maxdim(T::ITensor) = maxdim(inds(T))

"""
    mindim(A::ITensor)

The minimum dimension of the tensor indices.
"""
mindim(T::ITensor) = mindim(inds(T))

"""
    dim(A::ITensor, n::Int)

Get the nth dimension of the ITensors.
"""
dim(T::ITensor, n::Int) = dims(T)[n]

"""
    dims(A::ITensor)
    size(A::ITensor)

Tuple containing `dim(inds(A)[d]) for d in 1:ndims(A)`.
"""
(dims(T::ITensor{N})::NTuple{N,Int}) where {N} =
  dims(inds(T))


axes(T::ITensor) = map(Base.OneTo, dims(T))

size(T::ITensor) = dims(T)

size(A::ITensor, d::Int) = dim(inds(A), d)

copy(T::ITensor) = itensor(copy(tensor(T)))

"""
    Array{ElT}(T::ITensor, i:Index...)
    Array(T::ITensor, i:Index...)

    Matrix{ElT}(T::ITensor, row_i:Index, col_i::Index)
    Matrix(T::ITensor, row_i:Index, col_i::Index)

    Vector{ElT}(T::ITensor)
    Vector(T::ITensor)

Given an ITensor `T` with indices `i...`, returns
an Array with a copy of the ITensor's elements. The
order in which the indices are provided indicates
the order of the data in the resulting Array.
"""
function Array{ElT, N}(T::ITensor{N},
                       is::Vararg{Index, N}) where {ElT, N}
  TT = tensor(permute(T, is...; always_copy = true))
  return Array{ElT, N}(TT)::Array{ElT, N}
end

function Array{ElT}(T::ITensor{N},
                    is::Vararg{Index, N}) where {ElT, N}
  return Array{ElT, N}(T, is...)
end

function Array(T::ITensor{N},
               is::Vararg{Index, N}) where {N}
  return Array{eltype(T), N}(T, is...)::Array{<:Number, N}
end

function Array{<:Any, N}(T::ITensor{N},
                         is::Vararg{Index, N}) where {N}
  return Array(T, is...)
end

function Vector{ElT}(T::ITensor{1}) where {ElT}
  return Array{ElT}(T,inds(T)...)
end

function Vector(T::ITensor{1})
  return Array(T,inds(T)...)
end

"""
    scalar(T::ITensor)

Extract the element of an order zero ITensor.

Same as `T[]`.
"""
scalar(T::ITensor) = T[]::Number

struct LastVal
  n::Int
end

lastindex(A::ITensor, n::Int64) = LastVal(n)

# Implement when ITensors can be indexed by a single integer
#lastindex(A::ITensor) = dim(A)

lastval_to_int(n::Int, ::LastVal) = n

lastval_to_int(::Int, n::Int) = n

lastval_to_int(T::ITensor, I...) = lastval_to_int.(dims(T), I)

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
function getindex(T::ITensor{N}, I::Vararg{Union{Int, LastVal}, N}) where {N}
  I = lastval_to_int(T, I...)
  @boundscheck checkbounds(tensor(T), I...)
  return tensor(T)[I...]::Number
end

function getindex(T::ITensor{N}, b::Block{N}) where {N}
  # XXX: this should return an ITensor view
  return tensor(T)[b]
end

# Version accepting CartesianIndex, useful when iterating over
# CartesianIndices
getindex(T::ITensor{N}, I::CartesianIndex{N}) where {N} = T[Tuple(I)...]

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
function getindex(T::ITensor, ivs...)
  p = NDTensors.getperm(inds(T), ind.(ivs))
  vals = NDTensors.permute(val.(ivs), p)
  return T[vals...]::Number
end

function getindex(T::ITensor) 
  if order(T) != 0
    throw(DimensionMismatch("In scalar(T) or T[], ITensor T is not a scalar (it has indices $(inds(T)))."))
  end
  return tensor(T)[]::Number
end

"""
    setindex!(T::ITensor, x::Number, I::Int...)

    setindex!(T::ITensor, x::Number, I::CartesianIndex)

Set the specified element of the ITensor, using internal
Index ordering of the ITensor.

# Example
```julia
i = Index(2; tags = "i")
A = ITensor(i, i')
A[1, 2] = 1.0 # same as: A[i => 1, i' => 2] = 1.0
A[2, :] = [2.0 3.0]
```
"""
function setindex!(T::ITensor, x::Number, I::Int...)
  @boundscheck checkbounds(tensor(T), I...)
  fluxT = flux(T)
  if !isnothing(fluxT) && fluxT != flux(T, I...)
    error("In `setindex!`, the element you are trying to set is in a block that does not have the same flux as the other blocks of the ITensor. You may be trying to create an ITensor that does not have a well defined quantum number flux.")
  end
  TR = setindex!!(tensor(T), x, I...)
  setstore!(T, store(TR))
  return T
end

setindex!(T::ITensor, x::Number, I::CartesianIndex) =
  setindex!(T, x, Tuple(I)...)

"""
    setindex!(T::ITensor, x::Number, ivs...)

Set the specified element of the ITensor using a list
of `IndexVal`s or `Pair{<:Index, Int}`.

# Example
```julia
i = Index(2; tags = "i")
A = ITensor(i, i')
A[i => 1, i' => 2] = 1.0 # same as: A[i' => 2, i => 1] = 1.0
A[i => 2, i' => :] = [2.0 3.0]
```
"""
function setindex!(T::ITensor, x::Number, ivs...)
  p = NDTensors.getperm(inds(T), ind.(ivs))
  vals = NDTensors.permute(val.(ivs), p)
  T[vals...] = x
  return T
end

Base.checkbounds(::Any, ::Block) = nothing

function setindex!(T::ITensor, A::AbstractArray, I...)
  @boundscheck checkbounds(tensor(T), I...)
  TR = setindex!!(tensor(T), A, I...)
  setstore!(T, store(TR))
  return T
end

#function setindex!(T::ITensor, A::AbstractArray, b::Block)
#  # XXX: use setindex!! syntax
#  tensor(T)[b] = A
#  return T
#end

function setindex!(T::ITensor, A::AbstractArray,
                   ivs::Pair{<:Index}...)
  input_inds = IndexSet(first.(ivs))
  p = NDTensors.getperm(inds(T), input_inds)
  # Base.to_indices changes Colons into proper ranges, here
  # using the dimensions of the indices.
  vals = to_indices(CartesianIndices(input_inds), last.(ivs))
  # Lazily permute the array to correctly fit into the ITensor,
  # accounting for the input indices being in a different order
  # from the ITensor indices.
  pvals = NDTensors.permute(vals, p)
  T[pvals...] = PermutedDimsArray(reshape(A, length.(vals)), p)
  return T
end

function setindex!(::ITensor{Any}, ::Number, ivs...)
  error("Cannot set the element of an emptyITensor(). Must define indices to set elements")
end

"""
    eachindex(A::ITensor)

Create an iterable object for visiting each element of the ITensor `A` (including structually
zero elements for sparse tensors).

For example, for dense tensors this may return `1:length(A)`, while for sparse tensors
it may return a Cartesian range.
"""
eachindex(A::ITensor) = eachindex(tensor(A))

"""
    iterate(A::ITensor, args...)

Iterate over the elements of an ITensor.
"""
iterate(A::ITensor, args...) = iterate(tensor(A), args...)

"""
    fill!(T::ITensor, x::Number)

Fill all values of the ITensor with the specified value.
"""
function fill!(T::ITensor, x::Number)
  # TODO: automatically switch storage type if needed?
  # Use broadcasting `T .= x`?
  fill!(tensor(T), x)
  return T
end

# TODO: name this `indexset` or `IndexSet`,
# or maybe just `inds`?
itensor2inds(A::ITensor) = inds(A)
itensor2inds(i::Index) = IndexSet(i)
itensor2inds(is::Vector{<:Index}) = IndexSet(is)
itensor2inds(is::Tuple{Vararg{<:Index}}) = IndexSet(is)
itensor2inds(A) = A

# in
hasind(A, i::Index) = i ∈ itensor2inds(A)

# issubset
hasinds(A, is) = is ⊆ itensor2inds(A)
hasinds(A, is::Index...) = hasinds(A, IndexSet(is...))

"""
    hasinds(is...)

Returns an anonymous function `x -> hasinds(x, is...)` which
accepts an ITensor or IndexSet and returns `true` if the
ITensor or IndexSet has the indices `is`.
"""
hasinds(is::Indices) = x -> hasinds(x, is)
hasinds(is::Vector{ <: Index}) = x -> hasinds(x, is)
hasinds(is::Index...) = hasinds(IndexSet(is...))

"""
    hascommoninds(A, B; kwargs...)

    hascommoninds(B; kwargs...) -> f::Function

Check if the ITensors or sets of indices `A` and `B` have
common indices.

If only one ITensor or set of indices `B` is passed, return a
function `f` such that `f(A) = hascommoninds(A, B; kwargs...)`
"""
hascommoninds(A, B; kwargs...) =
  !isnothing(commonind(A, B; kwargs...))

hascommoninds(B; kwargs...) = x -> hascommoninds(x, B; kwargs...)

# issetequal
hassameinds(A, B) =
  issetequal(itensor2inds(A), itensor2inds(B))

# intersect
"""
    commoninds(A, B; kwargs...)
    commoninds(::Order{N}, A, B; kwargs...)

Return an IndexSet with indices that are common between the indices of `A` and `B` (the set intersection, similar to `Base.intersect`).

Optionally, specify the desired number of indices as `Order(N)`, which adds a check and can be a bit more efficient.
"""
commoninds(A...; kwargs...) =
  IndexSet(intersect(itensor2inds.(A)...; kwargs...))

commoninds(::Order{N}, A...; kwargs...) where {N} =
  intersect(Order(N), itensor2inds.(A)...; kwargs...)

# firstintersect
"""
    commonind(A, B; kwargs...)

Return the first `Index` common between the indices of `A` and `B`.

See also [`commoninds`](@ref).
"""
commonind(A...; kwargs...) =
  firstintersect(itensor2inds.(A)...; kwargs...)

# symdiff
"""
    noncommoninds(A, B; kwargs...)
    noncommoninds(::Order{N}, A, B; kwargs...)

Return an IndexSet with indices that are not common between the indices of `A` and `B` (the symmetric set difference, similar to `Base.symdiff`).

Optionally, specify the desired number of indices as `Order(N)`, which adds a check and can be a bit more efficient.
"""
noncommoninds(A...; kwargs...) =
  IndexSet(symdiff(itensor2inds.(A)...; kwargs...)...)

noncommoninds(::Order{N}, A...; kwargs...) where {N} =
  IndexSet{N}(symdiff(itensor2inds.(A)...; kwargs...)...)

# firstsymdiff
"""
    noncommonind(A, B; kwargs...)

Return the first `Index` not common between the indices of `A` and `B`.

See also [`noncommoninds`](@ref).
"""
noncommonind(A...; kwargs...) =
  getfirst(symdiff(itensor2inds.(A)...; kwargs...))

# setdiff
"""
    uniqueinds(A, B; kwargs...)
    uniqueinds(::Order{N}, A, B; kwargs...)

Return an IndexSet with indices that are unique to the set of indices of `A` and not in `B` (the set difference, similar to `Base.setdiff`).

Optionally, specify the desired number of indices as `Order(N)`, which adds a check and can be a bit more efficient.
"""
uniqueinds(A...; kwargs...) =
  IndexSet(setdiff(itensor2inds.(A)...; kwargs...)...)

uniqueinds(::Order{N}, A...; kwargs...) where {N} =
  setdiff(Order(N), ITensors.itensor2inds.(A)...; kwargs...)

# firstsetdiff
"""
    uniqueind(A, B; kwargs...)

Return the first `Index` unique to the set of indices of `A` and not in `B`.

See also [`uniqueinds`](@ref).
"""
uniqueind(A...; kwargs...) =
  firstsetdiff(itensor2inds.(A)...; kwargs...)

# union
"""
    unioninds(A, B; kwargs...)
    unioninds(::Order{N}, A, B; kwargs...)

Return an IndexSet with indices that are the union of the indices of `A` and `B` (the set union, similar to `Base.union`).

Optionally, specify the desired number of indices as `Order(N)`, which adds a check and can be a bit more efficient.
"""
unioninds(A...; kwargs...) =
  IndexSet(union(itensor2inds.(A)...; kwargs...)...)

unioninds(::Order{N}, A...; kwargs...) where {N} =
  IndexSet{N}(union(ITensors.itensor2inds.(A)...; kwargs...)...)

# firstunion
"""
    unionind(A, B; kwargs...)

Return the first `Index` in the union of the indices of `A` and `B`.

See also [`unioninds`](@ref).
"""
unionind(A...; kwargs...) =
  getfirst(union(itensor2inds.(A)...; kwargs...))

firstind(A...; kwargs...) =
  getfirst(itensor2inds.(A)...; kwargs...)

filterinds(A...; kwargs...) =
  filter(itensor2inds.(A)...; kwargs...)

# Faster version when no filtering is requested
filterinds(A::ITensor) = inds(A)

# For backwards compatibility
inds(A...; kwargs...) = filterinds(A...; kwargs...)

# in-place versions of priming and tagging
for fname in (:prime, :setprime, :noprime, :replaceprime, :swapprime,
              :addtags, :removetags, :replacetags, :settags, :swaptags,
              :replaceind, :replaceinds, :swapind, :swapinds)
  @eval begin
    $fname(f::Function, A::ITensor, args...) =
      setinds(A, $fname(f, inds(A), args...))

    $(Symbol(fname, :!))(f::Function, A::ITensor, args...) =
      setinds!(A, $fname(f, inds(A), args...))

    $fname(A::ITensor, args...; kwargs...) =
      setinds(A, $fname(inds(A), args...; kwargs...))

    $(Symbol(fname, :!))(A::ITensor, args...; kwargs...) =
      setinds!(A, $fname(inds(A), args...; kwargs...))
  end
end

priming_tagging_doc = """
Optionally, only modify the indices with the specified keyword arguments.

# Arguments
- `tags = nothing`: if specified, only modify Index `i` if `hastags(i, tags) == true`. 
- `plev = nothing`: if specified, only modify Index `i` if `hasplev(i, plev) == true`.

The ITensor functions come in two versions, `f` and `f!`. The latter modifies the ITensor in-place. In both versions, the ITensor storage is not modified or copied (so it returns an ITensor with a view of the original storage).
"""

@doc """
    prime[!](A::ITensor, plinc::Int = 1; <keyword arguments>) -> ITensor

    prime(is::IndexSet, plinc::Int = 1; <keyword arguments>) -> IndexSet

Increase the prime level of the indices of an ITensor or IndexSet.

$priming_tagging_doc
""" prime(::ITensor, ::Any...)

@doc """
    setprime[!](A::ITensor, plev::Int; <keyword arguments>) -> ITensor

    setprime(is::IndexSet, plev::Int; <keyword arguments>) -> IndexSet

Set the prime level of the indices of an ITensor or IndexSet.

$priming_tagging_doc
""" setprime(::ITensor, ::Any...)

@doc """
    noprime[!](A::ITensor; <keyword arguments>) -> ITensor

    noprime(is::IndexSet; <keyword arguments>) -> IndexSet

Set the prime level of the indices of an ITensor or IndexSet to zero.

$priming_tagging_doc
""" noprime(::ITensor, ::Any...)

@doc """
    replaceprime[!](A::ITensor, plold::Int, plnew::Int; <keyword arguments>) -> ITensor
    replaceprime[!](A::ITensor, plold => plnew; <keyword arguments>) -> ITensor
    mapprime[!](A::ITensor, <arguments>; <keyword arguments>) -> ITensor

    replaceprime(is::IndexSet, plold::Int, plnew::Int; <keyword arguments>) -> IndexSet
    replaceprime(is::IndexSet, plold => plnew; <keyword arguments>) -> IndexSet
    mapprime(is::IndexSet, <arguments>; <keyword arguments>) -> IndexSet

Set the prime level of the indices of an ITensor or IndexSet with prime level `plold` to `plnew`.

$priming_tagging_doc
""" mapprime(::ITensor, ::Any...)

@doc """
    swapprime[!](A::ITensor, pl1::Int, pl2::Int; <keyword arguments>) -> ITensor
    swapprime[!](A::ITensor, pl1 => pl2; <keyword arguments>) -> ITensor

    swapprime(is::ITensor, pl1::Int, pl2::Int; <keyword arguments>) -> IndexSet
    swapprime(is::ITensor, pl1 => pl2; <keyword arguments>) -> IndexSet

Set the prime level of the indices of an ITensor or IndexSetwith prime level `pl1` to `pl2`, and those with prime level `pl2` to `pl1`.

$priming_tagging_doc
""" swapprime(::ITensor, ::Any...)

@doc """
    addtags[!](A::ITensor, ts::String; <keyword arguments>) -> ITensor

    addtags(is::IndexSet, ts::String; <keyword arguments>) -> IndexSet

Add the tags `ts` to the indices of an ITensor or IndexSet.

$priming_tagging_doc
""" addtags(::ITensor, ::Any...)

@doc """
    removetags[!](A::ITensor, ts::String; <keyword arguments>) -> ITensor

    removetags(is::IndexSet, ts::String; <keyword arguments>) -> IndexSet

Remove the tags `ts` from the indices of an ITensor or IndexSet.

$priming_tagging_doc
""" removetags(::ITensor, ::Any...)

@doc """
    settags[!](A::ITensor, ts::String; <keyword arguments>) -> ITensor

    settags(is::IndexSet, ts::String; <keyword arguments>) -> IndexSet

Set the tags of the indices of an ITensor or IndexSet to `ts`.

$priming_tagging_doc
""" settags(::ITensor, ::Any...)

@doc """
    replacetags[!](A::ITensor, tsold::String, tsnew::String; <keyword arguments>) -> ITensor

    replacetags(is::IndexSet, tsold::String, tsnew::String; <keyword arguments>) -> IndexSet

Replace the tags `tsold` with `tsnew` for the indices of an ITensor.

$priming_tagging_doc
""" replacetags(::ITensor, ::Any...)

@doc """
    swaptags[!](A::ITensor, ts1::String, ts2::String; <keyword arguments>) -> ITensor

    swaptags(is::IndexSet, ts1::String, ts2::String; <keyword arguments>) -> IndexSet

Swap the tags `ts1` with `ts2` for the indices of an ITensor.

$priming_tagging_doc
""" swaptags(::ITensor, ::Any...)

@doc """
    replaceind[!](A::ITensor, i1::Index, i2::Index) -> ITensor

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

# XXX: rename to:
# hastags(any, A, ts)
"""
    anyhastags(A::ITensor, ts::Union{String, TagSet})
    hastags(A::ITensor, ts::Union{String, TagSet})

Check if any of the indices in the ITensor have the specified tags.
"""
anyhastags(A::ITensor, ts) = anyhastags(inds(A), ts)

hastags(A::ITensor, ts) = hastags(inds(A), ts)

# XXX: rename to:
# hastags(all, A, ts)
"""
    allhastags(A::ITensor, ts::Union{String, TagSet})

Check if all of the indices in the ITensor have the specified tags.
"""
allhastags(A::ITensor, ts) = allhastags(inds(A), ts)

"""
    adjoint(A::ITensor)

For `A'` notation to prime an ITensor by 1.
"""
adjoint(A::ITensor) = prime(A)

dirs(A::ITensor, is) = dirs(inds(A), is)

function (A::ITensor == B::ITensor)
  return norm(A - B) == zero(promote_type(eltype(A),eltype(B)))
end

function isapprox(A::ITensor, B::ITensor; kwargs...)
  B = permute(dense(B), inds(A))
  return isapprox(array(A), array(B); kwargs...)
end

randn!(T::ITensor) = randn!(tensor(T))

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

norm(T::ITensor) = norm(tensor(T))

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

function (T::ITensor{N} * x::Number)::ITensor{N} where {N}
  return itensor(x * tensor(T))
end

(x::Number * T::ITensor) = T * x

#TODO: make a proper element-wise division
(A::ITensor / x::Number) = A*(1.0/x)

-(A::ITensor) = itensor(-tensor(A))

function (A::ITensor{N} + B::ITensor{N}) where {N}
  C = copy(A)
  C .+= B
  return C
end

function (A::ITensor{N} - B::ITensor{N}) where {N}
  C = copy(A)
  C .-= B
  return C
end

(A::ITensor{Any} + B::ITensor) = copy(B)

(A::ITensor + B::ITensor{Any}) = B + A

(A::ITensor + B::ITensor) =
  error("cannot add ITensors with different numbers of indices")

(A::ITensor - B::ITensor) =
  error("cannot subtract ITensors with different numbers of indices")

function _contract(A::ITensor, B::ITensor)
  (labelsA,labelsB) = compute_contraction_labels(inds(A),inds(B))
  CT = contract(tensor(A),labelsA,tensor(B),labelsB)
  C = itensor(CT)
  warnTensorOrder = get_warn_order()
  if !isnothing(warnTensorOrder) > 0 &&
     order(C) >= warnTensorOrder
     println("Contraction resulted in ITensor with $(order(C)) indices, which is greater than or equal to the ITensor order warning threshold $warnTensorOrder. You can modify the threshold with macros like `@set_warn_order N`, `@reset_warn_order`, and `@disable_warn_order` or functions like `ITensors.set_warn_order(N::Int)`, `ITensors.reset_warn_order()`, and `ITensors.disable_warn_order()`.")
     # This prints a vector, not formatted well
     #show(stdout, MIME"text/plain"(), stacktrace())
     Base.show_backtrace(stdout, backtrace())
     println()
  end
  return C
end

_contract(T::ITensor, ::Nothing) = T

dag(::Nothing) = nothing

iscombiner(T::ITensor) = (store(T) isa Combiner)

isdiag(T::ITensor) = (store(T) isa Diag || store(T) isa DiagBlockSparse)

function can_combine_contract(A::ITensor, B::ITensor)
  return hasqns(A) && hasqns(B) &&
         !iscombiner(A) && !iscombiner(B) &&
         !isdiag(A) && !isdiag(B)
end

function combine_contract(A::ITensor, B::ITensor)
  # Combine first before contracting
  C = if can_combine_contract(A, B)
    uniqueindsA = uniqueinds(A, B)
    uniqueindsB = uniqueinds(B, A)
    commonindsAB = commoninds(A, B)
    combinerA = isempty(uniqueindsA) ? nothing : combiner(uniqueindsA)
    combinerB = isempty(uniqueindsB) ? nothing : combiner(uniqueindsB)
    combinerAB = isempty(commonindsAB) ? nothing : combiner(commonindsAB)
    AC = _contract(_contract(A, combinerA), combinerAB)
    BC = _contract(_contract(B, combinerB), dag(combinerAB))
    CC = _contract(AC, BC)
    _contract(_contract(CC, dag(combinerA)), dag(combinerB))
  else
    _contract(A, B)
  end
  return C
end

"""
    A::ITensor * B::ITensor

Contract ITensors A and B to obtain a new ITensor. This 
contraction `*` operator finds all matching indices common
to A and B and sums over them, such that the result will 
have only the unique indices of A and B. To prevent
indices from matching, their prime level or tags can be 
modified such that they no longer compare equal - for more
information see the documentation on Index objects.
"""
function (A::ITensor * B::ITensor)
  C = using_combine_contract() ? combine_contract(A, B) : _contract(A, B)
  return C
end

function (A::ITensor{0} * B::ITensor)
  return iscombiner(A) ? _contract(A, B) : A[] * B
end

(A::ITensor * B::ITensor{0}) = B * A

function (A::ITensor{0} * B::ITensor{0})
  return (iscombiner(A) || iscombiner(B)) ? _contract(A, B) : ITensor(A[] * B[])
end

# TODO: define for contraction order optimization
#*(A1::ITensor,
#  A2::ITensor,
#  A3::ITensor, As::ITensor...)

# XXX: rename contract!
function mul!(C::ITensor, A::ITensor, B::ITensor,
              α::Number, β::Number=0)
  labelsCAB = compute_contraction_labels(inds(C), inds(A), inds(B))
  labelsC, labelsA, labelsB = labelsCAB
  CT = NDTensors.contract!!(tensor(C), labelsC, tensor(A), labelsA,
                            tensor(B), labelsB, α, β)
  setstore!(C, store(CT))
  setinds!(C, inds(C))
  return C
end

# This is necessary for now since not all types implement contract!!
# with non-trivial α and β
function mul!(C::ITensor, A::ITensor, B::ITensor)
  labelsCAB = compute_contraction_labels(inds(C), inds(A), inds(B))
  labelsC, labelsA, labelsB = labelsCAB
  CT = NDTensors.contract!!(tensor(C), labelsC, tensor(A), labelsA,
                            tensor(B), labelsB)
  setstore!(C, store(CT))
  setinds!(C, inds(C))
  return C
end

dot(A::ITensor, B::ITensor) = (dag(A)*B)[]

# Returns a tuple of pairs of indices, where the pairs
# are determined by the prime level pairs `plev` and
# tag pairs `tags`.
function indpairs(T::ITensor; plev::Pair{Int, Int} = 0 => 1,
                  tags::Pair = ts"" => ts"")
  is1 = filterinds(T; plev = first(plev), tags = first(tags))
  is2 = filterinds(T, plev = last(plev), tags = last(tags))
  is2to1 = replacetags(mapprime(is2, last(plev) => first(plev)),
                       last(tags) => first(tags))
  is_first = commoninds(is1, is2to1)
  is_last = replacetags(mapprime(is_first, first(plev) => last(plev)),
                        first(tags) => last(tags))
  is_last = permute(commoninds(T, is_last), is_last)
  return Tuple(is_first) .=> Tuple(is_last)
end

# Trace an ITensor over pairs of indices determined by
# the prime levels and tags. Indices that are not in pairs
# are not traced over, corresponding to a "batched" trace.
function tr(T::ITensor; plev::Pair{Int, Int} = 0 => 1,
            tags::Pair = ts"" => ts"")
  trpairs = indpairs(T; plev = plev, tags = tags)
  for indpair in trpairs
    T *= δ(dag.(Tuple(indpair)))
  end
  if order(T) == 0
    return T[]
  end
  return T
end

"""
    exp(A::ITensor, Linds=Rinds', Rinds=inds(A,plev=0); ishermitian = false)

Compute the exponential of the tensor `A` by treating it as a matrix ``A_{lr}`` with
the left index `l` running over all indices in `Linds` and `r` running over all
indices in `Rinds`.

Only accepts index lists `Linds`,`Rinds` such that: (1) `length(Linds) +
length(Rinds) == length(inds(A))` (2) `length(Linds) == length(Rinds)` (3) For
each pair of indices `(Linds[n],Rinds[n])`, `Linds[n]` and `Rinds[n]` represent
the same Hilbert space (the same QN structure in the QN case, or just the same
length in the dense case), and appear in `A` with opposite directions.

When `ishermitian=true` the exponential of `Hermitian(A_{lr})` is
computed internally.
"""
function exp(A::ITensor{N}, Linds, Rinds; kwargs...) where {N}
  ishermitian=get(kwargs,:ishermitian,false)

  @debug_check begin
    if hasqns(A)
      @assert flux(A) == QN()
    end
  end

  NL = length(Linds)
  NR = length(Rinds)
  NL != NR && error("Must have equal number of left and right indices")
  N != NL + NR && error("Number of left and right indices must add up to total number of indices")

  # Linds, Rinds may not have the correct directions
  Lis = IndexSet(Linds...)
  Ris = IndexSet(Rinds...)

  # Ensure the indices have the correct directions,
  # QNs, etc.
  # First grab the indices in A, then permute them
  # correctly.
  Lis = permute(commoninds(A, Lis), Lis)
  Ris = permute(commoninds(A, Ris), Ris)

  for (l, r) in zip(Lis, Ris)
    if space(l) != space(r)
      error("In exp, indices must come in pairs with equal spaces.")
    end
    if hasqns(A)
      if dir(l) == dir(r)
        error("In exp, indices must come in pairs with opposite directions")
      end
    end
  end

  CL = combiner(Lis...; dir = Out)
  CR = combiner(Ris...; dir = In)
  AC = A * CR * CL
  expAT = ishermitian ? exp(Hermitian(tensor(AC))) : exp(tensor(AC))
  return itensor(expAT) * dag(CR) * dag(CL)
end

function exp(A::ITensor; kwargs...)
  Ris = filterinds(A; plev = 0)
  Lis = Ris'
  return exp(A, Lis, Ris; kwargs...)
end

"""
    hadamard_product!(C::ITensor{N}, A::ITensor{N}, B::ITensor{N})
    hadamard_product(A::ITensor{N}, B::ITensor{N})
    ⊙(A::ITensor{N}, B::ITensor{N})

Elementwise product of 2 ITensors with the same indices.

Alternative syntax `⊙` can be typed in the REPL with `\\odot <tab>`.
"""
function hadamard_product!(R::ITensor{N},
                           T1::ITensor{N},
                           T2::ITensor{N}) where {N}
  if !hassameinds(T1, T2)
    error("ITensors must have some indices to perform Hadamard product")
  end
  # Permute the indices to the same order
  #if inds(A) ≠ inds(B)
  #  B = permute(B, inds(A))
  #end
  #tensor(C) .= tensor(A) .* tensor(B)
  map!((t1, t2) -> *(t1, t2), R, T1, T2)
  return R
end

# TODO: instead of copy, use promote(A, B)
function hadamard_product(A::ITensor, B::ITensor)
  Ac = copy(A)
  return hadamard_product!(Ac, Ac, B)
end

⊙(A::ITensor, B::ITensor) = hadamard_product(A, B)

"""
    product(A::ITensor, B::ITensor)

Get the product of ITensor `A` and ITensor `B`, which
roughly speaking is a matrix-matrix product, a
matrix-vector product, or a vector-matrix product,
depending on the index structure.

There are three main modes:

1. Matrix-matrix product. In this case, ITensors `A`
and `B` have shared indices that come in pairs of primed
and unprimed indices. Then, `A` and `B` are multiplied 
together, treating them as matrices from the unprimed
to primed indices, resulting in an ITensor `C` that
has the same pairs of primed and unprimed indices. 
For example:
```
s1'-<-----<-s1            s1'-<-----<-s1   s1'-<-----<-s1
      |C|      = product(       |A|              |B|      )
s2'-<-----<-s2            s2'-<-----<-s2 , s2'-<-----<-s2
```
Essentially, this is implemented as 
`C = mapprime(A', B, 2 => 1)`.
If there are dangling indices that are not shared between
`A` and `B`, a "batched" matrix multiplication is
performed, i.e.:
```
       j                         j
       |                         |
s1'-<-----<-s1            s1'-<-----<-s1   s1'-<-----<-s1
      |C|      = product(       |A|              |B|      )
s2'-<-----<-s2            s2'-<-----<-s2 , s2'-<-----<-s2
```
In addition, if there are shared dangling indices,
they are summed over:
```
                                    j                j
                                    |                |
s1'-<-----<-s1               s1'-<-----<-s1   s1'-<-----<-s1
      |C|      = Σⱼ product(       |A|              |B|      )
s2'-<-----<-s2               s2'-<-----<-s2 , s2'-<-----<-s2
```
where the sum is not performed as an explicitly 
for-loop, but as part of a single tensor contraction.

2. Matrix-vector product. In this case, ITensor `A`
has pairs of primed and unprimed indices, and ITensor
`B` has unprimed indices that are shared with `A`.
Then, `A` and `B` are multiplied as a matrix-vector
product, and the result `C` has unprimed indices.
For example:
```
s1-<----            s1'-<-----<-s1   s1-<----
     |C| = product(       |A|             |B| )
s2-<----            s2'-<-----<-s2 , s2-<----
```
Again, like in the matrix-matrix product above, you can have
dangling indices to do "batched" matrix-vector products, or
sum over a batch of matrix-vector products.

3. Vector-matrix product. In this case, ITensor `B`
has pairs of primed and unprimed indices, and ITensor
`A` has unprimed indices that are shared with `B`.
Then, `B` and `A` are multiplied as a matrix-vector
product, and the result `C` has unprimed indices.
For example:
```
---<-s1            ----<-s1   s1'-<-----<-s1
|C|     = product( |A|              |B|      )
---<-s2            ----<-s2 , s2'-<-----<-s2
```
Again, like in the matrix-matrix product above, you can have
dangling indices to do "batched" vector-matrix products, or
sum over a batch of vector-matrix products.

4. Vector-vector product. In this case, ITensors `A`
and `B` share unprimed indices.
Then, `B` and `A` are multiplied as a vector-vector
product, and the result `C` is a scalar ITensor.
For example:
```
---            ----<-s1   s1-<----
|C| = product( |A|             |B| )
---            ----<-s2 , s2-<----
```
Again, like in the matrix-matrix product above, you can have
dangling indices to do "batched" vector-vector products, or
sum over a batch of vector-vector products.
"""
function product(A::ITensor, B::ITensor; apply_dag::Bool = false)
  commonindsAB = commoninds(A, B; plev = 0)
  isempty(commonindsAB) && error("In product, must have common indices with prime level 0.")
  common_paired_indsA = filterinds(i -> hasind(commonindsAB, i) &&
                                        hasind(A, setprime(i, 1)), A)
  common_paired_indsB = filterinds(i -> hasind(commonindsAB, i) &&
                                        hasind(B, setprime(i, 1)), B)

  if !isempty(common_paired_indsA)
    commoninds_pairs = unioninds(common_paired_indsA,
                                 common_paired_indsA')
  elseif !isempty(common_paired_indsB)
    commoninds_pairs = unioninds(common_paired_indsB,
                                 common_paired_indsB')
  else
    # vector-vector product
    apply_dag && error("apply_dag not supported for vector-vector product")
    return A * B
  end
  danglings_indsA = uniqueinds(A, commoninds_pairs)
  danglings_indsB = uniqueinds(B, commoninds_pairs)
  danglings_inds = unioninds(danglings_indsA, danglings_indsB)
  if hassameinds(common_paired_indsA, common_paired_indsB)
    # matrix-matrix product
    A′ = prime(A; inds = !danglings_inds)
    AB = mapprime(A′ * B, 2 => 1; inds = !danglings_inds)
    if apply_dag
      AB′ = prime(AB; inds = !danglings_inds)
      Adag = swapprime(dag(A), 0 => 1; inds = !danglings_inds)
      return mapprime(AB′ * Adag, 2 => 1; inds = !danglings_inds)
    end
    return AB
  elseif isempty(common_paired_indsA) && !isempty(common_paired_indsB)
    # vector-matrix product
    apply_dag && error("apply_dag not supported for matrix-vector product")
    A′ = prime(A; inds = !danglings_inds)
    return A′ * B
  elseif !isempty(common_paired_indsA) && isempty(common_paired_indsB)
    # matrix-vector product
    apply_dag && error("apply_dag not supported for vector-matrix product")
    return noprime(A * B; inds = !danglings_inds)
  end
end

"""
    product(As::Vector{<:ITensor}, A::ITensor)

Product the ITensors pairwise.
"""
function product(As::Vector{<: ITensor}, B::ITensor; kwargs...)
  AB = B
  for A in As
    AB = product(A, AB; kwargs...)
  end
  return AB
end

# Alias apply with product
const apply = product

#######################################################################
#
# In-place operations
#

"""
    normalize!(T::ITensor)

Normalize an ITensor in-place, such that norm(T)==1.
"""
normalize!(T::ITensor) = (T .*= 1/norm(T))

"""
    copyto!(B::ITensor, A::ITensor)

Copy the contents of ITensor A into ITensor B.
```
B .= A
```
"""
function copyto!(R::ITensor{N}, T::ITensor{N}) where {N}
  R .= T
  return R
end

function map!(f::Function,
              R::ITensor{N},
              T1::ITensor{N},
              T2::ITensor{N}) where {N}
  R !== T1 && error("`map!(f, R, T1, T2)` only supports `R === T1` right now")
  perm = NDTensors.getperm(inds(R),inds(T2))

  if hasqns(T2) && hasqns(R)
    # Check that Index arrows match
    for (n,p) in enumerate(perm)
      if dir(inds(R)[n]) != dir(inds(T2)[p])
        #println("Mismatched Index: \n$(inds(R)[n])")
        error("Index arrows must be the same to add, subtract, map, or scale QN ITensors")
      end
    end
  end

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
axpy!(a::Number, v::ITensor, w::ITensor) =
  (w .+= a .* v)

"""
axpby!(a,v,b,w)

```
w .= a .* v + b .* w
```
"""
axpby!(a::Number, v::ITensor, b::Number, w::ITensor) =
  (w .= a .* v + b .* w)

"""
    scale!(A::ITensor,x::Number) = rmul!(A,x)

Scale the ITensor A by x in-place. May also be written `rmul!`.
```
A .*= x
```
"""
scale!(T::ITensor, α::Number) = (T .*= α)

rmul!(T::ITensor, α::Number) = (T .*= α)

lmul!(T::ITensor, α::Number) = (T .= α .* T)

"""
    mul!(A::ITensor, x::Number, B::ITensor)

Scalar multiplication of ITensor B with x, and store the result in A.
Like `A .= x .* B`.
"""
mul!(R::ITensor, α::Number, T::ITensor) = (R .= α .* T)

mul!(R::ITensor, T::ITensor, α::Number) = (R .= T .* α)

#
# Block sparse related functions
# (Maybe create fallback definitions for dense tensors)
#

hasqns(T::ITensor) = hasqns(inds(T))

eachnzblock(T::ITensor) = eachnzblock(tensor(T))

nnz(T::ITensor) = nnz(tensor(T))

nnzblocks(T::ITensor) = nnzblocks(tensor(T))

nzblock(T::ITensor, args...) = nzblock(tensor(T), args...)

nzblocks(T::ITensor) = nzblocks(tensor(T))

blockoffsets(T::ITensor) = blockoffsets(tensor(T))

flux(T::ITensor, args...) = flux(inds(T), args...)

"""
    flux(T::ITensor)

Returns the flux of the ITensor.

If the ITensor is empty or it has no QNs, returns `nothing`.
"""
function flux(T::ITensor)
  (!hasqns(T) || isempty(T)) && return nothing
  @debug_check checkflux(T)
  block1 = first(eachnzblock(T))
  return flux(T, block1)
end

function checkflux(T::ITensor, flux_check)
  for b in nzblocks(T)
    fluxTb = flux(T, b)
    if fluxTb != flux_check
      error("Block $b has flux $fluxTb that is inconsistent with the desired flux $flux_check")
    end
  end
  return nothing
end

function checkflux(T::ITensor)
  b1 = first(nzblocks(T))
  fluxTb1 = flux(T, b1)
  return checkflux(T, fluxTb1)
end

function insertblock!(T::ITensor, args...)
  (!isnothing(flux(T)) && flux(T) ≠ flux(T, args...)) && 
   error("Block does not match current flux")
  TR = insertblock!!(tensor(T), args...)
  setstore!(T, store(TR))
  return T
end

"""
    isempty(T::ITensor)

Returns `true` if the ITensor contains no elements.

An ITensor with `Empty` storage always returns `true`.
"""
isempty(T::ITensor) = isempty(tensor(T))


#######################################################################
#
# Developer functions
#

"""
    array(T::ITensor)

Given an ITensor `T`, returns
an Array with a copy of the ITensor's elements,
or a view in the case the the ITensor's storage is Dense.
The ordering of the elements in the Array, in
terms of which Index is treated as the row versus
column, depends on the internal layout of the ITensor.
*Therefore this method is intended for developer use
only and not recommended for use in ITensor applications.*
"""
array(T::ITensor) = array(tensor(T))

"""
    matrix(T::ITensor)

Given an ITensor `T` with two indices, returns
a Matrix with a copy of the ITensor's elements,
or a view in the case the ITensor's storage is Dense.
The ordering of the elements in the Matrix, in
terms of which Index is treated as the row versus
column, depends on the internal layout of the ITensor.
*Therefore this method is intended for developer use
only and not recommended for use in ITensor applications.*
"""
matrix(T::ITensor{2}) = array(tensor(T))

"""
    vector(T::ITensor)

Given an ITensor `T` with one index, returns
a Vector with a copy of the ITensor's elements,
or a view in the case the ITensor's storage is Dense.
"""
vector(T::ITensor{1}) = array(tensor(T))

#######################################################################
#
# Printing, reading and writing ITensors
#

function summary(io::IO, T::ITensor)
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

function summary(io::IO, T::ITensor{Any})
  print(io,"ITensor ord=$(order(T))")
  print(io," \n", typeof(inds(T)))
  print(io," \n", typeof(store(T)))
end

# TODO: make a specialized printing from Diag
# that emphasizes the missing elements
function show(io::IO, T::ITensor)
  println(io,"ITensor ord=$(order(T))")
  show(io, MIME"text/plain"(), tensor(T))
end

function show(io::IO,
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

function HDF5.write(parent::Union{HDF5.File,HDF5.Group},
                    name::AbstractString,
                    T::ITensor)
  g = create_group(parent,name)
  attributes(g)["type"] = "ITensor"
  attributes(g)["version"] = 1
  write(g,"inds", inds(T))
  write(g,"store", store(T))
end

#function HDF5.read(parent::Union{HDF5.File,HDF5.Group},
#                   name::AbstractString)
#  g = open_group(parent,name)
#
#  try
#    typestr = read(attributes(g)["type"])
#    type_t = eval(Meta.parse(typestr))
#    res = read(parent,"name",type_t)
#    return res
#  end
#  return 
#end

function HDF5.read(parent::Union{HDF5.File,HDF5.Group},
                   name::AbstractString,
                   ::Type{ITensor})
  g = open_group(parent,name)
  if read(attributes(g)["type"]) != "ITensor"
    error("HDF5 group or file does not contain ITensor data")
  end
  inds = read(g,"inds",IndexSet)

  stypestr = read(attributes(open_group(g,"store"))["type"])
  stype = eval(Meta.parse(stypestr))

  store = read(g,"store",stype)

  return itensor(store,inds)
end

