"""
    ITensor

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

#
# Make an ITensor with random elements:
#
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
inds(A) = ((dim=2|id=287|"i")', (dim=2|id=287|"i"))

#
# Set the i==1, i'==2 element to 1.0:
#
julia> A[i => 1, i' => 2] = 1;

julia> @show A;
A = ITensor ord=2
Dim 1: (dim=2|id=287|"i")'
Dim 2: (dim=2|id=287|"i")
NDTensors.Dense{Float64,Array{Float64,1}}
 2×2
 0.28358594718392427   1.4342219756446355
 1.0                  -0.40952231269251566

julia> @show storage(A);
storage(A) = [0.28358594718392427, 1.0, 1.4342219756446355, -0.40952231269251566]

julia> B = randomITensor(i, i');

julia> @show B;
B = ITensor ord=2
Dim 1: (dim=2|id=287|"i")
Dim 2: (dim=2|id=287|"i")'
NDTensors.Dense{Float64,Array{Float64,1}}
 2×2
 -0.6510816500352691   0.2579101497658179
  0.256266641521826   -0.9464735926768166

#
# Can add or subtract ITensors as long as they
# have the same indices, in any order:
#
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
## TypeDefs
const RealOrComplex{T} = Union{T,Complex{T}}
##

## The categories in this file are
## constructors, properties, iterators,
## Accessor Functions and Operations
mutable struct ITensor
  tensor::Tensor

  function ITensor(::AllowAlias, T::Tensor{<:Any,<:Any,<:TensorStorage,<:Tuple})
    @debug_check begin
      is = inds(T)
      if !allunique(is)
        error(
          "Trying to create ITensors with collection of indices $is. Indices must be unique.",
        )
      end
    end
    return new(T)
  end
end

"""
    Tensor(::ITensor)

Create a `Tensor` that stores a copy of the storage and
indices of the input `ITensor`.
"""
Tensor(T::ITensor)::Tensor = Tensor(NeverAlias(), T)
Tensor(::NeverAlias, T::ITensor)::Tensor = Tensor(AllowAlias(), copy(T))

"""
    tensor(::ITensor)

Convert the `ITensor` to a `Tensor` that shares the same
storage and indices as the `ITensor`.
"""
Tensor(::AllowAlias, A::ITensor) = A.tensor

#########################
# ITensor constructors
#

## ITensors constructor hierarchy: 
## Tensor; TensorStorage; Data/Datatype; Element/Eltype; Indices 

# Version where the indices are not Tuple, so convert to Tuple
## This is unreachable code as NDTensors.Tensor is implemented
# function ITensor(::AllowAlias, T::Tensor)::ITensor
#   return ITensor(AllowAlias(), setinds(T, NTuple{ndims(T)}(inds(T))))
# end

ITensor(::NeverAlias, T::Tensor)::ITensor = ITensor(AllowAlias(), copy(T))

ITensor(T::Tensor)::ITensor = ITensor(NeverAlias(), T)

itensor(T::ITensor) = T
ITensor(T::ITensor) = copy(T)

"""
    ITensor(st::TensorStorage, is)

Constructor for an ITensor from a TensorStorage
and a set of indices.
The ITensor stores a view of the TensorStorage.
"""
ITensor(as::AliasStyle, st::TensorStorage, is)::ITensor =
  ITensor(as, Tensor(as, st, Tuple(is)))
ITensor(as::AliasStyle, is, st::TensorStorage)::ITensor = ITensor(as, st, is)

ITensor(st::TensorStorage, is)::ITensor = ITensor(NeverAlias(), st, is)

ITensor(is, st::TensorStorage)::ITensor = ITensor(NeverAlias(), st, is)

"""
    ITensor([ElT::Type, ]A::Array, inds)
    ITensor([ElT::Type, ]A::Array, inds::Index...)

    itensor([ElT::Type, ]A::Array, inds)
    itensor([ElT::Type, ]A::Array, inds::Index...)

Construct an ITensor from an Array `A` and indices `inds`.
The ITensor will be a view of the Array data if possible (if
no conversion to a different element type is necessary).

If specified, the ITensor will have element type `ElT`.

If the element type of `A` is `Int` or `Complex{Int}` and
the desired element type isn't specified, it will
be converted to `Float64` or `Complex{Float64}` automatically.
To keep the element type as an integer, specify it explicitly,
for example with:
```julia
i = Index(2, "i")
A = [0 1; 1 0]
T = ITensor(eltype(A), A, i', dag(i))
```

# Examples

```julia
i = Index(2,"index_i")
j = Index(2,"index_j")

M = [1. 2;
     3 4]
T = ITensor(M, i, j)
T[i => 1, j => 1] = 3.3
M[1, 1] == 3.3
T[i => 1, j => 1] == 3.3
```

!!! warning
    In future versions this may not automatically convert `Int`/`Complex{Int}` inputs to floating point versions with `float` (once tensor operations using `Int`/`Complex{Int}` are natively as fast as floating point operations), and in that case the particular element type should not be relied on. To avoid extra conversions (and therefore allocations) it is best practice to directly construct with `itensor([0. 1; 1 0], i', dag(i))` if you want a floating point element type. The conversion is done as a performance optimization since often tensors are passed to BLAS/LAPACK and need to be converted to floating point types compatible with those libraries, but future projects in Julia may allow for efficient operations with more general element types (for example see https://github.com/JuliaLinearAlgebra/Octavian.jl).
"""
function ITensor(
  as::AliasStyle,
  elt::Type{<:Number},
  A::AbstractArray{<:Number},
  inds::Indices;
  kwargs...,
)
  length(A) ≠ dim(inds) && throw(
    DimensionMismatch(
      "In ITensor(::AbstractArray, inds), length of AbstractArray ($(length(A))) must match total dimension of IndexSet ($(dim(inds)))",
    ),
  )
  data = set_eltype(typeof(A), elt)(as, A)
  return ITensor(as, NDTensors.default_storagetype(typeof(data), inds)(data), inds)
end

function ITensor(
  as::AliasStyle, elt::Type{<:Number}, A::AbstractArray{<:Number}, inds; kwargs...
)
  is = indices(inds)
  if !isa(is, Indices)
    error("Indices $inds are not valid for constructing an ITensor.")
  end
  return ITensor(as, elt, A, is; kwargs...)
end

# Convert `Adjoint` to `Matrix`
## TODO: This might have issues for different backends since Matrix converts to Base.Matrix only
function ITensor(
  as::AliasStyle, elt::Type{<:Number}, A::Adjoint, inds::Indices{Index{Int}}; kwargs...
)
  return ITensor(as, elt, Matrix(A), inds; kwargs...)
end

function ITensor(
  as::AliasStyle, elt::Type{<:Number}, A::AbstractArray{<:Number}, is...; kwargs...
)
  return ITensor(as, elt, A, indices(is...); kwargs...)
end

function ITensor(
  as::AliasStyle, A::AbstractArray{ElT}, is...; kwargs...
) where {ElT<:Number}
  return ITensor(as, ElT, A, indices(is...); kwargs...)
end

function ITensor(
  as::AliasStyle, A::AbstractArray{ElT}, is...; kwargs...
) where {ElT<:RealOrComplex{Int}}
  return ITensor(as, float(ElT), A, is...; kwargs...)
end

function ITensor(elt::Type{<:Number}, A::AbstractArray{<:Number}, is...; kwargs...)
  return ITensor(NeverAlias(), elt, A, is...; kwargs...)
end

function ITensor(A::AbstractArray{<:Number}, is...; kwargs...)
  return ITensor(NeverAlias(), A, is...; kwargs...)
end

"""
    ITensor([::Type{ElT} = Float64, ]inds)
    ITensor([::Type{ElT} = Float64, ]inds::Index...)

Construct an ITensor filled with zeros having indices `inds` and element type
`ElT`. If the element type is not specified, it defaults to `Float64`.

The storage will have `NDTensors.Dense` type.

# Examples

```julia
i = Index(2,"index_i")
j = Index(4,"index_j")
k = Index(3,"index_k")

A = ITensor(i,j)
B = ITensor(ComplexF64,k,j)
```
"""
function ITensor(ElT::Type{<:Number}, is::Indices)
  z = NDTensors.Zeros{ElT,1,NDTensors.default_datatype(ElT)}(is)
  return ITensor(NeverAlias(), ElT, z, is)
end

ITensor(ElT::Type{<:Number}, is...) = ITensor(ElT, indices(is...))

ITensor(is...) = ITensor(NDTensors.default_eltype(), is...)

# To fix ambiguity with QN Index version
# TODO: define as `emptyITensor(ElT)`
ITensor(ElT::Type{<:Number}=NDTensors.default_eltype()) = ITensor(ElT, ())

function ITensor(::Type{ElT}, inds::Tuple{}) where {ElT<:Number}
  z = NDTensors.Zeros{ElT,1,NDTensors.default_datatype(ElT)}(inds)
  return ITensor(AllowAlias(), ElT, z)
end

"""
    ITensor([::Type{ElT} = Float64, ]::UndefInitializer, inds)
    ITensor([::Type{ElT} = Float64, ]::UndefInitializer, inds::Index...)

Construct an ITensor filled with undefined elements having indices `inds` and
element type `ElT`. If the element type is not specified, it defaults to `Float64`.
One purpose for using this constructor is that initializing the elements in an
  undefined way is faster than initializing them to a set value such as zero.

The storage will have `NDTensors.Dense` type.

# Examples

```julia
i = Index(2,"index_i")
j = Index(4,"index_j")
k = Index(3,"index_k")

A = ITensor(undef,i,j)
B = ITensor(ComplexF64,undef,k,j)
```
"""
function ITensor(::Type{ElT}, ::UndefInitializer, inds::Indices) where {ElT<:Number}
  return itensor(Dense(ElT, undef, dim(inds)), indices(inds))
end

function ITensor(::Type{ElT}, ::UndefInitializer, inds...) where {ElT<:Number}
  return ITensor(ElT, undef, indices(inds...))
end

ITensor(::UndefInitializer, inds::Indices) = ITensor(Float64, undef, inds)

ITensor(::UndefInitializer, inds...) = ITensor(Float64, undef, indices(inds...))

"""
    ITensor([ElT::Type, ]x::Number, inds)
    ITensor([ElT::Type, ]x::Number, inds::Index...)

Construct an ITensor with all elements set to `x` and indices `inds`.

  If `x isa Int` or `x isa Complex{Int}` then the elements will be set to `float(x)`
  unless specified otherwise by the first input.

  The storage will have `NDTensors.Dense` type.

  # Examples

  ```julia
  i = Index(2,"index_i"); j = Index(4,"index_j"); k = Index(3,"index_k");

  A = ITensor(1.0, i, j)
  A = ITensor(1, i, j) # same as above
  B = ITensor(2.0+3.0im, j, k)
  ```

  !!! warning
      In future versions this may not automatically convert integer inputs with `float`, and in that case the particular element type should not be relied on.
  """
ITensor(elt::Type{<:Number}, x::Number, is::Indices) = _ITensor(elt, x, is)

# For disambiguation with QN version
ITensor(elt::Type{<:Number}, x::Number, is::Tuple{}) = _ITensor(elt, x, is)

function _ITensor(elt::Type{<:Number}, x::Number, is::Indices)
  return ITensor(Dense(convert(elt, x), dim(is)), is)
end

ITensor(elt::Type{<:Number}, x::Number, is...) = ITensor(elt, x, indices(is...))

ITensor(x::Number, is...) = ITensor(eltype(x), x, is...)

ITensor(x::RealOrComplex{Int}, is...) = ITensor(float(x), is...)

"""
    itensor(args...; kwargs...)

Like the `ITensor` constructor, but with attempt to make a view
of the input data when possible.
"""
itensor(args...; kwargs...)::ITensor = ITensor(AllowAlias(), args...; kwargs...)

ITensor(::AliasStyle, args...; kwargs...)::ITensor =
  error("ITensor constructor with input arguments of types `$(typeof.(args))` not defined.")

"""
    dense(T::ITensor)

Make a new ITensor where the storage is the closest Dense storage,
avoiding allocating new data if possible.
For example, an ITensor with Diag storage will become Dense storage,
filled with zeros except for the diagonal values.
"""
function dense(A::ITensor)
  return setinds(itensor(dense(tensor(A))), removeqns(inds(A)))
end

copy(T::ITensor)::ITensor = itensor(copy(tensor(T)))
zero(T::ITensor)::ITensor = itensor(zero(tensor(T)))

#
# Construct from Array
#

# Helper functions for different view behaviors
# TODO: Move to NDTensors.jl
function (arraytype::Type{<:AbstractArray})(::NeverAlias, A::AbstractArray)
  return set_unspecified_parameters(arraytype, get_parameters(A))(A)
end

function (arraytype::Type{<:AbstractArray})(::AllowAlias, A::AbstractArray)
  return convert(set_unspecified_parameters(arraytype, get_parameters(A)), A)
end

"""
    Array{ElT, N}(T::ITensor, i:Index...)
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
function Array{ElT,N}(T::ITensor, is::Indices) where {ElT,N}
  ndims(T) != N && throw(
    DimensionMismatch(
      "cannot convert an $(ndims(T)) dimensional ITensor to an $N-dimensional Array."
    ),
  )
  TT = tensor(permute(T, is))
  return Array{ElT,N}(TT)::Array{ElT,N}
end

function Array{ElT,N}(T::ITensor, is...) where {ElT,N}
  return Array{ElT,N}(T, indices(is...))
end

function Array{ElT}(T::ITensor, is::Indices) where {ElT}
  return Array{ElT,length(is)}(T, is)
end

function Array{ElT}(T::ITensor, is...) where {ElT}
  return Array{ElT}(T, indices(is...))
end

function Array(T::ITensor, is...)
  return Array{eltype(T)}(T, is...)
end

function Array{<:Any,N}(T::ITensor, is...) where {N}
  return Array{eltype(T),N}(T, is...)
end

function Vector{ElT}(T::ITensor)::Vector{ElT} where {ElT}
  ndims(T) != 1 && throw(
    DimensionMismatch("cannot convert an $(ndims(T)) dimensional ITensor to a Vector.")
  )
  return Array{ElT}(T, inds(T)...)
end

function Vector(T::ITensor)::Vector
  return Array(T, inds(T)...)
end
#########################
# End ITensor constructors
#

#########################
# ITensor properties
#
"""
    storage(T::ITensor)

Return a view of the TensorStorage of the ITensor.
"""
storage(T::ITensor)::TensorStorage = storage(tensor(T))

storagetype(x::ITensor) = storagetype(tensor(x))

"""
    data(T::ITensor)

Return a view of the raw data of the ITensor.

This is mostly an internal ITensor function, please
let the developers of ITensors.jl know if there is
functionality for ITensors that you would like
that is not currently available.
"""
data(T::ITensor) = NDTensors.data(tensor(T))

NDTensors.data(x::ITensor) = data(x)
datatype(x::ITensor) = datatype(tensor(x))

# Trait to check if the tensor has QN symmetry
symmetrystyle(T::Tensor) = symmetrystyle(inds(T))
symmetrystyle(T::ITensor)::SymmetryStyle = symmetrystyle(tensor(T))

eltype(T::ITensor) = eltype(tensor(T))
scalartype(x::ITensor) = eltype(x)

"""
    order(A::ITensor)
    ndims(A::ITensor)

The number of indices, `length(inds(A))`.
"""
order(T::ITensor)::Int = ndims(T)

Order(T::ITensor) = Order(order(T))

ndims(T::ITensor)::Int = ndims(tensor(T))

"""
    dim(A::ITensor)

The total dimension of the space the tensor lives in, `prod(dims(A))`.
"""
dim(T::ITensor)::Int = dim(tensor(T))

"""
    maxdim(A::ITensor)

The maximum dimension of the tensor indices.
"""
maxdim(T::ITensor)::Int = maxdim(tensor(T))

"""
    mindim(A::ITensor)

The minimum dimension of the tensor indices.
"""
mindim(T::ITensor)::Int = mindim(tensor(T))

"""
    dim(A::ITensor, n::Int)

Get the nth dimension of the ITensors.
"""
dim(T::ITensor, n::Int)::Int = dim(tensor(T), n)

"""
    dims(A::ITensor)
    size(A::ITensor)

Tuple containing `dim(inds(A)[d]) for d in 1:ndims(A)`.
"""
dims(T::ITensor) = dims(tensor(T))

axes(T::ITensor) = axes(tensor(T))

size(T::ITensor) = dims(T)

size(A::ITensor, d::Int) = size(tensor(A), d)

_isemptyscalar(A::ITensor) = _isemptyscalar(tensor(A))
_isemptyscalar(A::Tensor) = ndims(A) == 0 && isemptystorage(A)

"""
    dir(A::ITensor, i::Index)

Return the direction of the Index `i` in the ITensor `A`.
"""
dir(A::ITensor, i::Index) = dir(inds(A), i)

dirs(A::ITensor, is) = dirs(inds(A), is)

# TODO: add isdiag(::Tensor) to NDTensors
isdiag(T::ITensor)::Bool = (storage(T) isa Diag || storage(T) isa DiagBlockSparse)

diaglength(T::ITensor) = diaglength(tensor(T))

#
# Block sparse related functions
# (Maybe create fallback definitions for dense tensors)
#

hasqns(T::Union{Tensor,ITensor}) = hasqns(inds(T))

eachnzblock(T::ITensor) = eachnzblock(tensor(T))

nnz(T::ITensor) = nnz(tensor(T))

nblocks(T::ITensor, args...) = nblocks(tensor(T), args...)

nnzblocks(T::ITensor) = nnzblocks(tensor(T))

nzblock(T::ITensor, args...) = nzblock(tensor(T), args...)

nzblocks(T::ITensor) = nzblocks(tensor(T))

blockoffsets(T::ITensor) = blockoffsets(tensor(T))

"""
    isemptystorage(T::ITensor)

Returns `true` if the ITensor contains no elements.

An ITensor with `EmptyStorage` storage always returns `true`.
"""
isemptystorage(T::ITensor) = iszero(tensor(T))
isempty(T::ITensor) = iszero(T)

isreal(T::ITensor) = eltype(T) <: Real
iszero(T::ITensor) = all(iszero, T)
#########################
# End ITensor properties
#

#########################
# ITensor iterators
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

!!! warning
    Unlike standard `AbstractArray{T, N}` types, `ITensor`s do not have their
    order as type paramater, and therefore iterating using `CartesianIndices`
    is generally slow. If you are performing operations that use iterating over
    individual elements of an ITensor it is best to convert to `NDTensors.Tensor`.
"""
CartesianIndices(A::ITensor) = CartesianIndices(tensor(A))

"""
    eachindval(A::ITensor)

Create an iterable object for visiting each element of the ITensor `A` (including structually
zero elements for sparse tensors) in terms of pairs of indices and values.
"""
eachindval(T::ITensor) = eachindval(inds(T))

"""
    iterate(A::ITensor, args...)

Iterate over the elements of an ITensor.
"""
iterate(A::ITensor, args...) = iterate(tensor(A), args...)

#########################
# End ITensor iterators
#

#########################
# ITensor Accessor Functions
#

function settensor!(T::ITensor, t)::ITensor
  T.tensor = t
  return T
end

function setinds!(T::ITensor, is)::ITensor
  # TODO: always convert to Tuple with Tensor type?
  return settensor!(T, setinds(tensor(T), Tuple(is)))
end

function setstorage!(T::ITensor, st)::ITensor
  return settensor!(T, setstorage(tensor(T), st))
end

function setinds(T::ITensor, is)::ITensor
  # TODO: always convert to Tuple with Tensor type?
  return itensor(setinds(tensor(T), Tuple(is)))
end

function setstorage(T::ITensor, st)::ITensor
  return itensor(setstorage(tensor(T), st))
end

removeqns(T::ITensor) = dense(T)

"""
    denseblocks(T::ITensor)

Make a new ITensor where any blocks which have a sparse format, such
as diagonal sparsity, are made dense while still preserving the outer
block-sparse structure. This method avoids allocating new data if possible.

For example, an ITensor with DiagBlockSparse storage will have BlockSparse storage
afterwards.
"""
denseblocks(D::ITensor) = itensor(denseblocks(tensor(D)))

"""
    complex(T::ITensor)

Convert to the complex version of the storage.
"""
complex(T::ITensor) = itensor(complex(tensor(T)))

function complex!(T::ITensor)
  ct = complex(tensor(T))
  setstorage!(T, storage(ct))
  setinds!(T, inds(ct))
  return T
end

function convert_eltype(elt::Type, T::ITensor)
  if eltype(T) == elt
    return T
  end
  return itensor(adapt(elt, tensor(T)))
end

function convert_leaf_eltype(elt::Type, T::ITensor)
  return convert_eltype(elt, T)
end

"""
    convert_leaf_eltype(ElType::Type, A::Array)

Convert the element type of the lowest level containers
("leaves") of a recursive data structure, such as
an Vector of Vectors.
"""
function convert_leaf_eltype(elt::Type, A::Array)
  return map(x -> convert_leaf_eltype(elt, x), A)
end

"""
    scalar(T::ITensor)

Extract the element of an order zero ITensor.

Same as `T[]`.
"""
scalar(T::ITensor)::Any = T[]

lastindex(A::ITensor, n::Int64) = LastVal()
lastindex(A::ITensor) = LastVal()

"""
    fill!(T::ITensor, x::Number)

Fill all values of the ITensor with the specified value.
"""
function fill!(T::ITensor, x::Number)
  # Use broadcasting `T .= x`?
  return settensor!(T, fill!!(tensor(T), x))
end

#
# Block sparse related functions
# (Maybe create fallback definitions for dense tensors)
#

function insertblock!(T::ITensor, args...)
  (!isnothing(flux(T)) && flux(T) ≠ flux(T, args...)) &&
    error("Block does not match current flux")
  TR = insertblock!!(tensor(T), args...)
  settensor!(T, TR)
  return T
end

function insert_diag_blocks!(T::ITensor)
  ## TODO: Add a check that all diag blocks
  ## have the correct flux
  ## (!isnothing(flux(T)) && check_diagblock_flux(T)) &&
  ##   error("Block does not match current flux")
  insert_diag_blocks!(tensor(T))
  return T
end

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
@propagate_inbounds getindex(T::ITensor, I::Integer...)::Any = tensor(T)[I...]

@propagate_inbounds @inline _getindex(T::Tensor, I::Integer...) = T[I...]

# TODO: move to NDTensors (would require moving `LastVal` to NDTensors)
@propagate_inbounds @inline function _getindex(T::Tensor, I::Union{Integer,LastVal}...)
  return T[lastval_to_int(T, I)...]
end

# Special case that handles indexing with `end` like `A[i => end, j => 3]`
@propagate_inbounds getindex(T::ITensor, I::Union{Integer,LastVal}...)::Any =
  _getindex(tensor(T), I...)

# Simple version with just integer indexing, bounds checking gets done by NDTensors

@propagate_inbounds function getindex(T::ITensor, b::Block{N}) where {N}
  # XXX: this should return an ITensor view
  return tensor(T)[b]
end

# Version accepting CartesianIndex, useful when iterating over
# CartesianIndices
@propagate_inbounds getindex(T::ITensor, I::CartesianIndex)::Any = T[Tuple(I)...]

@propagate_inbounds @inline function _getindex(T::Tensor, ivs::Vararg{Any,N}) where {N}
  # Tried ind.(ivs), val.(ivs) but it is slower
  p = NDTensors.getperm(inds(T), ntuple(n -> ind(@inbounds ivs[n]), Val(N)))
  fac = NDTensors.permfactor(p, ivs...) #<fermions> possible sign
  return fac *
         _getindex(T, NDTensors.permute(ntuple(n -> val(@inbounds ivs[n]), Val(N)), p)...)
end

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
@propagate_inbounds (getindex(T::ITensor, ivs::Vararg{Any,N})::Any) where {N} =
  _getindex(tensor(T), ivs...)

@propagate_inbounds function getindex(T::ITensor)::Any
  if order(T) != 0
    throw(
      DimensionMismatch(
        "In scalar(T) or T[], ITensor T is not a scalar (it has indices $(inds(T)))."
      ),
    )
  end
  return tensor(T)[]
end

function _vals(is::Indices, I::String...)
  return val.(is, I)
end

function _vals(T::ITensor, I::String...)
  return _vals(inds(T), I...)
end

# Enable indexing with string values, like `A["Up"]`.
function getindex(T::ITensor, I1::String, Is::String...)
  return T[_vals(T, I1, Is...)...]
end

# Defining this with the type signature `I::Vararg{Integer, N}` instead of `I::Integer...` is much faster:
#
# 58.720 ns (1 allocation: 368 bytes)
#
# instead of:
#
# 465.454 ns (7 allocations: 1.86 KiB)
#
# for some reason! Maybe it helps with inlining?
#
@propagate_inbounds @inline function _setindex!!(
  ::SymmetryStyle, T::Tensor, x::Number, I::Vararg{Integer,N}
) where {N}
  # Generic version, doesn't check the flux
  return setindex!!(T, x, I...)
end

@propagate_inbounds @inline function _setindex!!(
  T::Tensor, x::Number, I::Vararg{Integer,N}
) where {N}
  # Use type trait dispatch to split off between QN version that checks the flux
  # and non-QN version that doesn't

  return _setindex!!(symmetrystyle(T), T, x, I...)
end

@propagate_inbounds @inline function _setindex!!(
  T::Tensor, x::Number, I::Vararg{Union{Integer,LastVal},N}
) where {N}
  return _setindex!!(T, x, lastval_to_int(T, I)...)
end

"""
    setindex!(T::ITensor, x::Number, ivs...)

    setindex!(T::ITensor, x::Number, I::Integer...)

    setindex!(T::ITensor, x::Number, I::CartesianIndex)

Set the specified element of the ITensor, using a list
of `Pair{<:Index, Integer}` (or `IndexVal`).

If just integers are used, set the specified element of the ITensor
using internal Index ordering of the ITensor (only for advanced usage,
only use if you know the axact ordering of the indices).

# Example
```julia
i = Index(2; tags = "i")
A = ITensor(i, i')
A[i => 1, i' => 2] = 1.0 # same as: A[i' => 2, i => 1] = 1.0
A[1, 2] = 1.0 # same as: A[i => 1, i' => 2] = 1.0

# Some simple slicing is also supported
A[i => 2, i' => :] = [2.0 3.0]
A[2, :] = [2.0 3.0]
```
"""
@propagate_inbounds @inline function setindex!(
  T::ITensor, x::Number, I::Vararg{Integer,N}
) where {N}
  # XXX: for some reason this is slow (257.467 ns (6 allocations: 1.14 KiB) for `A[1, 1, 1] = 1`)
  # Calling `setindex!` directly here is faster (56.635 ns (1 allocation: 368 bytes) for `A[1, 1, 1] = 1`)
  # but of course less generic. Can't figure out how to optimize it,
  # even the generic IndexVal version above is faster (126.818 ns (5 allocations: 768 bytes) for `A[i'' => 1, i' => 1, i => 1] = 1`)
  return settensor!(T, _setindex!!(tensor(T), x, I...))
end

@propagate_inbounds function setindex!(T::ITensor, x::Number, I::CartesianIndex)
  return setindex!(T, x, Tuple(I)...)
end

@propagate_inbounds @inline function _setindex!!(
  T::Tensor, x::Number, ivs::Vararg{Any,N}
) where {N}
  # Would be nice to split off the functions for extracting the `ind` and `val` as Tuples,
  # but it was slower.
  p = NDTensors.getperm(inds(T), ntuple(n -> ind(@inbounds ivs[n]), Val(N)))
  fac = NDTensors.permfactor(p, ivs...) #<fermions> possible sign
  return _setindex!!(
    T, fac * x, NDTensors.permute(ntuple(n -> val(@inbounds ivs[n]), Val(N)), p)...
  )
end

@propagate_inbounds @inline function setindex!(
  T::ITensor, x::Number, I::Vararg{Any,N}
) where {N}
  return settensor!(T, _setindex!!(tensor(T), x, I...))
end

@propagate_inbounds @inline function setindex!(
  T::ITensor, x::Number, I1::Pair{<:Index,String}, I::Pair{<:Index,String}...
)
  Iv = map(i -> i.first => val(i.first, i.second), (I1, I...))
  return setindex!(T, x, Iv...)
end

# XXX: what is this definition for?
Base.checkbounds(::Any, ::Block) = nothing

@propagate_inbounds function setindex!(T::ITensor, A::AbstractArray, I...)
  @boundscheck checkbounds(tensor(T), I...)
  TR = setindex!!(tensor(T), A, I...)
  setstorage!(T, storage(TR))
  return T
end

#function setindex!(T::ITensor, A::AbstractArray, b::Block)
#  # XXX: use setindex!! syntax
#  tensor(T)[b] = A
#  return T
#end

function setindex!(T::ITensor, A::AbstractArray, ivs::Pair{<:Index}...)
  input_inds = first.(ivs)
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

# Enable indexing with string values, like `A["Up"]`.
function setindex!(T::ITensor, x::Number, I1::String, Is::String...)
  T[_vals(T, I1, Is...)...] = x
  return T
end

#function setindex!(::ITensor{Any}, ::Number, ivs...)
#  error("Cannot set the element of an emptyITensor(). Must define indices to set elements")
#end

#########################
# End ITensor Accessor Functions
#

#########################
# ITensor Operations
#

similar(T::ITensor, args...)::ITensor = itensor(NDTensors.similar(tensor(T), args...))

function isapprox(A::ITensor, B::ITensor; kwargs...)
  if !hassameinds(A, B)
    error("In `isapprox(::ITensor, ::ITensor)`, the indices of the ITensors do not
          match. The first ITensor has indices: \n\n$(inds(A))\n\nbut the second
          ITensor has indices: \n\n$(inds(B))")
  end
  B = permute(B, inds(A))
  return isapprox(array(A), array(B); kwargs...)
end

function randn!(T::ITensor)
  return randn!(Random.default_rng(), T)
end

function randn!(rng::AbstractRNG, T::ITensor)
  return settensor!(T, randn!!(rng, tensor(T)))
end

norm(T::ITensor) = norm(tensor(T))

function dag(as::AliasStyle, T::Tensor{ElT,N}) where {ElT,N}
  if using_auto_fermion() && has_fermionic_subspaces(inds(T)) # <fermions>
    CT = conj(NeverAlias(), T)
    NDTensors.scale_blocks!(CT, block -> NDTensors.permfactor(reverse(1:N), block, inds(T)))
    return setinds(CT, dag(inds(T)))
  end
  return setinds(conj(as, T), dag(inds(T)))
end

function dag(as::AliasStyle, T::ITensor)
  return itensor(dag(as, tensor(T)))
end

# Helpful for generic code
dag(x::Number) = conj(x)

"""
    dag(T::ITensor; allow_alias = true)

Complex conjugate the elements of the ITensor `T` and dagger the indices.

By default, an alias of the ITensor is returned (i.e. the output ITensor
may share data with the input ITensor). If `allow_alias = false`,
an alias is never returned.
"""
function dag(T::ITensor; kwargs...)
  allow_alias::Bool = deprecated_keyword_argument(
    Bool,
    kwargs;
    new_kw=:allow_alias,
    old_kw=:always_copy,
    default=true,
    funcsym=:dag,
    map=!,
  )
  aliasstyle::Union{AllowAlias,NeverAlias} = allow_alias ? AllowAlias() : NeverAlias()
  return dag(aliasstyle, T)
end

function (T::ITensor * x::Number)::ITensor
  return itensor(x * tensor(T))
end

# TODO: what about noncommutative number types?
(x::Number * T::ITensor) = T * x

(A::ITensor / x::Number) = itensor(tensor(A) / x)

(T1::ITensor / T2::ITensor) = T1 / T2[]

-(A::ITensor) = itensor(-tensor(A))

function _add(A::Tensor, B::Tensor)
  if _isemptyscalar(A) && ndims(B) > 0
    return itensor(B)
  elseif _isemptyscalar(B) && ndims(A) > 0
    return itensor(A)
  end
  ndims(A) != ndims(B) &&
    throw(DimensionMismatch("cannot add ITensors with different numbers of indices"))
  itA = itensor(A)
  itB = itensor(B)
  itC = copy(itA)
  itC .+= itB
  return itC
end

# TODO: move the order-0 EmptyStorage ITensor special case to NDTensors.
# Unfortunately this is more complicated than it might seem since it
# has to pass through the broadcasting mechanism first.
function (A::ITensor + B::ITensor)
  return _add(tensor(A), tensor(B))
end

# TODO: move the order-0 EmptyStorage ITensor special to NDTensors
function (A::ITensor - B::ITensor)
  if _isemptyscalar(A) && ndims(B) > 0
    return -B
  elseif _isemptyscalar(B) && ndims(A) > 0
    return A
  end
  ndims(A) != ndims(B) &&
    throw(DimensionMismatch("cannot subtract ITensors with different numbers of indices"))
  C = copy(A)
  C .-= B
  return C
end

real(T::ITensor)::ITensor = itensor(real(tensor(T)))

imag(T::ITensor)::ITensor = itensor(imag(tensor(T)))

conj(T::ITensor)::ITensor = itensor(conj(tensor(T)))

dag(::Nothing) = nothing

function (A::ITensor == B::ITensor)
  !hassameinds(A, B) && return false
  return norm(A - B) == zero(promote_type(eltype(A), eltype(B)))
end

LinearAlgebra.promote_leaf_eltypes(A::ITensor) = eltype(A)

diag(T::ITensor) = diag(tensor(T))

mul!(C::ITensor, A::ITensor, B::ITensor, args...)::ITensor = contract!(C, A, B, args...)

dot(A::ITensor, B::ITensor) = (dag(A) * B)[]

inner(y::ITensor, A::ITensor, x::ITensor) = (dag(y) * A * x)[]
inner(y::ITensor, x::ITensor) = (dag(y) * x)[]

#
# In-place operations
#

"""
    normalize!(T::ITensor)

Normalize an ITensor in-place, such that norm(T)==1.
"""
normalize!(T::ITensor) = (T .*= 1 / norm(T))

"""
    copyto!(B::ITensor, A::ITensor)

Copy the contents of ITensor A into ITensor B.
```
B .= A
```
"""
function copyto!(R::ITensor, T::ITensor)
  R .= T
  return R
end

# Note this already assumes R === T1, which will be lifted
# in the future.
function _map!!(f::Function, R::Tensor, T1::Tensor, T2::Tensor)
  perm = NDTensors.getperm(inds(R), inds(T2))
  if !isperm(perm)
    error("""
          You are trying to add an ITensor with indices:

          $(inds(T2))

          into an ITensor with indices:

          $(inds(R))

          but the indices are not permutations of each other.
          """)
  end
  if hasqns(T2) && hasqns(R)
    # Check that Index arrows match
    for (n, p) in enumerate(perm)
      if dir(inds(R)[n]) != dir(inds(T2)[p])
        #println("Mismatched Index: \n$(inds(R)[n])")
        error("Index arrows must be the same to add, subtract, map, or scale QN ITensors")
      end
    end
  end
  return permutedims!!(R, T2, perm, f)
end

function map!(f::Function, R::ITensor, T1::ITensor, T2::ITensor)
  R !== T1 && error("`map!(f, R, T1, T2)` only supports `R === T1` right now")
  return settensor!(R, _map!!(f, tensor(R), tensor(T1), tensor(T2)))
end

map(f, x::ITensor) = itensor(map(f, tensor(x)))

"""
    axpy!(a::Number, v::ITensor, w::ITensor)
```
w .+= a .* v
```
"""
axpy!(a::Number, v::ITensor, w::ITensor) = (w .+= a .* v)

"""
axpby!(a,v,b,w)

```
w .= a .* v + b .* w
```
"""
axpby!(a::Number, v::ITensor, b::Number, w::ITensor) = (w .= a .* v + b .* w)

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

#########################
# End ITensor Operations
#

# Helper function for deprecating a keyword argument
function deprecated_keyword_argument(
  ::Type{T}, kwargs; new_kw, old_kw, default, funcsym, map=identity
)::T where {T}
  has_new_kw = haskey(kwargs, new_kw)
  has_old_kw = haskey(kwargs, old_kw)
  res::T = if has_old_kw
    Base.depwarn(
      "In `$func`, keyword argument `$old_kw` is deprecated in favor of `$new_kw`.", func
    )
    if has_new_kw
      println(
        "Warning: keyword arguments `$old_kw` and `$new_kw` are both specified, using `$new_kw`.",
      )
      kwargs[new_kw]
    else
      map(kwargs[old_kw])
    end
  else
    get(kwargs, new_kw, default)
  end
  return res
end

#######################################################################
#
# Printing, reading and writing ITensors
#

function summary(io::IO, T::ITensor)
  print(io, "ITensor ord=$(order(T))")
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
  return print(io, typeof(storage(T)))
end

# TODO: make a specialized printing from Diag
# that emphasizes the missing elements
function show(io::IO, T::ITensor)
  println(io, "ITensor ord=$(order(T))")
  return show(io, MIME"text/plain"(), tensor(T))
end

function show(io::IO, mime::MIME"text/plain", T::ITensor)
  return summary(io, T)
end

function readcpp(io::IO, ::Type{Dense{ValT}}; kwargs...) where {ValT}
  format = get(kwargs, :format, "v3")
  if format == "v3"
    size = read(io, UInt64)
    data = Vector{ValT}(undef, size)
    for n in 1:size
      data[n] = read(io, ValT)
    end
    return Dense(data)
  else
    throw(ArgumentError("read Dense: format=$format not supported"))
  end
end

function readcpp(io::IO, ::Type{ITensor}; kwargs...)
  format = get(kwargs, :format, "v3")
  if format == "v3"
    # TODO: use Vector{Index} here?
    inds = readcpp(io, IndexSet; kwargs...)
    read(io, 12) # ignore scale factor by reading 12 bytes
    storage_type = read(io, Int32)
    if storage_type == 0 # Null
      storage = Dense{Nothing}()
    elseif storage_type == 1  # DenseReal
      storage = readcpp(io, Dense{Float64}; kwargs...)
    elseif storage_type == 2  # DenseCplx
      storage = readcpp(io, Dense{ComplexF64}; kwargs...)
    elseif storage_type == 3  # Combiner
      storage = CombinerStorage(T.inds[1])
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
    return itensor(storage, inds)
  else
    throw(ArgumentError("read ITensor: format=$format not supported"))
  end
end

function HDF5.write(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, T::ITensor)
  g = create_group(parent, name)
  attributes(g)["type"] = "ITensor"
  attributes(g)["version"] = 1
  write(g, "inds", inds(T))
  return write(g, "storage", storage(T))
end

function HDF5.read(
  parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{ITensor}
)
  g = open_group(parent, name)
  if read(attributes(g)["type"]) != "ITensor"
    error("HDF5 group or file does not contain ITensor data")
  end
  # TODO: use Vector{Index} here?
  inds = read(g, "inds", IndexSet)

  # check input file for key name of ITensor data
  # ITensors.jl <= v0.1.x uses `store` as key
  # whereas ITensors.jl >= v0.2.x uses `storage` as key
  for key in ["storage", "store"]
    if haskey(g, key)
      stypestr = read(attributes(open_group(g, key))["type"])
      stype = eval(Meta.parse(stypestr))
      storage = read(g, key, stype)
      return itensor(storage, inds)
    end
  end
  return error("HDF5 file: $(g) does not contain correct ITensor data.\nNeither key
               `store` nor `storage` could be found.")
end
