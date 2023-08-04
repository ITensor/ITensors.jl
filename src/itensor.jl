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

## The categories in this file are
## constructors, properties, iterators,
## Accessor Functions, Index Functions and Operations
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

#########################
# ITensor constructors
#

# Version where the indices are not Tuple, so convert to Tuple
function ITensor(::AllowAlias, T::Tensor)::ITensor
  return ITensor(AllowAlias(), setinds(T, NTuple{ndims(T)}(inds(T))))
end

ITensor(::NeverAlias, T::Tensor)::ITensor = ITensor(AllowAlias(), copy(T))

ITensor(T::Tensor)::ITensor = ITensor(NeverAlias(), T)

"""
    ITensor(st::TensorStorage, is)

Constructor for an ITensor from a TensorStorage
and a set of indices.
The ITensor stores a view of the TensorStorage.
"""
ITensor(as::AliasStyle, st::TensorStorage, is)::ITensor =
  ITensor(as, Tensor(as, st, Tuple(is)))
ITensor(as::AliasStyle, is, st::TensorStorage)::ITensor = ITensor(as, st, is)

ITensor(st::TensorStorage, is)::ITensor = itensor(Tensor(NeverAlias(), st, Tuple(is)))
ITensor(is, st::TensorStorage)::ITensor = ITensor(NeverAlias(), st, is)

itensor(T::ITensor) = T
ITensor(T::ITensor) = copy(T)

"""
    itensor(args...; kwargs...)

Like the `ITensor` constructor, but with attempt to make a view
of the input data when possible.
"""
itensor(args...; kwargs...)::ITensor = ITensor(AllowAlias(), args...; kwargs...)

ITensor(::AliasStyle, args...; kwargs...)::ITensor =
  error("ITensor constructor with input arguments of types `$(typeof.(args))` not defined.")

"""
    Tensor(::ITensor)

Create a `Tensor` that stores a copy of the storage and
indices of the input `ITensor`.
"""
Tensor(T::ITensor)::Tensor = Tensor(NeverAlias(), T)
Tensor(as::NeverAlias, T::ITensor)::Tensor = Tensor(AllowAlias(), copy(T))

"""
    tensor(::ITensor)

Convert the `ITensor` to a `Tensor` that shares the same
storage and indices as the `ITensor`.
"""
Tensor(::AllowAlias, A::ITensor) = A.tensor

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
function ITensor(eltype::Type{<:Number}, is::Indices)
  return itensor(EmptyStorage(eltype), is)
end

ITensor(eltype::Type{<:Number}, is...) = ITensor(eltype, indices(is...))

ITensor(is...) = ITensor(EmptyNumber, is...)

# To fix ambiguity with QN Index version
# TODO: define as `emptyITensor(ElT)`
ITensor(eltype::Type{<:Number}=EmptyNumber) = ITensor(eltype, ())

# TODO: define as `emptyITensor(ElT)`
function ITensor(::Type{ElT}, inds::Tuple{}) where {ElT<:Number}
  return ITensor(EmptyStorage(ElT), inds)
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
ITensor(eltype::Type{<:Number}, x::Number, is::Indices) = _ITensor(eltype, x, is)

# For disambiguation with QN version
ITensor(eltype::Type{<:Number}, x::Number, is::Tuple{}) = _ITensor(eltype, x, is)

function _ITensor(eltype::Type{<:Number}, x::Number, is::Indices)
  return ITensor(Dense(convert(eltype, x), dim(is)), is)
end

ITensor(eltype::Type{<:Number}, x::Number, is...) = ITensor(eltype, x, indices(is...))

ITensor(x::Number, is...) = ITensor(eltype(x), x, is...)

const RealOrComplex{T} = Union{T,Complex{T}}

ITensor(x::RealOrComplex{Int}, is...) = ITensor(float(x), is...)

#
# EmptyStorage ITensor constructors
#

# TODO: Deprecated!
"""
    emptyITensor([::Type{ElT} = NDTensors.EmptyNumber, ]inds)
    emptyITensor([::Type{ElT} = NDTensors.EmptyNumber, ]inds::Index...)

Construct an ITensor with storage type `NDTensors.EmptyStorage`, indices `inds`, and element type `ElT`. If the element type is not specified, it defaults to `NDTensors.EmptyNumber`, which represents a number type that can take on any value (for example, the type of the first value it is set to).
"""
function emptyITensor(::Type{ElT}, is::Indices) where {ElT<:Number}
  return itensor(EmptyTensor(ElT, is))
end

function emptyITensor(::Type{ElT}, is...) where {ElT<:Number}
  return emptyITensor(ElT, indices(is...))
end

emptyITensor(is::Indices) = emptyITensor(EmptyNumber, is)

emptyITensor(is...) = emptyITensor(EmptyNumber, indices(is...))

function emptyITensor(::Type{ElT}=EmptyNumber) where {ElT<:Number}
  return itensor(EmptyTensor(ElT, ()))
end

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
  eltype::Type{<:Number},
  A::AbstractArray{<:Number},
  inds::Indices;
  kwargs...,
)
  length(A) ≠ dim(inds) && throw(
    DimensionMismatch(
      "In ITensor(::AbstractArray, inds), length of AbstractArray ($(length(A))) must match total dimension of IndexSet ($(dim(inds)))",
    ),
  )
  data = set_eltype(typeof(A), eltype)(as, A)
  return itensor(Dense(data), inds)
end

function ITensor(
  as::AliasStyle, eltype::Type{<:Number}, A::AbstractArray{<:Number}, inds; kwargs...
)
  is = indices(inds)
  if !isa(is, Indices)
    error("Indices $inds are not valid for constructing an ITensor.")
  end
  return ITensor(as, eltype, A, is; kwargs...)
end

# Convert `Adjoint` to `Matrix`
function ITensor(
  as::AliasStyle, eltype::Type{<:Number}, A::Adjoint, inds::Indices{Index{Int}}; kwargs...
)
  return ITensor(as, eltype, Matrix(A), inds; kwargs...)
end

function ITensor(
  as::AliasStyle, eltype::Type{<:Number}, A::AbstractArray{<:Number}, is...; kwargs...
)
  return ITensor(as, eltype, A, indices(is...); kwargs...)
end

function ITensor(eltype::Type{<:Number}, A::AbstractArray{<:Number}, is...; kwargs...)
  return ITensor(NeverAlias(), eltype, A, is...; kwargs...)
end

# For now, it's not well defined to construct an ITensor without indices
# from a non-zero dimensional Array
function ITensor(
  as::AliasStyle, eltype::Type{<:Number}, A::AbstractArray{<:Number}; kwargs...
)
  if length(A) > 1
    error(
      "Trying to create an ITensor without any indices from Array $A of dimensions $(size(A)). Cannot construct an ITensor from an Array with more than one element without any indices.",
    )
  end
  return ITensor(eltype, A[]; kwargs...)
end

function ITensor(eltype::Type{<:Number}, A::AbstractArray{<:Number}; kwargs...)
  return ITensor(NeverAlias(), eltype, A; kwargs...)
end
function ITensor(A::AbstractArray{<:Number}; kwargs...)
  return ITensor(NeverAlias(), eltype(A), A; kwargs...)
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

function ITensor(A::AbstractArray{<:Number}, is...; kwargs...)
  return ITensor(NeverAlias(), A, is...; kwargs...)
end

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
function diagITensor(::Type{ElT}, is::Indices) where {ElT<:Number}
  return itensor(Diag(ElT, mindim(is)), is)
end

diagITensor(::Type{ElT}, is...) where {ElT<:Number} = diagITensor(ElT, indices(is...))

diagITensor(is::Indices) = diagITensor(Float64, is)
diagITensor(is...) = diagITensor(indices(is...))

"""
    diagITensor([ElT::Type, ]v::Vector, inds...)
    diagitensor([ElT::Type, ]v::Vector, inds...)

Make a sparse ITensor with non-zero elements only along the diagonal.
In general, the diagonal elements will be those stored in `v` and
the ITensor will have element type `eltype(v)`, unless specified explicitly
by `ElT`. The storage will have `NDTensors.Diag` type.

In the case when `eltype(v) isa Union{Int, Complex{Int}}`, by default it will
be converted to `float(v)`. Note that this behavior is subject to change
in the future.

The version `diagITensor` will never output an ITensor whose storage data
is an alias of the input vector data.

The version `diagitensor` might output an ITensor whose storage data
is an alias of the input vector data in order to minimize operations.
"""
function diagITensor(
  as::AliasStyle, eltype::Type{<:Number}, v::Vector{<:Number}, is::Indices
)
  length(v) ≠ mindim(is) && error(
    "Length of vector for diagonal must equal minimum of the dimension of the input indices",
  )
  data = Vector{eltype}(as, v)
  return itensor(Diag(data), is)
end

function diagITensor(as::AliasStyle, eltype::Type{<:Number}, v::Vector{<:Number}, is...)
  return diagITensor(as, eltype, v, indices(is...))
end

function diagITensor(as::AliasStyle, v::Vector, is...)
  return diagITensor(as, eltype(v), v, is...)
end

function diagITensor(as::AliasStyle, v::Vector{<:RealOrComplex{Int}}, is...)
  return diagITensor(AllowAlias(), float(eltype(v)), v, is...)
end

diagITensor(v::Vector{<:Number}, is...) = diagITensor(NeverAlias(), v, is...)
function diagITensor(eltype::Type{<:Number}, v::Vector{<:Number}, is...)
  return diagITensor(NeverAlias(), eltype, v, is...)
end

diagitensor(args...; kwargs...) = diagITensor(AllowAlias(), args...; kwargs...)

# XXX TODO: explain conversion from Int
# XXX TODO: proper conversion
"""
    diagITensor([ElT::Type, ]x::Number, inds...)
    diagitensor([ElT::Type, ]x::Number, inds...)

Make a sparse ITensor with non-zero elements only along the diagonal.
In general, the diagonal elements will be set to the value `x` and
the ITensor will have element type `eltype(x)`, unless specified explicitly
by `ElT`. The storage will have `NDTensors.Diag` type.

In the case when `x isa Union{Int, Complex{Int}}`, by default it will
be converted to `float(x)`. Note that this behavior is subject to change
in the future.
"""
function diagITensor(as::AliasStyle, eltype::Type{<:Number}, x::Number, is::Indices)
  return diagITensor(AllowAlias(), eltype, fill(eltype(x), mindim(is)), is...)
end

function diagITensor(as::AliasStyle, eltype::Type{<:Number}, x::Number, is...)
  return diagITensor(as, eltype, x, indices(is...))
end

function diagITensor(as::AliasStyle, x::Number, is...)
  return diagITensor(as, typeof(x), x, is...)
end

function diagITensor(as::AliasStyle, x::RealOrComplex{Int}, is...)
  return diagITensor(as, float(typeof(x)), x, is...)
end

function diagITensor(eltype::Type{<:Number}, x::Number, is...)
  return diagITensor(NeverAlias(), eltype, x, is...)
end

diagITensor(x::Number, is...) = diagITensor(NeverAlias(), x, is...)

"""
    delta([::Type{ElT} = Float64, ]inds)
    delta([::Type{ElT} = Float64, ]inds::Index...)

Make a uniform diagonal ITensor with all diagonal elements
`one(ElT)`. Only a single diagonal element is stored.

This function has an alias `δ`.
"""
function delta(eltype::Type{<:Number}, is::Indices)
  return itensor(Diag(one(eltype)), is)
end

function delta(eltype::Type{<:Number}, is...)
  return delta(eltype, indices(is...))
end

delta(is...) = delta(Float64, is...)

const δ = delta

"""
    onehot(ivs...)
    setelt(ivs...)
    onehot(::Type, ivs...)
    setelt(::Type, ivs...)

Create an ITensor with all zeros except the specified value,
which is set to 1.

# Examples
```julia
i = Index(2,"i")
A = onehot(i=>2)
# A[i=>2] == 1, all other elements zero

# Specify the element type
A = onehot(Float32, i=>2)

j = Index(3,"j")
B = onehot(i=>1,j=>3)
# B[i=>1,j=>3] == 1, all other element zero
```
"""
function onehot(datatype::Type{<:AbstractArray}, ivs::Pair{<:Index}...)
  A = ITensor(eltype(datatype), ind.(ivs)...)
  A[val.(ivs)...] = one(eltype(datatype))
  return Adapt.adapt(datatype, A)
end

function onehot(eltype::Type{<:Number}, ivs::Pair{<:Index}...)
  return onehot(NDTensors.default_datatype(eltype), ivs...)
end
function onehot(eltype::Type{<:Number}, ivs::Vector{<:Pair{<:Index}})
  return onehot(NDTensors.default_datatype(eltype), ivs...)
end
function setelt(eltype::Type{<:Number}, ivs::Pair{<:Index}...)
  return onehot(NDTensors.default_datatype(eltype), ivs...)
end

function onehot(ivs::Pair{<:Index}...)
  return onehot(NDTensors.default_datatype(NDTensors.default_eltype()), ivs...)
end
onehot(ivs::Vector{<:Pair{<:Index}}) = onehot(ivs...)
setelt(ivs::Pair{<:Index}...) = onehot(ivs...)

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

"""
    randomITensor([::Type{ElT <: Number} = Float64, ]inds)
    randomITensor([::Type{ElT <: Number} = Float64, ]inds::Index...)

Construct an ITensor with type `ElT` and indices `inds`, whose elements are
normally distributed random numbers. If the element type is not specified,
it defaults to `Float64`.

# Examples

```julia
i = Index(2,"index_i")
j = Index(4,"index_j")
k = Index(3,"index_k")

A = randomITensor(i,j)
B = randomITensor(ComplexF64,undef,k,j)
```
"""
function randomITensor(::Type{S}, is::Indices) where {S<:Number}
  return randomITensor(Random.default_rng(), S, is)
end

function randomITensor(rng::AbstractRNG, ::Type{S}, is::Indices) where {S<:Number}
  T = ITensor(S, undef, is)
  randn!(rng, T)
  return T
end

function randomITensor(::Type{S}, is...) where {S<:Number}
  return randomITensor(Random.default_rng(), S, is...)
end

function randomITensor(rng::AbstractRNG, ::Type{S}, is...) where {S<:Number}
  return randomITensor(rng, S, indices(is...))
end

# To fix ambiguity with QN version
function randomITensor(::Type{ElT}, is::Tuple{}) where {ElT<:Number}
  return randomITensor(Random.default_rng(), ElT, is)
end

# To fix ambiguity with QN version
function randomITensor(rng::AbstractRNG, ::Type{ElT}, is::Tuple{}) where {ElT<:Number}
  return randomITensor(rng, ElT, Index{Int}[])
end

# To fix ambiguity with QN version
function randomITensor(is::Tuple{})
  return randomITensor(Random.default_rng(), is)
end

# To fix ambiguity with QN version
function randomITensor(rng::AbstractRNG, is::Tuple{})
  return randomITensor(rng, Float64, is)
end

# To fix ambiguity errors with QN version
function randomITensor(::Type{ElT}) where {ElT<:Number}
  return randomITensor(Random.default_rng(), ElT)
end

# To fix ambiguity errors with QN version
function randomITensor(rng::AbstractRNG, ::Type{ElT}) where {ElT<:Number}
  return randomITensor(rng, ElT, ())
end

randomITensor(is::Indices) = randomITensor(Random.default_rng(), is)
randomITensor(rng::AbstractRNG, is::Indices) = randomITensor(rng, Float64, is)
randomITensor(is...) = randomITensor(Random.default_rng(), is...)
randomITensor(rng::AbstractRNG, is...) = randomITensor(rng, Float64, indices(is...))

# To fix ambiguity errors with QN version
randomITensor() = randomITensor(Random.default_rng())

# To fix ambiguity errors with QN version
randomITensor(rng::AbstractRNG) = randomITensor(rng, Float64, ())

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
_isemptyscalar(A::Tensor) = ndims(A) == 0 && isemptystorage(A) && eltype(A) === EmptyNumber

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

# XXX: rename isemptystorage?
"""
    isemptystorage(T::ITensor)

Returns `true` if the ITensor contains no elements.

An ITensor with `EmptyStorage` storage always returns `true`.
"""
isemptystorage(T::ITensor) = isemptystorage(tensor(T))
isemptystorage(T::Tensor) = isempty(T)
isempty(T::ITensor) = isemptystorage(T)

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

function convert_eltype(ElType::Type, T::ITensor)
  if eltype(T) == ElType
    return T
  end
  return itensor(adapt(ElType, tensor(T)))
end

function convert_leaf_eltype(ElType::Type, T::ITensor)
  return convert_eltype(ElType, T)
end

"""
    convert_leaf_eltype(ElType::Type, A::Array)

Convert the element type of the lowest level containers
("leaves") of a recursive data structure, such as
an Vector of Vectors.
"""
function convert_leaf_eltype(ElType::Type, A::Array)
  return map(x -> convert_leaf_eltype(ElType, x), A)
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
# ITensor Index Functions
#

"""
    inds(T::ITensor)

Return the indices of the ITensor as a Tuple.
"""
inds(T::ITensor) = inds(tensor(T))

"""
    ind(T::ITensor, i::Int)

Get the Index of the ITensor along dimension i.
"""
ind(T::ITensor, i::Int) = ind(tensor(T), i)

"""
    eachindex(A::ITensor)

Create an iterable object for visiting each element of the ITensor `A` (including structually
zero elements for sparse tensors).

For example, for dense tensors this may return `1:length(A)`, while for sparse tensors
it may return a Cartesian range.
"""
eachindex(A::ITensor) = eachindex(tensor(A))

# TODO: name this `inds` or `indscollection`?
itensor2inds(A::ITensor)::Any = inds(A)
itensor2inds(A::Tensor) = inds(A)
itensor2inds(i::Index) = (i,)
itensor2inds(A) = A
function map_itensor2inds(A::Tuple{Vararg{Any,N}}) where {N}
  return ntuple(i -> itensor2inds(A[i]), Val(N))
end

# in
hasind(A, i::Index) = i ∈ itensor2inds(A)

# issubset
hasinds(A, is) = is ⊆ itensor2inds(A)
hasinds(A, is::Index...) = hasinds(A, is)

"""
    hasinds(is...)

Returns an anonymous function `x -> hasinds(x, is...)` which
accepts an ITensor or IndexSet and returns `true` if the
ITensor or IndexSet has the indices `is`.
"""
hasinds(is::Indices) = x -> hasinds(x, is)
hasinds(is::Index...) = hasinds(is)

"""
    hascommoninds(A, B; kwargs...)

    hascommoninds(B; kwargs...) -> f::Function

Check if the ITensors or sets of indices `A` and `B` have
common indices.

If only one ITensor or set of indices `B` is passed, return a
function `f` such that `f(A) = hascommoninds(A, B; kwargs...)`
"""
hascommoninds(A, B; kwargs...) = !isnothing(commonind(A, B; kwargs...))

hascommoninds(B; kwargs...) = x -> hascommoninds(x, B; kwargs...)

# issetequal
hassameinds(A, B) = issetequal(itensor2inds(A), itensor2inds(B))

# Apply the Index set function and then filter the results
function filter_inds_set_function(
  ffilter::Function, fset::Function, A::Vararg{Any,N}
) where {N}
  return filter(ffilter, fset(map_itensor2inds(A)...))
end

function filter_inds_set_function(fset::Function, A...; kwargs...)
  return filter_inds_set_function(fmatch(; kwargs...), fset, A...)
end

for (finds, fset) in (
  (:commoninds, :_intersect),
  (:noncommoninds, :_symdiff),
  (:uniqueinds, :_setdiff),
  (:unioninds, :_union),
)
  @eval begin
    $finds(args...; kwargs...) = filter_inds_set_function($fset, args...; kwargs...)
  end
end

for find in (:commonind, :noncommonind, :uniqueind, :unionind)
  @eval begin
    $find(args...; kwargs...) = getfirst($(Symbol(find, :s))(args...; kwargs...))
  end
end

function index_filter_kwargs_docstring()
  return """
  Optional keyword arguments:
  * tags::String - a tag name or comma separated list of tag names that the returned indices must all have
  * plev::Int - common prime level that the returned indices must all have
  * inds - Index or collection of indices. Returned indices must come from this set of indices.
  """
end

# intersect
@doc """
    commoninds(A, B; kwargs...)

Return a Vector with indices that are common between the indices of `A` and `B`
(the set intersection, similar to `Base.intersect`).

$(index_filter_kwargs_docstring())
""" commoninds

# firstintersect
@doc """
    commonind(A, B; kwargs...)

Return the first `Index` common between the indices of `A` and `B`.

See also [`commoninds`](@ref).

$(index_filter_kwargs_docstring())
""" commonind

# symdiff
@doc """
    noncommoninds(A, B; kwargs...)

Return a Vector with indices that are not common between the indices of `A` and
`B` (the symmetric set difference, similar to `Base.symdiff`).

$(index_filter_kwargs_docstring())
""" noncommoninds

# firstsymdiff
@doc """
    noncommonind(A, B; kwargs...)

Return the first `Index` not common between the indices of `A` and `B`.

See also [`noncommoninds`](@ref).

$(index_filter_kwargs_docstring())
""" noncommonind

# setdiff
@doc """
    uniqueinds(A, B; kwargs...)

Return Vector with indices that are unique to the set of indices of `A` and not
in `B` (the set difference, similar to `Base.setdiff`).

$(index_filter_kwargs_docstring())
""" uniqueinds

# firstsetdiff
@doc """
    uniqueind(A, B; kwargs...)

Return the first `Index` unique to the set of indices of `A` and not in `B`.

See also [`uniqueinds`](@ref).

$(index_filter_kwargs_docstring())
""" uniqueind

# union
@doc """
    unioninds(A, B; kwargs...)

Return a Vector with indices that are the union of the indices of `A` and `B`
(the set union, similar to `Base.union`).

$(index_filter_kwargs_docstring())
""" unioninds

# firstunion
@doc """
    unionind(A, B; kwargs...)

Return the first `Index` in the union of the indices of `A` and `B`.

See also [`unioninds`](@ref).

$(index_filter_kwargs_docstring())
""" unionind

firstind(A...; kwargs...) = getfirst(map_itensor2inds(A)...; kwargs...)

filterinds(f::Function, A...) = filter(f, map_itensor2inds(A)...)
filterinds(A...; kwargs...) = filter(map_itensor2inds(A)...; kwargs...)

# Faster version when no filtering is requested
filterinds(A::ITensor) = inds(A)
filterinds(is::Indices) = is

# For backwards compatibility
inds(A...; kwargs...) = filterinds(A...; kwargs...)

# in-place versions of priming and tagging
for fname in (
  :prime,
  :setprime,
  :noprime,
  :replaceprime,
  :swapprime,
  :addtags,
  :removetags,
  :replacetags,
  :settags,
  :swaptags,
  :replaceind,
  :replaceinds,
  :swapind,
  :swapinds,
)
  @eval begin
    $fname(f::Function, A::ITensor, args...) = ITensor($fname(f, tensor(A), args...))

    # Inlining makes the ITensor functions slower
    @noinline function $fname(f::Function, A::Tensor, args...)
      return setinds(A, $fname(f, inds(A), args...))
    end

    function $(Symbol(fname, :!))(f::Function, A::ITensor, args...)
      return settensor!(A, $fname(f, tensor(A), args...))
    end

    $fname(A::ITensor, args...; kwargs...) = itensor($fname(tensor(A), args...; kwargs...))

    # Inlining makes the ITensor functions slower
    @noinline function $fname(A::Tensor, args...; kwargs...)
      return setinds(A, $fname(inds(A), args...; kwargs...))
    end

    function $(Symbol(fname, :!))(A::ITensor, args...; kwargs...)
      return settensor!(A, $fname(tensor(A), args...; kwargs...))
    end
  end
end

priming_tagging_doc = """
Optionally, only modify the indices with the specified keyword arguments.

# Arguments
- `tags = nothing`: if specified, only modify Index `i` if `hastags(i, tags) == true`.
- `plev = nothing`: if specified, only modify Index `i` if `hasplev(i, plev) == true`.

The ITensor functions come in two versions, `f` and `f!`. The latter modifies
the ITensor in-place. In both versions, the ITensor storage is not modified or
copied (so it returns an ITensor with a view of the original storage).
"""

@doc """
    prime[!](A::ITensor, plinc::Int = 1; <keyword arguments>) -> ITensor

    prime(inds, plinc::Int = 1; <keyword arguments>) -> IndexSet

Increase the prime level of the indices of an ITensor or collection of indices.

$priming_tagging_doc
""" prime(::ITensor, ::Any...)

@doc """
    setprime[!](A::ITensor, plev::Int; <keyword arguments>) -> ITensor

    setprime(inds, plev::Int; <keyword arguments>) -> IndexSet

Set the prime level of the indices of an ITensor or collection of indices.

$priming_tagging_doc
""" setprime(::ITensor, ::Any...)

@doc """
    noprime[!](A::ITensor; <keyword arguments>) -> ITensor

    noprime(inds; <keyword arguments>) -> IndexSet

Set the prime level of the indices of an ITensor or collection of indices to zero.

$priming_tagging_doc
""" noprime(::ITensor, ::Any...)

@doc """
    replaceprime[!](A::ITensor, plold::Int, plnew::Int; <keyword arguments>) -> ITensor
    replaceprime[!](A::ITensor, plold => plnew; <keyword arguments>) -> ITensor
    mapprime[!](A::ITensor, <arguments>; <keyword arguments>) -> ITensor

    replaceprime(inds, plold::Int, plnew::Int; <keyword arguments>)
    replaceprime(inds::IndexSet, plold => plnew; <keyword arguments>)
    mapprime(inds, <arguments>; <keyword arguments>)

Set the prime level of the indices of an ITensor or collection of indices with
prime level `plold` to `plnew`.

$priming_tagging_doc
""" mapprime(::ITensor, ::Any...)

@doc """
    swapprime[!](A::ITensor, pl1::Int, pl2::Int; <keyword arguments>) -> ITensor
    swapprime[!](A::ITensor, pl1 => pl2; <keyword arguments>) -> ITensor

    swapprime(inds, pl1::Int, pl2::Int; <keyword arguments>)
    swapprime(inds, pl1 => pl2; <keyword arguments>)

Set the prime level of the indices of an ITensor or collection of indices with
prime level `pl1` to `pl2`, and those with prime level `pl2` to `pl1`.

$priming_tagging_doc
""" swapprime(::ITensor, ::Any...)

@doc """
    addtags[!](A::ITensor, ts::String; <keyword arguments>) -> ITensor

    addtags(inds, ts::String; <keyword arguments>)

Add the tags `ts` to the indices of an ITensor or collection of indices.

$priming_tagging_doc
""" addtags(::ITensor, ::Any...)

@doc """
    removetags[!](A::ITensor, ts::String; <keyword arguments>) -> ITensor

    removetags(inds, ts::String; <keyword arguments>)

Remove the tags `ts` from the indices of an ITensor or collection of indices.

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

Replace the Index `inds1[n]` with the Index `inds2[n]` in the ITensor, where `n`
runs from `1` to `length(inds1) == length(inds2)`.

The indices must have the same space (i.e. the same dimension and QNs, if applicable).

The storage of the ITensor is not modified or copied (the output ITensor is a
view of the input ITensor).
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

Swap the Index `inds1[n]` with the Index `inds2[n]` in the ITensor, where `n`
runs from `1` to `length(inds1) == length(inds2)`.

The indices must have the same space (i.e. the same dimension and QNs, if applicable).

The storage of the ITensor is not modified or copied (the output ITensor is a
view of the input ITensor).
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

# Returns a tuple of pairs of indices, where the pairs
# are determined by the prime level pairs `plev` and
# tag pairs `tags`.
function indpairs(T::ITensor; plev::Pair{Int,Int}=0 => 1, tags::Pair=ts"" => ts"")
  is1 = filterinds(T; plev=first(plev), tags=first(tags))
  is2 = filterinds(T; plev=last(plev), tags=last(tags))
  is2to1 = replacetags(mapprime(is2, last(plev) => first(plev)), last(tags) => first(tags))
  is_first = commoninds(is1, is2to1)
  is_last = replacetags(
    mapprime(is_first, first(plev) => last(plev)), first(tags) => last(tags)
  )
  is_last = permute(commoninds(T, is_last), is_last)
  return is_first .=> is_last
end

#########################
# End ITensor Index Functions
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
