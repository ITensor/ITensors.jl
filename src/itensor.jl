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

"""
    itensor(args...; kwargs...)

Like the `ITensor` constructor, but with attempt to make a view
of the input data when possible.
"""
itensor(args...; kwargs...)::ITensor = ITensor(AllowAlias(), args...; kwargs...)

ITensor(::AliasStyle, args...; kwargs...)::ITensor =
  error("ITensor constructor with input arguments of types `$(typeof.(args))` not defined.")

"""
    inds(T::ITensor)

Return the indices of the ITensor as a Tuple.
"""
inds(T::ITensor) = inds(tensor(T))

# Trait to check if the tensor has QN symmetry
symmetrystyle(T::Tensor) = symmetrystyle(inds(T))
symmetrystyle(T::ITensor)::SymmetryStyle = symmetrystyle(tensor(T))

"""
    ind(T::ITensor, i::Int)

Get the Index of the ITensor along dimension i.
"""
ind(T::ITensor, i::Int) = ind(tensor(T), i)

"""
    storage(T::ITensor)

Return a view of the TensorStorage of the ITensor.
"""
storage(T::ITensor)::TensorStorage = storage(tensor(T))

"""
    data(T::ITensor)

Return a view of the raw data of the ITensor.

This is mostly an internal ITensor function, please
let the developers of ITensors.jl know if there is
functionality for ITensors that you would like
that is not currently available.
"""
data(T::ITensor) = NDTensors.data(tensor(T))

similar(T::ITensor, args...)::ITensor = itensor(NDTensors.similar(tensor(T), args...))

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

!!! warning
    Unlike standard `AbstractArray{T, N}` types, `ITensor`s do not have their order as type paramater, and therefore iterating using `CartesianIndices` is generally slow. If you are performing operations that use iterating over individual elements of an ITensor it is best to convert to `NDTensors.Tensor`.
"""
CartesianIndices(A::ITensor) = CartesianIndices(tensor(A))

#
# ITensor constructors
#

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

Construct an ITensor filled with zeros having indices `inds` and element type `ElT`. If the element type is not specified, it defaults to `Float64`.

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

Construct an ITensor filled with undefined elements having indices `inds` and element type `ElT`. If the element type is not specified, it defaults to `Float64`. One purpose for using this constructor is that initializing the elements in an undefined way is faster than initializing them to a set value such as zero.

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

const RealOrComplex{T} = Union{T,Complex{T}}

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

ITensor(x::RealOrComplex{Int}, is...) = ITensor(float(x), is...)

#
# EmptyStorage ITensor constructors
#

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

#
# Construct from Array
#

# Helper functions for different view behaviors
Array{ElT,N}(::NeverAlias, A::AbstractArray) where {ElT,N} = Array{ElT,N}(A)
function Array{ElT,N}(::AllowAlias, A::AbstractArray) where {ElT,N}
  return convert(AbstractArray{ElT,N}, A)
end
function Array{ElT}(as::AliasStyle, A::AbstractArray{ElTA,N}) where {ElT,N,ElTA}
  return Array{ElT,N}(as, A)
end

# TODO: Change to:
# (Array{ElT, N} where {ElT})([...]) = [...]
# once support for `VERSION < v"1.6"` is dropped.
# Previous to Julia v1.6 `where` syntax couldn't be used in a function name
function Array{<:Any,N}(as::AliasStyle, A::AbstractArray{ElTA,N}) where {N,ElTA}
  return Array{ElTA,N}(as, A)
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
  inds::Indices{Index{Int}};
  kwargs...,
)
  length(A) ≠ dim(inds) && throw(
    DimensionMismatch(
      "In ITensor(::AbstractArray, inds), length of AbstractArray ($(length(A))) must match total dimension of IndexSet ($(dim(inds)))",
    ),
  )
  data = Array{eltype}(as, A)
  return itensor(Dense(vec(data)), inds)
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

Create an ITensor with all zeros except the specified value,
which is set to 1.

# Examples
```julia
i = Index(2,"i")
A = onehot(i=>2)
# A[i=>2] == 1, all other elements zero

j = Index(3,"j")
B = onehot(i=>1,j=>3)
# B[i=>1,j=>3] == 1, all other element zero
```
"""
function onehot(ivs::Pair{<:Index}...)
  A = emptyITensor(ind.(ivs)...)
  A[val.(ivs)...] = 1.0
  return A
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

eltype(T::ITensor) = eltype(tensor(T))

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

copy(T::ITensor)::ITensor = itensor(copy(tensor(T)))

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
function Array{ElT,N}(T::ITensor, is::Vararg{Index,N}) where {ElT,N}
  ndims(T) != N && throw(
    DimensionMismatch(
      "cannot convert an $(ndims(T)) dimensional ITensor to an $N-dimensional Array."
    ),
  )
  TT = tensor(permute(T, is...))
  return Array{ElT,N}(TT)::Array{ElT,N}
end

function Array{ElT}(T::ITensor, is::Vararg{Index,N}) where {ElT,N}
  return Array{ElT,N}(T, is...)
end

function Array(T::ITensor, is::Vararg{Index,N}) where {N}
  return Array{eltype(T),N}(T, is...)::Array{<:Number,N}
end

function Array{<:Any,N}(T::ITensor, is::Vararg{Index,N}) where {N}
  return Array(T, is...)
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

"""
    scalar(T::ITensor)

Extract the element of an order zero ITensor.

Same as `T[]`.
"""
scalar(T::ITensor)::Any = T[]

lastindex(A::ITensor, n::Int64) = LastVal()
lastindex(A::ITensor) = LastVal()

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

@propagate_inbounds @inline function _getindex(T::Tensor, ivs::Vararg{<:Any,N}) where {N}
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
@propagate_inbounds (getindex(T::ITensor, ivs::Vararg{<:Any,N})::Any) where {N} =
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

# Defining this with the type signature `I::Vararg{Integer, N}` instead of `I::Integere...` is much faster:
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
  T::Tensor, x::Number, ivs::Vararg{<:Any,N}
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
  T::ITensor, x::Number, I::Vararg{<:Any,N}
) where {N}
  return settensor!(T, _setindex!!(tensor(T), x, I...))
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

#function setindex!(::ITensor{Any}, ::Number, ivs...)
#  error("Cannot set the element of an emptyITensor(). Must define indices to set elements")
#end

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
  # Use broadcasting `T .= x`?
  return settensor!(T, fill!!(tensor(T), x))
end

# TODO: name this `inds` or `indscollection`?
itensor2inds(A::ITensor)::Any = inds(A)
itensor2inds(A::Tensor) = inds(A)
itensor2inds(i::Index) = (i,)
itensor2inds(A) = A
function map_itensor2inds(A::Tuple{Vararg{<:Any,N}}) where {N}
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
  ffilter::Function, fset::Function, A::Vararg{<:Any,N}
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

# intersect
@doc """
    commoninds(A, B; kwargs...)

Return a Vector with indices that are common between the indices of `A` and `B` (the set intersection, similar to `Base.intersect`).
""" commoninds

# firstintersect
@doc """
    commonind(A, B; kwargs...)

Return the first `Index` common between the indices of `A` and `B`.

See also [`commoninds`](@ref).
""" commonind

# symdiff
@doc """
    noncommoninds(A, B; kwargs...)

Return a Vector with indices that are not common between the indices of `A` and `B` (the symmetric set difference, similar to `Base.symdiff`).
""" noncommoninds

# firstsymdiff
@doc """
    noncommonind(A, B; kwargs...)

Return the first `Index` not common between the indices of `A` and `B`.

See also [`noncommoninds`](@ref).
""" noncommonind

# setdiff
@doc """
    uniqueinds(A, B; kwargs...)

Return Vector with indices that are unique to the set of indices of `A` and not in `B` (the set difference, similar to `Base.setdiff`).
""" uniqueinds

# firstsetdiff
@doc """
    uniqueind(A, B; kwargs...)

Return the first `Index` unique to the set of indices of `A` and not in `B`.

See also [`uniqueinds`](@ref).
""" uniqueind

# union
@doc """
    unioninds(A, B; kwargs...)

Return a Vector with indices that are the union of the indices of `A` and `B` (the set union, similar to `Base.union`).
""" unioninds

# firstunion
@doc """
    unionind(A, B; kwargs...)

Return the first `Index` in the union of the indices of `A` and `B`.

See also [`unioninds`](@ref).
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

The ITensor functions come in two versions, `f` and `f!`. The latter modifies the ITensor in-place. In both versions, the ITensor storage is not modified or copied (so it returns an ITensor with a view of the original storage).
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

Set the prime level of the indices of an ITensor or collection of indices with prime level `plold` to `plnew`.

$priming_tagging_doc
""" mapprime(::ITensor, ::Any...)

@doc """
    swapprime[!](A::ITensor, pl1::Int, pl2::Int; <keyword arguments>) -> ITensor
    swapprime[!](A::ITensor, pl1 => pl2; <keyword arguments>) -> ITensor

    swapprime(inds, pl1::Int, pl2::Int; <keyword arguments>)
    swapprime(inds, pl1 => pl2; <keyword arguments>)

Set the prime level of the indices of an ITensor or collection of indices with prime level `pl1` to `pl2`, and those with prime level `pl2` to `pl1`.

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
  !hassameinds(A, B) && return false
  return norm(A - B) == zero(promote_type(eltype(A), eltype(B)))
end

function isapprox(A::ITensor, B::ITensor; kwargs...)
  if !hassameinds(A, B)
    error(
      "In `isapprox(::ITensor, ::ITensor)`, the indices of the ITensors do not match. The first ITensor has indices: \n\n$(inds(A))\n\nbut the second ITensor has indices: \n\n$(inds(B))",
    )
  end
  B = permute(B, inds(A))
  return isapprox(array(A), array(B); kwargs...)
end

function randn!(T::ITensor)
  return settensor!(T, randn!!(tensor(T)))
end

"""
    randomITensor([::Type{ElT <: Number} = Float64, ]inds)
    randomITensor([::Type{ElT <: Number} = Float64, ]inds::Index...)

Construct an ITensor with type `ElT` and indices `inds`, whose elements are normally distributed random numbers. If the element type is not specified, it defaults to `Float64`.

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
  T = ITensor(S, undef, is)
  randn!(T)
  return T
end

function randomITensor(::Type{S}, is...) where {S<:Number}
  return randomITensor(S, indices(is...))
end

# To fix ambiguity errors with QN version
function randomITensor(::Type{ElT}) where {ElT<:Number}
  return randomITensor(ElT, ())
end

randomITensor(is::Indices) = randomITensor(Float64, is)
randomITensor(is...) = randomITensor(Float64, indices(is...))

# To fix ambiguity errors with QN version
randomITensor() = randomITensor(Float64, ())

function combiner(is::Indices; kwargs...)
  tags = get(kwargs, :tags, "CMB,Link")
  new_ind = Index(prod(dims(is)), tags)
  new_is = (new_ind, is...)
  return itensor(Combiner(), new_is)
end

combiner(is...; kwargs...) = combiner(indices(is...); kwargs...)
combiner(i::Index; kwargs...) = combiner((i,); kwargs...)

# Special case when no indices are combined (useful for generic code)
function combiner(; kwargs...)
  return itensor(Combiner(), ())
end

function combinedind(T::ITensor)
  if storage(T) isa Combiner && order(T) > 0
    return inds(T)[1]
  end
  return nothing
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

"""
    dir(A::ITensor, i::Index)

Return the direction of the Index `i` in the ITensor `A`.
"""
dir(A::ITensor, i::Index) = dir(inds(A), i)

"""
    permute(T::ITensor, inds...; allow_alias = false)

Return a new ITensor `T` with indices permuted according
to the input indices `inds`. The storage of the ITensor
is permuted accordingly.

If called with `allow_alias = true`, it avoids
copying data if possible. Therefore, it may return an alias
of the input ITensor (an ITensor that shares the same data),
such as if the permutation turns out to be trivial.

By default, `allow_alias = false`, and it never
returns an alias of the input ITensor.

# Examples

```julia
i = Index(2, "index_i"); j = Index(4, "index_j"); k = Index(3, "index_k");
T = randomITensor(i, j, k)

pT_1 = permute(T, k, i, j)
pT_2 = permute(T, j, i, k)

pT_noalias_1 = permute(T, i, j, k)
pT_noalias_1[1, 1, 1] = 12
T[1, 1, 1] != pT_noalias_1[1, 1, 1]

pT_noalias_2 = permute(T, i, j, k; allow_alias = false)
pT_noalias_2[1, 1, 1] = 12
T[1, 1, 1] != pT_noalias_1[1, 1, 1]

pT_alias = permute(T, i, j, k; allow_alias = true)
pT_alias[1, 1, 1] = 12
T[1, 1, 1] == pT_alias[1, 1, 1]
```
"""
function permute(T::ITensor, new_inds...; kwargs...)
  if !hassameinds(T, indices(new_inds))
    error(
      "In `permute(::ITensor, inds...)`, the input ITensor has indices: \n\n$(inds(T))\n\nbut the desired Index ordering is: \n\n$(indices(new_inds))",
    )
  end
  allow_alias = deprecated_keyword_argument(
    Bool,
    kwargs;
    new_kw=:allow_alias,
    old_kw=:always_copy,
    default=false,
    funcsym=:permute,
    map=!,
  )
  aliasstyle::Union{AllowAlias,NeverAlias} = allow_alias ? AllowAlias() : NeverAlias()
  return permute(aliasstyle, T, new_inds...)
end

# TODO: move to NDTensors
function permutedims(::AllowAlias, T::Tensor, perm)
  return NDTensors.is_trivial_permutation(perm) ? T : permutedims(NeverAlias(), T, perm)
end

# TODO: move to NDTensors, define `permutedims` in terms of `NeverAlias`
function permutedims(::NeverAlias, T::Tensor, perm)
  return permutedims(T, perm)
end

function _permute(as::AliasStyle, T::Tensor, new_inds)
  perm = NDTensors.getperm(new_inds, inds(T))
  return permutedims(as, T, perm)
end

function permute(as::AliasStyle, T::ITensor, new_inds)
  return itensor(_permute(as, tensor(T), new_inds))
end

# Version listing indices
function permute(as::AliasStyle, T::ITensor, new_inds::Index...)
  return permute(as, T, new_inds)
end

function (T::ITensor * x::Number)::ITensor
  return itensor(x * tensor(T))
end

# TODO: what about noncommutative number types?
(x::Number * T::ITensor) = T * x

#TODO: make a proper element-wise division
(A::ITensor / x::Number) = A * (1.0 / x)

-(A::ITensor) = itensor(-tensor(A))

_isemptyscalar(A::ITensor) = _isemptyscalar(tensor(A))
_isemptyscalar(A::Tensor) = ndims(A) == 0 && isemptystorage(A) && eltype(A) === EmptyNumber

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

Base.real(T::ITensor)::ITensor = itensor(real(tensor(T)))

Base.imag(T::ITensor)::ITensor = itensor(imag(tensor(T)))

Base.conj(T::ITensor)::ITensor = itensor(conj(tensor(T)))

# Function barrier
function _contract(A::Tensor, B::Tensor)
  labelsA, labelsB = compute_contraction_labels(inds(A), inds(B))
  return contract(A, labelsA, B, labelsB)
  # TODO: Alternative to try (`noncommoninds` is too slow right now)
  #return _contract!!(EmptyTensor(Float64, _Tuple(noncommoninds(inds(A), inds(B)))), A, B)
end

function _contract(A::ITensor, B::ITensor)::ITensor
  C = itensor(_contract(tensor(A), tensor(B)))
  warnTensorOrder = get_warn_order()
  if !isnothing(warnTensorOrder) > 0 && order(C) >= warnTensorOrder
    println(
      "Contraction resulted in ITensor with $(order(C)) indices, which is greater than or equal to the ITensor order warning threshold $warnTensorOrder. You can modify the threshold with macros like `@set_warn_order N`, `@reset_warn_order`, and `@disable_warn_order` or functions like `ITensors.set_warn_order(N::Int)`, `ITensors.reset_warn_order()`, and `ITensors.disable_warn_order()`.",
    )
    # This prints a vector, not formatted well
    #show(stdout, MIME"text/plain"(), stacktrace())
    Base.show_backtrace(stdout, backtrace())
    println()
  end
  return C
end

_contract(T::ITensor, ::Nothing) = T

dag(::Nothing) = nothing

# TODO: add iscombiner(::Tensor) to NDTensors
iscombiner(T::ITensor)::Bool = (storage(T) isa Combiner)

# TODO: add isdiag(::Tensor) to NDTensors
isdiag(T::ITensor)::Bool = (storage(T) isa Diag || storage(T) isa DiagBlockSparse)

function can_combine_contract(A::ITensor, B::ITensor)::Bool
  return hasqns(A) &&
         hasqns(B) &&
         !iscombiner(A) &&
         !iscombiner(B) &&
         !isdiag(A) &&
         !isdiag(B)
end

function combine_contract(A::ITensor, B::ITensor)::ITensor
  # Combine first before contracting
  C::ITensor = if can_combine_contract(A, B)
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
    contract(A::ITensor, B::ITensor)

Contract ITensors A and B to obtain a new ITensor. This 
contraction `*` operator finds all matching indices common
to A and B and sums over them, such that the result will 
have only the unique indices of A and B. To prevent
indices from matching, their prime level or tags can be 
modified such that they no longer compare equal - for more
information see the documentation on Index objects.

# Examples

```julia
i = Index(2,"index_i"); j = Index(4,"index_j"); k = Index(3,"index_k")

A = randomITensor(i,j)
B = randomITensor(j,k)
C = A * B # contract over Index j

A = randomITensor(i,i')
B = randomITensor(i,i'')
C = A * B # contract over Index i

A = randomITensor(i)
B = randomITensor(j)
C = A * B # outer product of A and B, no contraction

A = randomITensor(i,j,k)
B = randomITensor(k,i,j)
C = A * B # inner product of A and B, all indices contracted
```
"""
(A::ITensor * B::ITensor)::ITensor = contract(A, B)

function contract(A::ITensor, B::ITensor)::ITensor
  NA::Int = ndims(A)
  NB::Int = ndims(B)
  if NA == 0 && NB == 0
    return (iscombiner(A) || iscombiner(B)) ? _contract(A, B) : ITensor(A[] * B[])
  elseif NA == 0
    return iscombiner(A) ? _contract(A, B) : A[] * B
  elseif NB == 0
    return iscombiner(B) ? _contract(B, A) : B[] * A
  else
    C = using_combine_contract() ? combine_contract(A, B) : _contract(A, B)
    return C
  end
end

function optimal_contraction_sequence(A::Union{Vector{<:ITensor},Tuple{Vararg{<:ITensor}}})
  if length(A) == 1
    return optimal_contraction_sequence(A[1])
  elseif length(A) == 2
    return optimal_contraction_sequence(A[1], A[2])
  elseif length(A) == 3
    return optimal_contraction_sequence(A[1], A[2], A[3])
  else
    return _optimal_contraction_sequence(A)
  end
end

optimal_contraction_sequence(A::ITensor) = Any[1]
optimal_contraction_sequence(A1::ITensor, A2::ITensor) = Any[1, 2]
function optimal_contraction_sequence(A1::ITensor, A2::ITensor, A3::ITensor)
  return optimal_contraction_sequence(inds(A1), inds(A2), inds(A3))
end
optimal_contraction_sequence(As::ITensor...) = _optimal_contraction_sequence(As)

_optimal_contraction_sequence(As::Tuple{<:ITensor}) = Any[1]
_optimal_contraction_sequence(As::Tuple{<:ITensor,<:ITensor}) = Any[1, 2]
function _optimal_contraction_sequence(As::Tuple{<:ITensor,<:ITensor,<:ITensor})
  return optimal_contraction_sequence(inds(As[1]), inds(As[2]), inds(As[3]))
end
function _optimal_contraction_sequence(As::Tuple{Vararg{<:ITensor}})
  return __optimal_contraction_sequence(As)
end

_optimal_contraction_sequence(As::Vector{<:ITensor}) = __optimal_contraction_sequence(As)

function __optimal_contraction_sequence(As)
  indsAs = [inds(A) for A in As]
  return optimal_contraction_sequence(indsAs)
end

function default_sequence()
  return using_contraction_sequence_optimization() ? "automatic" : "left_associative"
end

function contraction_cost(As::Union{Vector{<:ITensor},Tuple{Vararg{<:ITensor}}}; kwargs...)
  indsAs = [inds(A) for A in As]
  return contraction_cost(indsAs; kwargs...)
end

# TODO: provide `contractl`/`contractr`/`*ˡ`/`*ʳ` as shorthands for left associative and right associative contractions.
"""
    *(As::ITensor...; sequence = default_sequence(), kwargs...)
    *(As::Vector{<: ITensor}; sequence = default_sequence(), kwargs...)
    contract(As::ITensor...; sequence = default_sequence(), kwargs...)

Contract the set of ITensors according to the contraction sequence.

The default sequence is "automatic" if `ITensors.using_contraction_sequence_optimization()`
is true, otherwise it is "left_associative" (the ITensors are contracted from left to right).

You can change the default with `ITensors.enable_contraction_sequence_optimization()` and
`ITensors.disable_contraction_sequence_optimization()`.

For a custom sequence, the sequence should be provided as a binary tree where the leaves are
integers `n` specifying the ITensor `As[n]` and branches are accessed
by indexing with `1` or `2`, i.e. `sequence = Any[Any[1, 3], Any[2, 4]]`.
"""
function contract(tn::AbstractVector; kwargs...)
  return if all(x -> x isa ITensor, tn)
    contract(convert(Vector{ITensor}, tn); kwargs...)
  else
    deepcontract(tn; kwargs...)
  end
end

# Contract a tensor network such as:
# [A, B, [[C, D], [E, [F, G]]]]
deepcontract(t::ITensor, ts::ITensor...) = *(t, ts...)
function deepcontract(tn::AbstractVector)
  return deepcontract(deepcontract.(tn)...)
end

function contract(
  As::Union{Vector{ITensor},Tuple{Vararg{ITensor}}}; sequence=default_sequence(), kwargs...
)::ITensor
  if sequence == "left_associative"
    return foldl((A, B) -> contract(A, B; kwargs...), As)
  elseif sequence == "right_associative"
    return foldr((A, B) -> contract(A, B; kwargs...), As)
  elseif sequence == "automatic"
    return _contract(As, optimal_contraction_sequence(As); kwargs...)
  else
    return _contract(As, sequence; kwargs...)
  end
end

contract(As::ITensor...; kwargs...)::ITensor = contract(As; kwargs...)

_contract(As, sequence::Int) = As[sequence]

# Given a contraction sequence, contract the tensors recursively according
# to that sequence.
function _contract(As, sequence::AbstractVector; kwargs...)::ITensor
  return contract(_contract.((As,), sequence)...; kwargs...)
end

*(As::ITensor...; kwargs...)::ITensor = contract(As...; kwargs...)

#! format: off
# Turns off formatting since JuliaFormatter tries to change β to a keyword argument, i.e.
# contract!(C::ITensor, A::ITensor, B::ITensor, α::Number; β::Number=0)::ITensor
function contract!(C::ITensor, A::ITensor, B::ITensor, α::Number, β::Number=0)::ITensor
#! format: on
  labelsCAB = compute_contraction_labels(inds(C), inds(A), inds(B))
  labelsC, labelsA, labelsB = labelsCAB
  CT = NDTensors.contract!!(
    Tensor(C), _Tuple(labelsC), tensor(A), _Tuple(labelsA), Tensor(B), _Tuple(labelsB), α, β
  )
  setstorage!(C, storage(CT))
  setinds!(C, inds(C))
  return C
end

function _contract!!(C::Tensor, A::Tensor, B::Tensor)
  labelsCAB = compute_contraction_labels(inds(C), inds(A), inds(B))
  labelsC, labelsA, labelsB = labelsCAB
  CT = NDTensors.contract!!(C, labelsC, A, labelsA, B, labelsB)
  return CT
end

# This is necessary for now since not all types implement contract!!
# with non-trivial α and β
function contract!(C::ITensor, A::ITensor, B::ITensor)::ITensor
  return settensor!(C, _contract!!(Tensor(C), tensor(A), Tensor(B)))
end

mul!(C::ITensor, A::ITensor, B::ITensor, args...)::ITensor = contract!(C, A, B, args...)

dot(A::ITensor, B::ITensor) = (dag(A) * B)[]

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

# Trace an ITensor over pairs of indices determined by
# the prime levels and tags. Indices that are not in pairs
# are not traced over, corresponding to a "batched" trace.
function tr(T::ITensor; plev::Pair{Int,Int}=0 => 1, tags::Pair=ts"" => ts"")
  trpairs = indpairs(T; plev=plev, tags=tags)
  Cᴸ = combiner(first.(trpairs))
  Cᴿ = combiner(last.(trpairs))
  Tᶜ = T * Cᴸ * Cᴿ
  cᴸ = uniqueind(Cᴸ, T)
  cᴿ = uniqueind(Cᴿ, T)
  Tᶜ *= δ(dag((cᴸ, cᴿ)))
  if order(Tᶜ) == 0
    return Tᶜ[]
  end
  return Tᶜ
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
function exp(A::ITensor, Linds, Rinds; kwargs...)
  ishermitian = get(kwargs, :ishermitian, false)

  @debug_check begin
    if hasqns(A)
      @assert flux(A) == QN()
    end
  end

  N = ndims(A)
  NL = length(Linds)
  NR = length(Rinds)
  NL != NR && error("Must have equal number of left and right indices")
  N != NL + NR &&
    error("Number of left and right indices must add up to total number of indices")

  # Linds, Rinds may not have the correct directions
  # TODO: does the need a conversion?
  Lis = Linds
  Ris = Rinds

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

  CL = combiner(Lis...; dir=Out)
  CR = combiner(Ris...; dir=In)
  AC = A * CR * CL
  expAT = ishermitian ? exp(Hermitian(Tensor(AC))) : exp(Tensor(AC))
  return itensor(expAT) * dag(CR) * dag(CL)
end

function exp(A::ITensor; kwargs...)
  Ris = filterinds(A; plev=0)
  Lis = Ris'
  return exp(A, Lis, Ris; kwargs...)
end

"""
    hadamard_product!(C::ITensor, A::ITensor, B::ITensor)
    hadamard_product(A::ITensor, B::ITensor)
    ⊙(A::ITensor, B::ITensor)

Elementwise product of 2 ITensors with the same indices.

Alternative syntax `⊙` can be typed in the REPL with `\\odot <tab>`.
"""
function hadamard_product!(R::ITensor, T1::ITensor, T2::ITensor)
  if !hassameinds(T1, T2)
    error("ITensors must have some indices to perform Hadamard product")
  end
  # Permute the indices to the same order
  #if inds(A) ≠ inds(B)
  #  B = permute(B, inds(A))
  #end
  #Tensor(C) .= tensor(A) .* Tensor(B)
  map!((t1, t2) -> *(t1, t2), R, T1, T2)
  return R
end

# TODO: instead of copy, use promote(A, B)
function hadamard_product(A::ITensor, B::ITensor)
  Ac = copy(A)
  return hadamard_product!(Ac, Ac, B)
end

⊙(A::ITensor, B::ITensor) = hadamard_product(A, B)

# Helper tensors for performing a partial direct sum
function directsum_itensors(i::Index, j::Index, ij::Index)
  S1 = zeros(dim(i), dim(ij))
  for ii in 1:dim(i)
    S1[ii, ii] = 1
  end
  S2 = zeros(dim(j), dim(ij))
  for jj in 1:dim(j)
    S2[jj, dim(i) + jj] = 1
  end
  D1 = itensor(S1, dag(i), ij)
  D2 = itensor(S2, dag(j), ij)
  return D1, D2
end

function directsum(A_and_I::Pair{ITensor}, B_and_J::Pair{ITensor}; kwargs...)
  A, I = A_and_I
  B, J = B_and_J
  return directsum(A, B, I, J; kwargs...)
end

"""
    directsum(A::Pair{ITensor}, B::Pair{ITensor}, ...; tags)

Given a list of pairs of ITensors and collections of indices, perform a partial
direct sum of the tensors over the specified indices. Indices that are
not specified to be summed must match between the tensors.

If all indices are specified then the operation is equivalent to creating
a block diagonal tensor.

Returns the ITensor representing the partial direct sum as well as the new
direct summed indices. The tags of the direct summed indices are specified
by the keyword arguments.

See Section 2.3 of https://arxiv.org/abs/1405.7786 for a definition of a partial
direct sum of tensors.

# Examples
```julia
x = Index(2, "x")
i1 = Index(3, "i1")
j1 = Index(4, "j1")
i2 = Index(5, "i2")
j2 = Index(6, "j2")

A1 = randomITensor(i1, x, j1)
A2 = randomITensor(x, j2, i2)
S, s = ITensors.directsum(A1 => (i1, j1), A2 => (i2, j2); tags = ["sum_i", "sum_j"])
```
"""
function directsum(
  A_and_I::Pair{ITensor},
  B_and_J::Pair{ITensor},
  C_and_K::Pair{ITensor},
  itensor_and_inds...;
  tags=["sum$i" for i in 1:length(last(A_and_I))],
)
  return directsum(
    Pair(directsum(A_and_I, B_and_J; kwargs...)...), C_and_K, itensor_and_inds...; tags=tags
  )
end

function directsum(A::ITensor, B::ITensor, I, J; tags)
  N = length(I)
  (N != length(J)) &&
    error("In directsum(::ITensor, ::ITensor, ...), must sum equal number of indices")
  IJ = Vector{Base.promote_eltype(I, J)}(undef, N)
  for n in 1:N
    In = I[n]
    Jn = J[n]
    In = dir(A, In) != dir(In) ? dag(In) : In
    Jn = dir(B, Jn) != dir(Jn) ? dag(Jn) : Jn
    IJn = directsum(In, Jn; tags=tags[n])
    D1, D2 = directsum_itensors(In, Jn, IJn)
    IJ[n] = IJn
    A *= D1
    B *= D2
  end
  C = A + B
  return C, IJ
end

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
function product(A::ITensor, B::ITensor; apply_dag::Bool=false)
  commonindsAB = commoninds(A, B; plev=0)
  isempty(commonindsAB) && error("In product, must have common indices with prime level 0.")
  common_paired_indsA = filterinds(
    i -> hasind(commonindsAB, i) && hasind(A, setprime(i, 1)), A
  )
  common_paired_indsB = filterinds(
    i -> hasind(commonindsAB, i) && hasind(B, setprime(i, 1)), B
  )

  if !isempty(common_paired_indsA)
    commoninds_pairs = unioninds(common_paired_indsA, common_paired_indsA')
  elseif !isempty(common_paired_indsB)
    commoninds_pairs = unioninds(common_paired_indsB, common_paired_indsB')
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
    A′ = prime(A; inds=!danglings_inds)
    AB = mapprime(A′ * B, 2 => 1; inds=!danglings_inds)
    if apply_dag
      AB′ = prime(AB; inds=!danglings_inds)
      Adag = swapprime(dag(A), 0 => 1; inds=!danglings_inds)
      return mapprime(AB′ * Adag, 2 => 1; inds=!danglings_inds)
    end
    return AB
  elseif isempty(common_paired_indsA) && !isempty(common_paired_indsB)
    # vector-matrix product
    apply_dag && error("apply_dag not supported for matrix-vector product")
    A′ = prime(A; inds=!danglings_inds)
    return A′ * B
  elseif !isempty(common_paired_indsA) && isempty(common_paired_indsB)
    # matrix-vector product
    apply_dag && error("apply_dag not supported for vector-matrix product")
    return noprime(A * B; inds=!danglings_inds)
  end
end

"""
    product(As::Vector{<:ITensor}, A::ITensor)

Product the ITensors pairwise.
"""
function product(As::Vector{<:ITensor}, B::ITensor; kwargs...)
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

flux(T::Union{Tensor,ITensor}, args...) = flux(inds(T), args...)

"""
    flux(T::ITensor)

Returns the flux of the ITensor.

If the ITensor is empty or it has no QNs, returns `nothing`.
"""
function flux(T::Union{Tensor,ITensor})
  (!hasqns(T) || isempty(T)) && return nothing
  @debug_check checkflux(T)
  block1 = first(eachnzblock(T))
  return flux(T, block1)
end

function checkflux(T::Union{Tensor,ITensor}, flux_check)
  for b in nzblocks(T)
    fluxTb = flux(T, b)
    if fluxTb != flux_check
      error(
        "Block $b has flux $fluxTb that is inconsistent with the desired flux $flux_check"
      )
    end
  end
  return nothing
end

function checkflux(T::Union{Tensor,ITensor})
  b1 = first(nzblocks(T))
  fluxTb1 = flux(T, b1)
  return checkflux(T, fluxTb1)
end

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

# XXX: rename isemptystorage?
"""
    isemptystorage(T::ITensor)

Returns `true` if the ITensor contains no elements.

An ITensor with `EmptyStorage` storage always returns `true`.
"""
isemptystorage(T::ITensor) = isemptystorage(tensor(T))
isemptystorage(T::Tensor) = isempty(T)
isempty(T::ITensor) = isemptystorage(T)

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
function matrix(T::ITensor)
  ndims(T) != 2 && throw(DimensionMismatch())
  return array(tensor(T))
end

"""
    vector(T::ITensor)

Given an ITensor `T` with one index, returns
a Vector with a copy of the ITensor's elements,
or a view in the case the ITensor's storage is Dense.
"""
function vector(T::ITensor)
  ndims(T) != 1 && throw(DimensionMismatch())
  return array(tensor(T))
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
  return error(
    "HDF5 file: $(g) does not contain correct ITensor data.\nNeither key `store` nor `storage` could be found.",
  )
end
