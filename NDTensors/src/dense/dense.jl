#
# Dense storage
#
using LinearAlgebra: BlasFloat

struct Dense{ElT,VecT<:AbstractArray} <: TensorStorage{ElT}
  data::VecT
  function Dense{ElT,VecT}(data::AbstractArray) where {ElT,VecT<:AbstractArray{ElT}}
    return new{ElT,VecT}(data)
  end

  # Special case of Vector
  function Dense{ElT,VecT}(data::Vector) where {ElT,VecT<:AbstractArray{ElT}}
    return new{ElT,VecT}(data)
  end

  # Special case of Array which gets turned into a Vector
  function Dense{ElT,VecT}(data::Array) where {ElT,VecT<:AbstractArray{ElT}}
    return new{ElT,VecT}(vec(data))
  end
end

function Dense(data::VecT) where {VecT<:AbstractArray{ElT}} where {ElT}
  return Dense{ElT,VecT}(data)
end

function Dense(data::Array{ElT}) where {ElT}
  return Dense{ElT,Vector{ElT}}(vec(data))
end

function Dense{ElR}(data::AbstractArray{ElT}) where {ElR,ElT}
  return ElT == ElR ? Dense(data) : Dense(ElR.(data))
end

# Construct from a set of indices
function Dense{ElT,VecT}(inds) where {ElT,VecT<:AbstractArray{ElT}}
  return Dense(VecT(dim(inds)))
end

Dense{ElT}(dim::Integer) where {ElT} = Dense(zeros(ElT, dim))

Dense{ElT}(::UndefInitializer, dim::Integer) where {ElT} = Dense(Vector{ElT}(undef, dim))

Dense(::Type{ElT}, dim::Integer) where {ElT} = Dense{ElT}(dim)

Dense(x::ElT, dim::Integer) where {ElT<:Number} = Dense(fill(x, dim))

Dense(dim::Integer) = Dense(Float64, dim)

Dense(::Type{ElT}, ::UndefInitializer, dim::Integer) where {ElT} = Dense{ElT}(undef, dim)

Dense(::UndefInitializer, dim::Integer) = Dense(Float64, undef, dim)

Dense{ElT}() where {ElT} = Dense(ElT[])
Dense(::Type{ElT}) where {ElT} = Dense{ElT}()

setdata(D::Dense, ndata) = Dense(ndata)

#
# Random constructors
#

function randn(::Type{StoreT}, dim::Integer) where {StoreT<:Dense}
  return Dense(randn(eltype(StoreT), dim))
end

copy(D::Dense) = Dense(copy(data(D)))

Base.real(::Type{Dense{ElT,Vector{ElT}}}) where {ElT} = Dense{real(ElT),Vector{real(ElT)}}

function complex(::Type{Dense{ElT,Vector{ElT}}}) where {ElT}
  return Dense{complex(ElT),Vector{complex(ElT)}}
end

similar(D::Dense) = Dense(similar(data(D)))

similar(D::Dense, length::Int) = Dense(similar(data(D), length))

function similar(storagetype::Type{<:Dense}, length::Int)
  return Dense(similar(datatype(storagetype), length))
end

function similartype(::Type{StoreT}, ::Type{ElT}) where {StoreT<:Dense,ElT}
  return Dense{ElT,similartype(datatype(StoreT), ElT)}
end

# TODO: make these more general, move to tensorstorage.jl
datatype(::Type{<:Dense{<:Any,DataT}}) where {DataT} = DataT

function similar(::Type{StorageT}, ::Type{ElT}, length::Int) where {StorageT<:Dense,ElT}
  return Dense(similar(datatype(StorageT), ElT, length))
end

similar(D::Dense, ::Type{T}) where {T<:Number} = Dense(similar(data(D), T))

zeros(DenseT::Type{<:Dense}, inds) = zeros(DenseT, dim(inds))

# Generic for handling `Vector` and `CuVector`
function zeros(storagetype::Type{<:Dense}, dim::Int)
  return fill!(similar(storagetype, dim), zero(eltype(storagetype)))
end

function promote_rule(
  ::Type{<:Dense{ElT1,VecT1}}, ::Type{<:Dense{ElT2,VecT2}}
) where {ElT1,VecT1,ElT2,VecT2}
  ElR = promote_type(ElT1, ElT2)
  VecR = promote_type(VecT1, VecT2)
  return Dense{ElR,VecR}
end

# This is to get around the issue in Julia that:
# promote_type(Vector{ComplexF32},Vector{Float64}) == Vector{T} where T
function promote_rule(
  ::Type{<:Dense{ElT1,Vector{ElT1}}}, ::Type{<:Dense{ElT2,Vector{ElT2}}}
) where {ElT1,ElT2}
  ElR = promote_type(ElT1, ElT2)
  VecR = Vector{ElR}
  return Dense{ElR,VecR}
end

# This is for type promotion for Scalar*Dense
function promote_rule(
  ::Type{<:Dense{ElT1,Vector{ElT1}}}, ::Type{ElT2}
) where {ElT1,ElT2<:Number}
  ElR = promote_type(ElT1, ElT2)
  VecR = Vector{ElR}
  return Dense{ElR,VecR}
end

function convert(::Type{<:Dense{ElR,VecR}}, D::Dense) where {ElR,VecR}
  return Dense(convert(VecR, data(D)))
end

#
# DenseTensor (Tensor using Dense storage)
#

const DenseTensor{ElT,N,StoreT,IndsT} = Tensor{ElT,N,StoreT,IndsT} where {StoreT<:Dense}

DenseTensor(::Type{ElT}, inds) where {ElT} = tensor(Dense(ElT, dim(inds)), inds)

# Special convenience function for Int
# dimensions
DenseTensor(::Type{ElT}, inds::Int...) where {ElT} = DenseTensor(ElT, inds)

DenseTensor(inds) = tensor(Dense(dim(inds)), inds)

DenseTensor(inds::Int...) = DenseTensor(inds)

function DenseTensor(::Type{ElT}, ::UndefInitializer, inds) where {ElT}
  return tensor(Dense(ElT, undef, dim(inds)), inds)
end

function DenseTensor(::Type{ElT}, ::UndefInitializer, inds::Int...) where {ElT}
  return DenseTensor(ElT, undef, inds)
end

DenseTensor(::UndefInitializer, inds) = tensor(Dense(undef, dim(inds)), inds)

DenseTensor(::UndefInitializer, inds::Int...) = DenseTensor(undef, inds)

# For convenience, direct Tensor constructors default to Dense
Tensor(::Type{ElT}, inds...) where {ElT} = DenseTensor(ElT, inds...)

Tensor(inds...) = Tensor(Float64, inds...)

function Tensor(::Type{ElT}, ::UndefInitializer, inds...) where {ElT}
  return DenseTensor(ElT, undef, inds...)
end

Tensor(::UndefInitializer, inds...) = DenseTensor(undef, inds...)

Tensor(A::Array{<:Number,N}, inds::Dims{N}) where {N} = tensor(Dense(vec(A)), inds)

#
# Random constructors
#

function randomDenseTensor(::Type{ElT}, inds) where {ElT}
  return tensor(randn(Dense{ElT}, dim(inds)), inds)
end

function randomDenseTensor(::Type{ElT}, inds::Int...) where {ElT}
  return randomDenseTensor(ElT, inds)
end

randomDenseTensor(inds) = randomDenseTensor(Float64, inds)

randomDenseTensor(inds::Int...) = randomDenseTensor(Float64, inds)

function randomTensor(::Type{ElT}, inds) where {ElT}
  return randomDenseTensor(ElT, inds)
end

function randomTensor(::Type{ElT}, inds::Int...) where {ElT}
  return randomDenseTensor(ElT, inds...)
end

randomTensor(inds) = randomDenseTensor(Float64, inds)

randomTensor(inds::Int...) = randomDenseTensor(Float64, inds)

# Basic functionality for AbstractArray interface
IndexStyle(::Type{<:DenseTensor}) = IndexLinear()

# Override CartesianIndices iteration to iterate
# linearly through the Dense storage (faster)
iterate(T::DenseTensor, args...) = iterate(storage(T), args...)

function _zeros(TensorT::Type{<:DenseTensor}, inds)
  return tensor(zeros(storagetype(TensorT), dim(inds)), inds)
end

function zeros(TensorT::Type{<:DenseTensor}, inds)
  return _zeros(TensorT, inds)
end

# To fix method ambiguity with zeros(::Type, ::Tuple)
function zeros(TensorT::Type{<:DenseTensor}, inds::Dims)
  return _zeros(TensorT, inds)
end

function zeros(TensorT::Type{<:DenseTensor}, inds::Tuple{})
  return _zeros(TensorT, inds)
end

# To fix method ambiguity with similar(::AbstractArray,::Type)
function similar(T::DenseTensor, ::Type{ElT}) where {ElT}
  return tensor(similar(storage(T), ElT), inds(T))
end

_similar(T::DenseTensor, inds) = similar(typeof(T), inds)

similar(T::DenseTensor, inds) = _similar(T, inds)

# To fix method ambiguity with similar(::AbstractArray,::Tuple)
similar(T::DenseTensor, inds::Dims) = _similar(T, inds)

#
# Single index
#

@propagate_inbounds function getindex(T::DenseTensor{<:Number}, I::Integer...)
  Base.@_inline_meta
  return getindex(data(T), Base._sub2ind(T, I...))
end

@propagate_inbounds function getindex(T::DenseTensor{<:Number}, I::CartesianIndex)
  Base.@_inline_meta
  return getindex(T, I.I...)
end

@propagate_inbounds function setindex!(
  T::DenseTensor{<:Number}, x::Number, I::Vararg{Integer}
)
  Base.@_inline_meta
  setindex!(data(T), x, Base._sub2ind(T, I...))
  return T
end

@propagate_inbounds function setindex!(
  T::DenseTensor{<:Number}, x::Number, I::CartesianIndex
)
  Base.@_inline_meta
  setindex!(T, x, I.I...)
  return T
end

#
# Linear indexing
#

@propagate_inbounds @inline getindex(T::DenseTensor, i::Integer) = storage(T)[i]

@propagate_inbounds @inline function setindex!(T::DenseTensor, v, i::Integer)
  return (storage(T)[i] = v; T)
end

#
# Slicing
# TODO: this doesn't allow colon right now
# Create a DenseView that stores a Dense and an offset?
#

## @propagate_inbounds function _getindex(
##   T::DenseTensor{ElT,N}, I::CartesianIndices{N}
## ) where {ElT,N}
##   storeR = Dense(vec(@view array(T)[I]))
##   indsR = Tuple(I[end] - I[1] + CartesianIndex(ntuple(_ -> 1, Val(N))))
##   return tensor(storeR, indsR)
## end
## 
## @propagate_inbounds function getindex(T::DenseTensor{ElT,N}, I...) where {ElT,N}
##   return _getindex(T, CartesianIndices(I))
## end

@propagate_inbounds function getindex(T::DenseTensor, I...)
  AI = @view array(T)[I...]
  storeR = Dense(vec(AI))
  indsR = size(AI)
  return tensor(storeR, indsR)
end

# Reshape a DenseTensor using the specified dimensions
# This returns a view into the same Tensor data
function reshape(T::DenseTensor, dims)
  dim(T) == dim(dims) || error("Total new dimension must be the same as the old dimension")
  return tensor(storage(T), dims)
end

# This version fixes method ambiguity with AbstractArray reshape
function reshape(T::DenseTensor, dims::Dims)
  dim(T) == dim(dims) || error("Total new dimension must be the same as the old dimension")
  return tensor(storage(T), dims)
end

function reshape(T::DenseTensor, dims::Int...)
  return tensor(storage(T), tuple(dims...))
end

convert(::Type{Array}, T::DenseTensor) = reshape(data(storage(T)), dims(inds(T)))

# Create an Array that is a view of the Dense Tensor
# Useful for using Base Array functions
array(T::DenseTensor) = convert(Array, T)

function Array{ElT,N}(T::DenseTensor{ElT,N}) where {ElT,N}
  return copy(array(T))
end

function Array(T::DenseTensor{ElT,N}) where {ElT,N}
  return Array{ElT,N}(T)
end

# If the storage data are regular Vectors, use Base.copyto!
function copyto!(
  R::Tensor{<:Number,N,<:Dense{<:Number,<:Vector}},
  T::Tensor{<:Number,N,<:Dense{<:Number,<:Vector}},
) where {N}
  RA = array(R)
  TA = array(T)
  RA .= TA
  return R
end

# If they are something more complicated like views, use Strided copyto!
function copyto!(
  R::DenseTensor{<:Number,N,StoreT}, T::DenseTensor{<:Number,N,StoreT}
) where {N,StoreT<:StridedArray}
  RA = array(R)
  TA = array(T)
  @strided RA .= TA
  return R
end

# TODO: call permutedims!(R,T,perm,(r,t)->t)?
function permutedims!(
  R::DenseTensor{<:Number,N,StoreT}, T::DenseTensor{<:Number,N,StoreT}, perm::NTuple{N,Int}
) where {N,StoreT<:StridedArray}
  RA = array(R)
  TA = array(T)
  @strided RA .= permutedims(TA, perm)
  return R
end

function copyto!(R::DenseTensor{<:Number,N}, T::DenseTensor{<:Number,N}) where {N}
  RA = array(R)
  TA = array(T)
  RA .= TA
  return R
end

# TODO: call permutedims!(R,T,perm,(r,t)->t)?
function permutedims!(
  R::DenseTensor{<:Number,N}, T::DenseTensor{<:Number,N}, perm::NTuple{N,Int}
) where {N}
  RA = array(R)
  TA = array(T)
  RA .= permutedims(TA, perm)
  return R
end

function apply!(
  R::DenseTensor{<:Number,N,StoreT},
  T::DenseTensor{<:Number,N,StoreT},
  f::Function=(r, t) -> t,
) where {N,StoreT<:StridedArray}
  RA = array(R)
  TA = array(T)
  @strided RA .= f.(RA, TA)
  return R
end

function apply!(R::DenseTensor, T::DenseTensor, f::Function=(r, t) -> t)
  RA = array(R)
  TA = array(T)
  RA .= f.(RA, TA)
  return R
end

# Version that may overwrite the result or promote
# and return the result
function permutedims!!(
  R::DenseTensor{<:Number,N,StoreT},
  T::DenseTensor{<:Number,N,StoreT},
  perm::Tuple,
  f::Function=(r, t) -> t,
) where {N,StoreT<:StridedArray}
  RR = convert(promote_type(typeof(R), typeof(T)), R)
  RA = ReshapedArray(data(RR), dims(RR), ())
  TA = ReshapedArray(data(T), dims(T), ())
  if !is_trivial_permutation(perm)
    @strided RA .= f.(RA, permutedims(TA, perm))
  else
    # TODO: specialize for specific functions
    RA .= f.(RA, TA)
  end
  return RR
end

function permutedims!!(R::DenseTensor, T::DenseTensor, perm::Tuple, f::Function=(r, t) -> t)
  RR = convert(promote_type(typeof(R), typeof(T)), R)
  RA = ReshapedArray(data(RR), dims(RR), ())
  TA = ReshapedArray(data(T), dims(T), ())
  if !is_trivial_permutation(perm)
    TB = permutedims(TA, perm)
    RA .= f.(RA, TB)
  else
    # TODO: specialize for specific functions
    RA .= f.(RA, TA)
  end
  return RR
end

function permutedims!(
  R::DenseTensor{<:Number,N}, T::DenseTensor{<:Number,N}, perm, f::Function
) where {N}
  if nnz(R) == 1 && nnz(T) == 1
    R[1] = f(R[1], T[1])
    return R
  end
  RA = array(R)
  TA = array(T)
  @strided RA .= f.(RA, permutedims(TA, perm))
  return R
end

function outer!(
  R::DenseTensor{ElR}, T1::DenseTensor{ElT1}, T2::DenseTensor{ElT2}
) where {ElR,ElT1,ElT2}
  if ElT1 != ElT2
    # TODO: use promote instead
    # T1,T2 = promote(T1,T2)

    ElT1T2 = promote_type(ElT1, ElT2)
    if ElT1 != ElT1T2
      # TODO: get this working
      # T1 = ElR.(T1)
      T1 = one(ElT1T2) * T1
    end
    if ElT2 != ElT1T2
      # TODO: get this working
      # T2 = ElR.(T2)
      T2 = one(ElT1T2) * T2
    end
  end

  v1 = data(T1)
  v2 = data(T2)
  RM = reshape(R, length(v1), length(v2))
  #RM .= v1 .* transpose(v2)
  #mul!(RM, v1, transpose(v2))
  _gemm!('N', 'T', one(ElR), v1, v2, zero(ElR), RM)
  return R
end

export backend_auto, backend_blas, backend_generic

@eval struct GemmBackend{T}
  (f::Type{<:GemmBackend})() = $(Expr(:new, :f))
end
GemmBackend(s) = GemmBackend{Symbol(s)}()
macro GemmBackend_str(s)
  return :(GemmBackend{$(Expr(:quote, Symbol(s)))})
end

const gemm_backend = Ref(:Auto)
function backend_auto()
  return gemm_backend[] = :Auto
end
function backend_blas()
  return gemm_backend[] = :BLAS
end
function backend_generic()
  return gemm_backend[] = :Generic
end

@inline function auto_select_backend(
  ::Type{<:StridedVecOrMat{<:BlasFloat}},
  ::Type{<:StridedVecOrMat{<:BlasFloat}},
  ::Type{<:StridedVecOrMat{<:BlasFloat}},
)
  return GemmBackend(:BLAS)
end

@inline function auto_select_backend(
  ::Type{<:AbstractVecOrMat}, ::Type{<:AbstractVecOrMat}, ::Type{<:AbstractVecOrMat}
)
  return GemmBackend(:Generic)
end

function _gemm!(
  tA, tB, alpha, A::TA, B::TB, beta, C::TC
) where {TA<:AbstractVecOrMat,TB<:AbstractVecOrMat,TC<:AbstractVecOrMat}
  if gemm_backend[] == :Auto
    _gemm!(auto_select_backend(TA, TB, TC), tA, tB, alpha, A, B, beta, C)
  else
    _gemm!(GemmBackend(gemm_backend[]), tA, tB, alpha, A, B, beta, C)
  end
end

# BLAS matmul
function _gemm!(
  ::GemmBackend{:BLAS},
  tA,
  tB,
  alpha,
  A::AbstractVecOrMat,
  B::AbstractVecOrMat,
  beta,
  C::AbstractVecOrMat,
)
  #@timeit_debug timer "BLAS.gemm!" begin
  return BLAS.gemm!(tA, tB, alpha, A, B, beta, C)
  #end # @timeit
end

# generic matmul
function _gemm!(
  ::GemmBackend{:Generic},
  tA,
  tB,
  alpha::AT,
  A::AbstractVecOrMat,
  B::AbstractVecOrMat,
  beta::BT,
  C::AbstractVecOrMat,
) where {AT,BT}
  mul!(C, tA == 'T' ? transpose(A) : A, tB == 'T' ? transpose(B) : B, alpha, beta)
  return C
end

# TODO: call outer!!, make this generic
function outer(T1::DenseTensor{ElT1}, T2::DenseTensor{ElT2}) where {ElT1,ElT2}
  array_outer = vec(array(T1)) * transpose(vec(array(T2)))
  inds_outer = unioninds(inds(T1), inds(T2))
  return tensor(Dense{promote_type(ElT1, ElT2)}(vec(array_outer)), inds_outer)
end

function contraction_output(
  ::TensorT1, ::TensorT2, indsR::IndsR
) where {TensorT1<:DenseTensor,TensorT2<:DenseTensor,IndsR}
  TensorR = contraction_output_type(TensorT1, TensorT2, IndsR)
  return similar(TensorR, indsR)
end

Strided.StridedView(T::DenseTensor) = StridedView(convert(Array, T))

# Both are scalar-like tensors
function _contract_scalar!(
  R::DenseTensor{ElR},
  labelsR,
  T1::Number,
  labelsT1,
  T2::Number,
  labelsT2,
  α=one(ElR),
  β=zero(ElR),
) where {ElR}
  if iszero(β)
    R[1] = α * T1 * T2
  elseif iszero(α)
    R[1] = β * R[1]
  else
    R[1] = α * T1 * T2 + β * R[1]
  end
  return R
end

# Trivial permutation
# Version where R and T have different element types, so we can't call BLAS
# Instead use Julia's broadcasting (maybe consider Strided in the future)
function _contract_scalar_noperm!(
  R::DenseTensor{ElR}, T::DenseTensor, α, β=zero(ElR)
) where {ElR}
  Rᵈ = data(R)
  Tᵈ = data(T)
  if iszero(β)
    if iszero(α)
      fill!(Rᵈ, 0)
    else
      Rᵈ .= α .* Tᵈ
    end
  elseif isone(β)
    if iszero(α)
      # No-op
      # Rᵈ .= Rᵈ
    else
      Rᵈ .= α .* Tᵈ .+ Rᵈ
    end
  else
    if iszero(α)
      # Rᵈ .= β .* Rᵈ
      BLAS.scal!(length(Rᵈ), β, Rᵈ, 1)
    else
      Rᵈ .= α .* Tᵈ .+ β .* Rᵈ
    end
  end
  return R
end

# Trivial permutation
# Version where R and T are the same element type, so we can
# call BLAS
function _contract_scalar_noperm!(
  R::DenseTensor{ElR}, T::DenseTensor{ElR}, α, β=zero(ElR)
) where {ElR}
  Rᵈ = data(R)
  Tᵈ = data(T)
  if iszero(β)
    if iszero(α)
      fill!(Rᵈ, 0)
    else
      # Rᵈ .= α .* T₂ᵈ
      BLAS.axpby!(α, Tᵈ, β, Rᵈ)
    end
  elseif isone(β)
    if iszero(α)
      # No-op
      # Rᵈ .= Rᵈ
    else
      # Rᵈ .= α .* Tᵈ .+ Rᵈ
      BLAS.axpy!(α, Tᵈ, Rᵈ)
    end
  else
    if iszero(α)
      # Rᵈ .= β .* Rᵈ
      BLAS.scal!(length(Rᵈ), β, Rᵈ, 1)
    else
      # Rᵈ .= α .* Tᵈ .+ β .* Rᵈ
      BLAS.axpby!(α, Tᵈ, β, Rᵈ)
    end
  end
  return R
end

# Non-trivial permutation
function _contract_scalar_perm!(
  Rᵃ::AbstractArray{ElR}, Tᵃ::AbstractArray, perm, α, β=zero(ElR)
) where {ElR}
  if iszero(β)
    if iszero(α)
      fill!(Rᵃ, 0)
    else
      @strided Rᵃ .= α .* permutedims(Tᵃ, perm)
    end
  elseif isone(β)
    if iszero(α)
      # Rᵃ .= Rᵃ
      # No-op
    else
      @strided Rᵃ .= α .* permutedims(Tᵃ, perm) .+ Rᵃ
    end
  else
    if iszero(α)
      # Rᵃ .= β .* Rᵃ
      BLAS.scal!(length(Rᵃ), β, Rᵃ, 1)
    else
      Rᵃ .= α .* permutedims(Tᵃ, perm) .+ β .* Rᵃ
    end
  end
  return Rᵃ
end

function drop_singletons(::Order{N}, labels, dims) where {N}
  labelsᵣ = ntuple(zero, Val(N))
  dimsᵣ = labelsᵣ
  nkeep = 1
  for n in 1:length(dims)
    if dims[n] > 1
      labelsᵣ = @inbounds setindex(labelsᵣ, labels[n], nkeep)
      dimsᵣ = @inbounds setindex(dimsᵣ, dims[n], nkeep)
      nkeep += 1
    end
  end
  return labelsᵣ, dimsᵣ
end

function _contract_scalar_maybe_perm!(
  ::Order{N}, R::DenseTensor{ElR,NR}, labelsR, T::DenseTensor, labelsT, α, β=zero(ElR)
) where {ElR,NR,N}
  labelsRᵣ, dimsRᵣ = drop_singletons(Order(N), labelsR, dims(R))
  labelsTᵣ, dimsTᵣ = drop_singletons(Order(N), labelsT, dims(T))
  perm = getperm(labelsRᵣ, labelsTᵣ)
  if is_trivial_permutation(perm)
    # trivial permutation
    _contract_scalar_noperm!(R, T, α, β)
  else
    # non-trivial permutation
    Rᵣ = ReshapedArray(data(R), dimsRᵣ, ())
    Tᵣ = ReshapedArray(data(T), dimsTᵣ, ())
    _contract_scalar_perm!(Rᵣ, Tᵣ, perm, α, β)
  end
  return R
end

function _contract_scalar_maybe_perm!(
  R::DenseTensor{ElR,NR}, labelsR, T::DenseTensor, labelsT, α, β=zero(ElR)
) where {ElR,NR}
  N = count(≠(1), dims(R))
  _contract_scalar_maybe_perm!(Order(N), R, labelsR, T, labelsT, α, β)
  return R
end

# XXX: handle case of non-trivial permutation
function _contract_scalar_maybe_perm!(
  R::DenseTensor{ElR,NR},
  labelsR,
  T₁::DenseTensor,
  labelsT₁,
  T₂::DenseTensor,
  labelsT₂,
  α=one(ElR),
  β=zero(ElR),
) where {ElR,NR}
  if nnz(T₁) == 1
    _contract_scalar_maybe_perm!(R, labelsR, T₂, labelsT₂, α * T₁[1], β)
  elseif nnz(T₂) == 1
    _contract_scalar_maybe_perm!(R, labelsR, T₁, labelsT₁, α * T₂[1], β)
  else
    error("In _contract_scalar_perm!, one tensor must be a scalar")
  end
  return R
end

# At least one of the tensors is size 1
function _contract_scalar!(
  R::DenseTensor{ElR},
  labelsR,
  T1::DenseTensor,
  labelsT1,
  T2::DenseTensor,
  labelsT2,
  α=one(ElR),
  β=zero(ElR),
) where {ElR}
  if nnz(T1) == nnz(T2) == 1
    _contract_scalar!(R, labelsR, T1[1], labelsT1, T2[1], labelsT2, α, β)
  else
    _contract_scalar_maybe_perm!(R, labelsR, T1, labelsT1, T2, labelsT2, α, β)
  end
  return R
end

function contract!(
  R::DenseTensor{ElR,NR},
  labelsR,
  T1::DenseTensor{ElT1,N1},
  labelsT1,
  T2::DenseTensor{ElT2,N2},
  labelsT2,
  α::Elα=one(ElR),
  β::Elβ=zero(ElR),
  ;
  kwargs...,
) where {Elα,Elβ,ElR,ElT1,ElT2,NR,N1,N2}
  # Special case for scalar tensors
  if nnz(T1) == 1 || nnz(T2) == 1
    _contract_scalar!(R, labelsR, T1, labelsT1, T2, labelsT2, α, β)
    return R
  end

  if using_tblis() && ElR <: LinearAlgebra.BlasReal && (ElR == ElT1 == ElT2 == Elα == Elβ)
    #@timeit_debug timer "TBLIS contract!" begin
    contract!(Val(:TBLIS), R, labelsR, T1, labelsT1, T2, labelsT2, α, β)
    #end
    return R
  end

  if N1 + N2 == NR
    outer!(R, T1, T2)
    labelsRp = (labelsT1..., labelsT2...)
    perm = getperm(labelsR, labelsRp)
    if !is_trivial_permutation(perm)
      permutedims!(R, copy(R), perm)
    end
    return R
  end

  props = ContractionProperties(labelsT1, labelsT2, labelsR)
  compute_contraction_properties!(props, T1, T2, R)

  if ElT1 != ElT2
    # TODO: use promote instead
    # T1, T2 = promote(T1, T2)

    ElT1T2 = promote_type(ElT1, ElT2)
    if ElT1 != ElR
      # TODO: get this working
      # T1 = ElR.(T1)
      T1 = one(ElT1T2) * T1
    end
    if ElT2 != ElR
      # TODO: get this working
      # T2 = ElR.(T2)
      T2 = one(ElT1T2) * T2
    end
  end

  _contract!(R, T1, T2, props, α, β; kwargs...)
  return R
  #end
end

function _contract!(
  CT::DenseTensor{El,NC},
  AT::DenseTensor{El,NA},
  BT::DenseTensor{El,NB},
  props::ContractionProperties,
  α::Number=one(El),
  β::Number=zero(El),
  ;
  kwargs...,
) where {El,NC,NA,NB}
  # TODO: directly use Tensor instead of Array
  C = ReshapedArray(data(storage(CT)), dims(inds(CT)), ())
  A = ReshapedArray(data(storage(AT)), dims(inds(AT)), ())
  B = ReshapedArray(data(storage(BT)), dims(inds(BT)), ())

  tA = 'N'
  if props.permuteA
    pA = NTuple{NA,Int}(props.PA)
    #@timeit_debug timer "_contract!: permutedims A" begin
    @strided Ap = permutedims(A, pA)
    #end # @timeit
    AM = ReshapedArray(Ap, (props.dmid, props.dleft), ())
    tA = 'T'
  else
    #A doesn't have to be permuted
    if Atrans(props)
      AM = ReshapedArray(A.parent, (props.dmid, props.dleft), ())
      tA = 'T'
    else
      AM = ReshapedArray(A.parent, (props.dleft, props.dmid), ())
    end
  end

  tB = 'N'
  if props.permuteB
    pB = NTuple{NB,Int}(props.PB)
    #@timeit_debug timer "_contract!: permutedims B" begin
    @strided Bp = permutedims(B, pB)
    #end # @timeit
    BM = ReshapedArray(Bp, (props.dmid, props.dright), ())
  else
    if Btrans(props)
      BM = ReshapedArray(B.parent, (props.dright, props.dmid), ())
      tB = 'T'
    else
      BM = ReshapedArray(B.parent, (props.dmid, props.dright), ())
    end
  end

  # TODO: this logic may be wrong
  if props.permuteC
    # Need to copy here since we will be permuting
    # into C later
    CM = ReshapedArray(copy(C), (props.dleft, props.dright), ())
  else
    if Ctrans(props)
      CM = ReshapedArray(C.parent, (props.dright, props.dleft), ())
      (AM, BM) = (BM, AM)
      if tA == tB
        tA = tB = (tA == 'T' ? 'N' : 'T')
      end
    else
      CM = ReshapedArray(C.parent, (props.dleft, props.dright), ())
    end
  end

  _gemm!(tA, tB, El(α), AM, BM, El(β), CM)

  if props.permuteC
    pC = NTuple{NC,Int}(props.PC)
    Cr = ReshapedArray(CM.parent, props.newCrange, ())
    # TODO: use invperm(pC) here?
    #@timeit_debug timer "_contract!: permutedims C" begin
    @strided C .= permutedims(Cr, pC)
    #end # @timeit
  end

  return CT
end

"""
    NDTensors.permute_reshape(T::Tensor,pos...)

Takes a permutation that is split up into tuples. Index positions
within the tuples are combined.

For example:

permute_reshape(T,(3,2),1)

First T is permuted as `permutedims(3,2,1)`, then reshaped such
that the original indices 3 and 2 are combined.
"""
function permute_reshape(
  T::DenseTensor{ElT,NT,IndsT}, pos::Vararg{<:Any,N}
) where {ElT,NT,IndsT,N}
  perm = flatten(pos...)

  length(perm) ≠ NT && error("Index positions must add up to order of Tensor ($N)")
  isperm(perm) || error("Index positions must be a permutation")

  dimsT = dims(T)
  indsT = inds(T)
  if !is_trivial_permutation(perm)
    T = permutedims(T, perm)
  end
  if all(p -> length(p) == 1, pos) && N == NT
    return T
  end
  newdims = MVector(ntuple(_ -> eltype(IndsT)(1), Val(N)))
  for i in 1:N
    if length(pos[i]) == 1
      # No reshape needed, just use the
      # original index
      newdims[i] = indsT[pos[i][1]]
    else
      newdim_i = 1
      for p in pos[i]
        newdim_i *= dimsT[p]
      end
      newdims[i] = eltype(IndsT)(newdim_i)
    end
  end
  newinds = similartype(IndsT, Val{N})(Tuple(newdims))
  return reshape(T, newinds)
end

# svd of an order-n tensor according to positions Lpos
# and Rpos
function LinearAlgebra.svd(
  T::DenseTensor{<:Number,N,IndsT}, Lpos::NTuple{NL,Int}, Rpos::NTuple{NR,Int}; kwargs...
) where {N,IndsT,NL,NR}
  M = permute_reshape(T, Lpos, Rpos)
  UM, S, VM, spec = svd(M; kwargs...)
  u = ind(UM, 2)
  v = ind(VM, 2)

  Linds = similartype(IndsT, Val{NL})(ntuple(i -> inds(T)[Lpos[i]], Val(NL)))
  Uinds = push(Linds, u)

  # TODO: do these positions need to be reversed?
  Rinds = similartype(IndsT, Val{NR})(ntuple(i -> inds(T)[Rpos[i]], Val(NR)))
  Vinds = push(Rinds, v)

  U = reshape(UM, Uinds)
  V = reshape(VM, Vinds)

  return U, S, V, spec
end

# qr decomposition of an order-n tensor according to 
# positions Lpos and Rpos
function LinearAlgebra.qr(
  T::DenseTensor{<:Number,N,IndsT}, Lpos::NTuple{NL,Int}, Rpos::NTuple{NR,Int}; kwargs...
) where {N,IndsT,NL,NR}
  M = permute_reshape(T, Lpos, Rpos)
  QM, RM = qr(M; kwargs...)
  q = ind(QM, 2)
  r = ind(RM, 1)
  # TODO: simplify this by permuting inds(T) by (Lpos,Rpos)
  # then grab Linds,Rinds
  Linds = similartype(IndsT, Val{NL})(ntuple(i -> inds(T)[Lpos[i]], Val(NL)))
  Qinds = push(Linds, r)
  Q = reshape(QM, Qinds)
  Rinds = similartype(IndsT, Val{NR})(ntuple(i -> inds(T)[Rpos[i]], Val(NR)))
  Rinds = pushfirst(Rinds, r)
  R = reshape(RM, Rinds)
  return Q, R
end

# polar decomposition of an order-n tensor according to positions Lpos
# and Rpos
function polar(
  T::DenseTensor{<:Number,N,IndsT}, Lpos::NTuple{NL,Int}, Rpos::NTuple{NR,Int}
) where {N,IndsT,NL,NR}
  M = permute_reshape(T, Lpos, Rpos)
  UM, PM = polar(M)

  # TODO: turn these into functions
  Linds = similartype(IndsT, Val{NL})(ntuple(i -> inds(T)[Lpos[i]], Val(NL)))
  Rinds = similartype(IndsT, Val{NR})(ntuple(i -> inds(T)[Rpos[i]], Val(NR)))

  # Use sim to create "similar" indices, in case
  # the indices have identifiers. If not this should
  # act as an identity operator
  simRinds = sim(Rinds)
  Uinds = (Linds..., simRinds...)
  Pinds = (simRinds..., Rinds...)

  U = reshape(UM, Uinds)
  P = reshape(PM, Pinds)
  return U, P
end

function LinearAlgebra.exp(
  T::DenseTensor{ElT,N}, Lpos::NTuple{NL,Int}, Rpos::NTuple{NR,Int}; ishermitian::Bool=false
) where {ElT,N,NL,NR}
  M = permute_reshape(T, Lpos, Rpos)
  indsTp = permute(inds(T), (Lpos..., Rpos...))
  if ishermitian
    expM = parent(exp(Hermitian(matrix(M))))
    return tensor(Dense{ElT}(vec(expM)), indsTp)
  else
    expM = exp(M)
    return reshape(expM, indsTp)
  end
end

function HDF5.write(
  parent::Union{HDF5.File,HDF5.Group}, name::String, D::Store
) where {Store<:Dense}
  g = create_group(parent, name)
  attributes(g)["type"] = "Dense{$(eltype(Store))}"
  attributes(g)["version"] = 1
  if eltype(D) != Nothing
    write(g, "data", D.data)
  end
end

function HDF5.read(
  parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{Store}
) where {Store<:Dense}
  g = open_group(parent, name)
  ElT = eltype(Store)
  typestr = "Dense{$ElT}"
  if read(attributes(g)["type"]) != typestr
    error("HDF5 group or file does not contain $typestr data")
  end
  if ElT == Nothing
    return Dense{Nothing}()
  end
  # Attribute __complex__ is attached to the "data" dataset
  # by the h5 library used by C++ version of ITensor:
  if haskey(attributes(g["data"]), "__complex__")
    M = read(g, "data")
    nelt = size(M, 1) * size(M, 2)
    data = Vector(reinterpret(ComplexF64, reshape(M, nelt)))
  else
    data = read(g, "data")
  end
  return Dense{ElT}(data)
end

function show(io::IO, mime::MIME"text/plain", T::DenseTensor)
  summary(io, T)
  return print_tensor(io, T)
end
