#
# EmptyTensor (Tensor using EmptyStorage storage)
#

const EmptyTensor{ElT,N,StoreT,IndsT} =
  Tensor{ElT,N,StoreT,IndsT} where {StoreT<:EmptyStorage}

## Start constructors
function EmptyTensor(::Type{ElT}, inds) where {ElT<:Number}
  return tensor(EmptyStorage(ElT), inds)
end

function EmptyTensor(::Type{StoreT}, inds) where {StoreT<:TensorStorage}
  return tensor(empty(StoreT), inds)
end

function EmptyBlockSparseTensor(::Type{ElT}, inds) where {ElT<:Number}
  StoreT = BlockSparse{ElT,Vector{ElT},length(inds)}
  return EmptyTensor(StoreT, inds)
end
## End constructors

fulltype(::Type{EmptyStorage{ElT,StoreT}}) where {ElT,StoreT} = StoreT
fulltype(T::EmptyStorage) = fulltype(typeof(T))

fulltype(T::Tensor) = fulltype(typeof(T))

# Needed for correct `NDTensors.ndims` definitions, for
# example `EmptyStorage` that wraps a `BlockSparse` which
# can have non-unity dimensions.
function ndims(storagetype::Type{<:EmptyStorage})
  return ndims(fulltype(storagetype))
end

# From an EmptyTensor, return the closest Tensor type
function fulltype(::Type{TensorT}) where {TensorT<:Tensor}
  return Tensor{
    eltype(TensorT),ndims(TensorT),fulltype(storetype(TensorT)),indstype(TensorT)
  }
end

function fulltype(
  ::Type{ElR}, ::Type{<:Tensor{ElT,N,EStoreT,IndsT}}
) where {ElR,ElT<:Number,N,EStoreT<:EmptyStorage{ElT,StoreT},IndsT} where {StoreT}
  return Tensor{ElR,N,similartype(StoreT, ElR),IndsT}
end

function emptytype(::Type{TensorT}) where {TensorT<:Tensor}
  return Tensor{
    eltype(TensorT),ndims(TensorT),emptytype(storagetype(TensorT)),indstype(TensorT)
  }
end

# XXX TODO: add bounds checking
getindex(T::EmptyTensor, I::Integer...) = zero(eltype(T))

function getindex(T::EmptyTensor{Complex{EmptyNumber}}, I::Integer...)
  return Complex(EmptyNumber(), EmptyNumber())
end

similar(T::EmptyTensor, inds::Tuple) = setinds(T, inds)
function similar(T::EmptyTensor, ::Type{ElT}) where {ElT<:Number}
  return tensor(similar(storage(T), ElT), inds(T))
end

function randn!!(T::EmptyTensor)
  return randn!!(Random.default_rng(), T)
end

function randn!!(rng::AbstractRNG, T::EmptyTensor)
  Tf = similar(fulltype(T), inds(T))
  randn!(rng, Tf)
  return Tf
end

# Default to Float64
function randn!!(T::EmptyTensor{EmptyNumber})
  return randn!!(Random.default_rng(), T)
end

# Default to Float64
function randn!!(rng::AbstractRNG, T::EmptyTensor{EmptyNumber})
  return randn!!(rng, similar(T, Float64))
end

function _fill!!(::Type{ElT}, T::EmptyTensor, α::Number) where {ElT}
  Tf = similar(fulltype(T), ElT, inds(T))
  fill!(Tf, α)
  return Tf
end

fill!!(T::EmptyTensor, α::Number) = _fill!!(eltype(T), T, α)

# Determine the element type from the number you are filling with
fill!!(T::EmptyTensor{EmptyNumber}, α::Number) = _fill!!(eltype(α), T, α)

isempty(::EmptyTensor) = true

function zeros(T::TensorT) where {TensorT<:EmptyTensor}
  TensorR = fulltype(TensorT)
  return zeros(TensorR, inds(T))
end

function zeros(::Type{ElT}, T::TensorT) where {ElT,TensorT<:EmptyTensor}
  TensorR = fulltype(ElT, TensorT)
  return zeros(TensorR, inds(T))
end

function insertblock(T::EmptyTensor{<:Number,N}, block) where {N}
  R = zeros(T)
  insertblock!(R, Block(block))
  return R
end

insertblock!!(T::EmptyTensor{<:Number,N}, block) where {N} = insertblock(T, block)

blockoffsets(tensor::EmptyTensor) = BlockOffsets{ndims(tensor)}()

# Special case with element type of EmptyNumber: storage takes the type
# of the input.
@propagate_inbounds function _setindex(T::EmptyTensor{EmptyNumber}, x, I...)
  R = zeros(typeof(x), T)
  R[I...] = x
  return R
end

# Special case with element type of Complex{EmptyNumber}: storage takes the type
# of the complex version of the input.
@propagate_inbounds function _setindex(T::EmptyTensor{Complex{EmptyNumber}}, x, I...)
  R = zeros(typeof(complex(x)), T)
  R[I...] = x
  return R
end

@propagate_inbounds function _setindex(T::EmptyTensor, x, I...)
  R = zeros(T)
  R[I...] = x
  return R
end

@propagate_inbounds function setindex(T::EmptyTensor, x, I...)
  return _setindex(T, x, I...)
end

# This is needed to fix an ambiguity error with ArrayInterface.jl
# https://github.com/ITensor/NDTensors.jl/issues/62
@propagate_inbounds function setindex(T::EmptyTensor, x, I::Int...)
  return _setindex(T, x, I...)
end

setindex!!(T::EmptyTensor, x, I...) = setindex(T, x, I...)

promote_rule(::Type{EmptyNumber}, ::Type{T}) where {T<:Number} = T

function promote_rule(
  ::Type{T1}, ::Type{T2}
) where {T1<:EmptyStorage{EmptyNumber},T2<:TensorStorage}
  return T2
end
function promote_rule(::Type{T1}, ::Type{T2}) where {T1<:EmptyStorage,T2<:TensorStorage}
  return promote_type(similartype(T2, eltype(T1)), T2)
end

function permutedims!!(R::Tensor, T::EmptyTensor, perm::Tuple, f::Function=(r, t) -> t)
  RR = convert(promote_type(typeof(R), typeof(T)), R)
  RR = permutedims!!(RR, RR, ntuple(identity, Val(ndims(R))), (r, t) -> f(r, false))
  return RR
end

function permutedims!!(R::EmptyTensor, T::Tensor, perm::Tuple, f::Function=(r, t) -> t)
  RR = similar(promote_type(typeof(R), typeof(T)), inds(R))
  RR = permutedims!!(RR, T, perm, (r, t) -> f(false, t))
  return RR
end

function permutedims!!(R::EmptyTensor, T::EmptyTensor, perm::Tuple, f::Function=(r, t) -> t)
  RR = convert(promote_type(typeof(R), typeof(T)), R)
  return RR
end

function show(io::IO, mime::MIME"text/plain", T::EmptyTensor)
  summary(io, T)
  return println(io)
end
