
#
# Represents a tensor order that could be set to any order.
#

struct EmptyOrder end

#
# Represents a number that can be set to any type.
#

struct EmptyNumber <: Real end

zero(::Type{EmptyNumber}) = EmptyNumber()
zero(n::EmptyNumber) = zero(typeof(n))

# This helps handle a lot of basic algebra, like:
# EmptyNumber() + 2.3 == 2.3
convert(::Type{T}, x::EmptyNumber) where {T<:Number} = T(zero(T))

# TODO: Should this be implemented?
#Complex(x::Real, ::EmptyNumber) = x

# This is to help define `float(::EmptyNumber) = 0.0`.
# This helps with defining `norm` of `EmptyStorage{EmptyNumber}`.
AbstractFloat(::NDTensors.EmptyNumber) = zero(AbstractFloat)

# Basic arithmetic
(::EmptyNumber + ::EmptyNumber) = EmptyNumber()
(::EmptyNumber - ::EmptyNumber) = EmptyNumber()
(::Number * ::EmptyNumber) = EmptyNumber()
(::EmptyNumber * ::Number) = EmptyNumber()
(::EmptyNumber * ::EmptyNumber) = EmptyNumber()
(::EmptyNumber / ::Number) = EmptyNumber()
(::Number / ::EmptyNumber) = throw(DivideError())
(::EmptyNumber / ::EmptyNumber) = throw(DivideError())
-(::EmptyNumber) = EmptyNumber()

# TODO: Delete this.
# This is a backup definition to make:
# A = ITensor(i, j)
# complex!(A)
# work. It acts as if the "default" type is `Float64`
## complex(::Type{EmptyNumber}) = ComplexF64

function similartype(::Type{StoreT}, ::Type{ElT}) where {StoreT<:Dense{EmptyNumber},ElT}
  return Dense{ElT,similartype(datatype(StoreT), ElT)}
end

function similartype(
  ::Type{StoreT}, ::Type{ElT}
) where {StoreT<:BlockSparse{EmptyNumber},ElT}
  return BlockSparse{ElT,similartype(datatype(StoreT), ElT),ndims(StoreT)}
end

#
# Empty storage
#

struct EmptyStorage{ElT,StoreT<:TensorStorage} <: TensorStorage{ElT} end

data(S::EmptyStorage) = NoData()

# Get the EmptyStorage version of the TensorStorage
function emptytype(::Type{StoreT}) where {StoreT}
  return EmptyStorage{eltype(StoreT),StoreT}
end

empty(::Type{StoreT}) where {StoreT} = emptytype(StoreT)()

norm(::EmptyStorage{ElT}) where {ElT} = norm(zero(ElT))

# Defaults to Dense
function EmptyStorage(::Type{ElT}) where {ElT}
  return emptytype(Dense{ElT,Vector{ElT}})()
end

# TODO: should this be `EmptyNumber`?
EmptyStorage() = EmptyStorage(Float64)

similar(S::EmptyStorage) = S
similar(S::EmptyStorage, ::Type{ElT}) where {ElT} = empty(similartype(fulltype(S), ElT))

copy(S::EmptyStorage) = S

size(::EmptyStorage) = (0,)
length(::EmptyStorage) = 0

isempty(::EmptyStorage) = true

nnzblocks(::EmptyStorage) = 0

nnz(::EmptyStorage) = 0

function conj(::AllowAlias, S::EmptyStorage)
  return S
end

# TODO: promote the element type properly
(S::EmptyStorage * x::Number) = S
(x::Number * S::EmptyStorage) = S * x

function Base.real(::Type{<:EmptyStorage{ElT,StoreT}}) where {ElT,StoreT}
  return EmptyStorage{real(ElT),real(StoreT)}
end

Base.real(S::EmptyStorage) = real(typeof(S))()

function complex(::Type{<:EmptyStorage{ElT,StoreT}}) where {ElT,StoreT}
  return EmptyStorage{complex(ElT),complex(StoreT)}
end

complex(S::EmptyStorage) = complex(typeof(S))()

#size(::EmptyStorage) = 0

function show(io::IO, mime::MIME"text/plain", S::EmptyStorage)
  return println(io, typeof(S))
end

#
# EmptyTensor (Tensor using EmptyStorage storage)
#

const EmptyTensor{ElT,N,StoreT,IndsT} =
  Tensor{ElT,N,StoreT,IndsT} where {StoreT<:EmptyStorage}

function emptytype(::Type{TensorT}) where {TensorT<:Tensor}
  return Tensor{
    eltype(TensorT),ndims(TensorT),emptytype(storagetype(TensorT)),indstype(TensorT)
  }
end

# XXX TODO: add bounds checking
getindex(T::EmptyTensor, I::Integer...) = zero(eltype(T))
getindex(T::EmptyTensor{EmptyNumber}, I::Integer...) = EmptyNumber()
function getindex(T::EmptyTensor{Complex{EmptyNumber}}, I::Integer...)
  return Complex(NDTensors.EmptyNumber(), NDTensors.EmptyNumber())
end

similar(T::EmptyTensor, inds::Tuple) = setinds(T, inds)
function similar(T::EmptyTensor, ::Type{ElT}) where {ElT<:Number}
  return tensor(similar(storage(T), ElT), inds(T))
end

function randn!!(T::EmptyTensor)
  Tf = similar(fulltype(T), inds(T))
  randn!(Tf)
  return Tf
end

# Default to Float64
function randn!!(T::EmptyTensor{EmptyNumber})
  return randn!!(similar(T, Float64))
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

fulltype(::Type{EmptyStorage{ElT,StoreT}}) where {ElT,StoreT} = StoreT
fulltype(T::EmptyStorage) = fulltype(typeof(T))

fulltype(T::Tensor) = fulltype(typeof(T))

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

# Version of contraction where output storage is empty
function contract!!(R::EmptyTensor, labelsR, T1::Tensor, labelsT1, T2::Tensor, labelsT2)
  RR = contract(T1, labelsT1, T2, labelsT2, labelsR)
  return RR
end

# When one of the tensors is empty, return an empty
# tensor.
# XXX: make sure `R` is actually correct!
function contract!!(
  R::EmptyTensor, labelsR, T1::EmptyTensor, labelsT1, T2::Tensor, labelsT2
)
  return R
end

# When one of the tensors is empty, return an empty
# tensor.
# XXX: make sure `R` is actually correct!
function contract!!(
  R::EmptyTensor, labelsR, T1::Tensor, labelsT1, T2::EmptyTensor, labelsT2
)
  return R
end

function contract!!(
  R::EmptyTensor, labelsR, T1::EmptyTensor, labelsT1, T2::EmptyTensor, labelsT2
)
  return R
end

# For ambiguity with versions in combiner.jl
function contract!!(
  R::EmptyTensor, labelsR, T1::CombinerTensor, labelsT1, T2::Tensor, labelsT2
)
  RR = contract(T1, labelsT1, T2, labelsT2, labelsR)
  return RR
end

# For ambiguity with versions in combiner.jl
function contract!!(
  R::EmptyTensor, labelsR, T1::Tensor, labelsT1, T2::CombinerTensor, labelsT2
)
  RR = contract(T1, labelsT1, T2, labelsT2, labelsR)
  return RR
end

# For ambiguity with versions in combiner.jl
function contract!!(
  R::EmptyTensor, labelsR, T1::EmptyTensor, labelsT1, T2::CombinerTensor, labelsT2
)
  RR = contraction_output(T1, labelsT1, T2, labelsT2, labelsR)
  return RR
end

promote_rule(::Type{EmptyNumber}, ::Type{T}) where {T<:Number} = T

function promote_rule(
  ::Type{T1}, ::Type{T2}
) where {T1<:EmptyStorage{EmptyNumber},T2<:TensorStorage}
  return T2
end
function promote_rule(::Type{T1}, ::Type{T2}) where {T1<:EmptyStorage,T2<:TensorStorage}
  return promote_type(similartype(T2, eltype(T1)), T2)
end

function contraction_output(T1::EmptyTensor, T2::EmptyTensor, indsR::Tuple)
  fulltypeR = contraction_output_type(fulltype(T1), fulltype(T2), indsR)
  storagetypeR = storagetype(fulltypeR)
  emptystoragetypeR = emptytype(storagetypeR)
  return Tensor(emptystoragetypeR(), indsR)
end

function contraction_output(T1::Tensor, T2::EmptyTensor, indsR)
  fulltypeR = contraction_output_type(typeof(T1), fulltype(T2), indsR)
  storagetypeR = storagetype(fulltypeR)
  emptystoragetypeR = emptytype(storagetypeR)
  return Tensor(emptystoragetypeR(), indsR)
end

function contraction_output(T1::EmptyTensor, T2::Tensor, indsR)
  fulltypeR = contraction_output_type(fulltype(T1), typeof(T2), indsR)
  storagetypeR = storagetype(fulltypeR)
  emptystoragetypeR = emptytype(storagetypeR)
  return Tensor(emptystoragetypeR(), indsR)
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

# XXX: this seems a bit strange and fragile?
# Takes the type very literally.
function HDF5.read(
  parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{StoreT}
) where {StoreT<:EmptyStorage}
  g = open_group(parent, name)
  typestr = string(StoreT)
  if read(attributes(g)["type"]) != typestr
    error("HDF5 group or file does not contain $typestr data")
  end
  return StoreT()
end

function HDF5.write(
  parent::Union{HDF5.File,HDF5.Group}, name::String, ::StoreT
) where {StoreT<:EmptyStorage}
  g = create_group(parent, name)
  attributes(g)["type"] = string(StoreT)
  return attributes(g)["version"] = 1
end
