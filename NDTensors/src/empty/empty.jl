
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

function similartype(StoreT::Type{<:TensorStorage{EmptyNumber}}, ElT::Type)
  return set_eltype(StoreT, ElT)
end

function similartype(
  StoreT::Type{<:TensorStorage{EmptyNumber}}, DataT::Type{<:AbstractArray}
)
  return set_datatype(StoreT, DataT)
end

## TODO fix this similartype to use set eltype for BlockSparse
function similartype(
  ::Type{StoreT}, ::Type{ElT}
) where {StoreT<:BlockSparse{EmptyNumber},ElT}
  return BlockSparse{ElT,similartype(datatype(StoreT), ElT),ndims(StoreT)}
end

#
# Empty storage
#

struct EmptyStorage{ElT,StoreT<:TensorStorage} <: TensorStorage{ElT} end

function EmptyStorage(::Type{ElT}) where {ElT}
  return empty(default_storagetype(default_datatype(ElT)))
  #return emptytype(Dense{ElT,Vector{ElT}})()
end

# TODO: should this be `EmptyNumber`?
EmptyStorage() = EmptyStorage(default_eltype())

# Get the EmptyStorage version of the TensorStorage
function emptytype(::Type{StoreT}) where {StoreT}
  return EmptyStorage{eltype(StoreT),StoreT}
end

empty(::Type{StoreT}) where {StoreT} = emptytype(StoreT)()

data(S::EmptyStorage) = NoData()

## TODO Why is the norm of an empty tensor 0???
norm(::EmptyStorage{ElT}) where {ElT} = norm(zero(ElT))#EmptyNumber

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

function complex(::Type{<:EmptyStorage{ElT,StoreT}}) where {ElT,StoreT}
  return EmptyStorage{complex(ElT),complex(StoreT)}
end

real(S::EmptyStorage) = real(typeof(S))()

complex(S::EmptyStorage) = complex(typeof(S))()

function show(io::IO, mime::MIME"text/plain", S::EmptyStorage)
  return println(io, typeof(S))
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
