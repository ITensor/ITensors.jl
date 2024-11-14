using SparseArrays: SparseArrays
using .TypeParameterAccessors: TypeParameterAccessors, set_eltype, similartype

#
# Represents a tensor order that could be set to any order.
#

struct EmptyOrder end

function TypeParameterAccessors.similartype(
  StoreT::Type{<:TensorStorage{EmptyNumber}}, ElT::Type
)
  return set_eltype(StoreT, ElT)
end

function TypeParameterAccessors.similartype(
  StoreT::Type{<:TensorStorage{EmptyNumber}}, DataT::Type{<:AbstractArray}
)
  return set_datatype(StoreT, DataT)
end

## TODO fix this similartype to use set eltype for BlockSparse
function TypeParameterAccessors.similartype(
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

storagetype(::Type{EmptyStorage{ElT,StoreT}}) where {ElT,StoreT} = StoreT
storagetype(::EmptyStorage{ElT,StoreT}) where {ElT,StoreT} = StoreT

# Get the EmptyStorage version of the TensorStorage
function emptytype(storagetype::Type{<:TensorStorage})
  return EmptyStorage{eltype(storagetype),storagetype}
end

empty(storagetype::Type{<:TensorStorage}) = emptytype(storagetype)()

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

SparseArrays.nnz(::EmptyStorage) = 0

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

blockoffsets(storage::EmptyStorage) = BlockOffsets{ndims(storage)}()

function Base.show(io::IO, mime::MIME"text/plain", S::EmptyStorage)
  return println(io, typeof(S))
end

using .TypeParameterAccessors: TypeParameterAccessors
TypeParameterAccessors.parenttype(empty::Type{<:EmptyStorage}) = storagetype(empty)
zero(empty::EmptyStorage) = empty
