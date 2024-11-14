using SparseArrays: SparseArrays

abstract type TensorStorage{ElT} <: AbstractVector{ElT} end

data(S::TensorStorage) = S.data

datatype(::Type{<:TensorStorage}) = error("Not implemented")

datatype(S::TensorStorage) = typeof(data(S))

Base.eltype(::TensorStorage{ElT}) where {ElT} = ElT
scalartype(T::TensorStorage) = eltype(T)

Base.eltype(::Type{<:TensorStorage{ElT}}) where {ElT} = ElT

Base.iterate(S::TensorStorage, args...) = iterate(data(S), args...)

dense(S::TensorStorage) = S

# This is necessary since for some reason inference doesn't work
# with the more general definition (eltype(Nothing) === Any)
Base.eltype(::TensorStorage{Nothing}) = Nothing

Base.length(S::TensorStorage) = length(data(S))

Base.size(S::TensorStorage) = size(data(S))

Base.@propagate_inbounds Base.getindex(S::TensorStorage, i::Integer) = data(S)[i]
Base.@propagate_inbounds function Base.setindex!(S::TensorStorage, v, i::Integer)
  return (setindex!(data(S), v, i); S)
end

(S::TensorStorage * x::Number) = setdata(S, x * data(S))
(x::Number * S::TensorStorage) = S * x
(S::TensorStorage / x::Number) = setdata(S, data(S) / x)

-(S::TensorStorage) = setdata(S, -data(S))

# Needed for passing Tensor{T,2} to BLAS/LAPACK
function Base.unsafe_convert(::Type{Ptr{ElT}}, T::TensorStorage{ElT}) where {ElT}
  return Base.unsafe_convert(Ptr{ElT}, data(T))
end

# This may need to be overloaded, since storage types
# often have other fields besides data

Base.conj!(S::TensorStorage) = (conj!(data(S)); return S)

Base.conj(S::TensorStorage) = conj(AllowAlias(), S)

function Base.conj(::AllowAlias, S::TensorStorage)
  return setdata(S, conj(data(S)))
end

function Base.conj(::NeverAlias, S::TensorStorage)
  return conj!(copy(S))
end

Base.complex(S::TensorStorage) = setdata(S, complex(data(S)))

Base.real(S::TensorStorage) = setdata(S, real(data(S)))

Base.imag(S::TensorStorage) = setdata(S, imag(data(S)))

function copyto!(S1::TensorStorage, S2::TensorStorage)
  copyto!(expose(data(S1)), expose(data(S2)))
  return S1
end

Random.randn!(S::TensorStorage) = randn!(Random.default_rng(), S)
Random.randn!(rng::AbstractRNG, S::TensorStorage) = (randn!(rng, data(S)); S)

function Base.map(f, t1::TensorStorage, t_tail::TensorStorage...; kwargs...)
  return setdata(t1, map(f, data(t1), data.(t_tail)...; kwargs...))
end

function Base.mapreduce(f, op, t1::TensorStorage, t_tail::TensorStorage...; kwargs...)
  return mapreduce(f, op, data(t1), data.(t_tail)...; kwargs...)
end

Base.fill!(S::TensorStorage, v) = (fill!(data(S), v); S)

LinearAlgebra.rmul!(S::TensorStorage, v::Number) = (rmul!(data(S), v); S)

scale!(S::TensorStorage, v::Number) = rmul!(S, v)

norm(S::TensorStorage) = norm(data(S))

Base.convert(::Type{T}, S::T) where {T<:TensorStorage} = S

blockoffsets(S::TensorStorage) = S.blockoffsets

"""
nzblocks(T::TensorStorage)

Return a vector of the non-zero blocks of the BlockSparse storage.
"""
nzblocks(T::TensorStorage) = nzblocks(blockoffsets(T))

eachnzblock(T::TensorStorage) = eachnzblock(blockoffsets(T))

nnzblocks(S::TensorStorage) = length(blockoffsets(S))
SparseArrays.nnz(S::TensorStorage) = length(S)

offset(S::TensorStorage, block) = offset(blockoffsets(S), block)
