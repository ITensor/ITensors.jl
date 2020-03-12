export data,
       TensorStorage,
       randn!,
       scale!

abstract type TensorStorage{ElT} <: AbstractVector{ElT} end

data(S::TensorStorage) = S.data

Base.eltype(::TensorStorage{ElT}) where {ElT} = ElT

Base.eltype(::Type{<:TensorStorage{ElT}}) where {ElT} = ElT

# This is necessary since for some reason inference doesn't work
# with the more general definition (eltype(Nothing) === Any)
Base.eltype(::TensorStorage{Nothing}) = Nothing

Base.length(D::TensorStorage) = length(data(D))

Base.size(D::TensorStorage) = size(data(D))

Base.@propagate_inbounds Base.getindex(S::TensorStorage,
                                       i::Integer) = data(S)[i]
Base.@propagate_inbounds Base.setindex!(S::TensorStorage,v,
                                        i::Integer) = (setindex!(data(S),v,i); S)

# Needed for passing Tensor{T,2} to BLAS/LAPACK
function Base.unsafe_convert(::Type{Ptr{ElT}},
                             T::TensorStorage{ElT}) where {ElT}
  return Base.unsafe_convert(Ptr{ElT},data(T))
end

# This may need to be overloaded, since storage types
# often have other fields besides data
Base.conj(S::T) where {T<:TensorStorage} = T(conj(data(S)))

Base.complex(S::T) where {T<:TensorStorage} = complex(T)(complex(data(S)))

Base.copyto!(S1::TensorStorage,S2::TensorStorage) = (copyto!(data(S1),data(S2)); S1)

Random.randn!(S::TensorStorage) = (randn!(data(S)); S)

Base.fill!(S::TensorStorage,v) = (fill!(data(S),v); S)

LinearAlgebra.rmul!(S::TensorStorage,v::Number) = (rmul!(data(S),v); S)
scale!(S::TensorStorage,v::Number) = rmul!(S,v)

LinearAlgebra.norm(S::TensorStorage) = norm(data(S))

Base.convert(::Type{T},D::T) where {T<:TensorStorage} = D

blockoffsets(D::TensorStorage) = D.blockoffsets

"""
nzblocks(T::TensorStorage)

Return a vector of the non-zero blocks of the BlockSparse storage.
"""
nzblocks(T::TensorStorage) = nzblocks(blockoffsets(T))

nnzblocks(D::TensorStorage) = length(blockoffsets(D))
nnz(D::TensorStorage) = length(D)

offset(D::TensorStorage,block) = offset(blockoffsets(D),block)

block(D::TensorStorage,n::Int) = block(blockoffsets(D),n)



