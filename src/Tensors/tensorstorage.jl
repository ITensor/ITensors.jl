export data,
       TensorStorage,
       randn!

abstract type TensorStorage{ElT} <: AbstractVector{ElT} end

data(S::TensorStorage) = S.data

Base.@propagate_inbounds Base.getindex(S::TensorStorage,
                                       i::Integer) = getindex(data(S),i)
Base.@propagate_inbounds Base.setindex!(S::TensorStorage,v,
                                        i::Integer) = setindex!(data(S),v,i)

Random.randn!(S::TensorStorage) = randn!(data(S))

Base.fill!(S::TensorStorage,v) = fill!(data(S),v)

scale!(S::TensorStorage,v) = scale!(data(S),v)

LinearAlgebra.norm(S::TensorStorage) = norm(data(S))

Base.convert(::Type{T},D::T) where {T<:TensorStorage} = D

blockoffsets(D::TensorStorage) = D.blockoffsets

"""
nzblocks(T::TensorStorage)

Return a vector of the non-zero blocks of the BlockSparse storage.
"""
nzblocks(T::TensorStorage) = nzblocks(blockoffsets(T))

nnzblocks(D::TensorStorage) = length(blockoffsets(D))
Base.length(D::TensorStorage) = length(data(D))
Base.size(D::TensorStorage) = (length(D),)
nnz(D::TensorStorage) = length(D)

offset(D::TensorStorage,block) = offset(blockoffsets(D),block)

block(D::TensorStorage,n::Int) = block(blockoffsets(D),n)



