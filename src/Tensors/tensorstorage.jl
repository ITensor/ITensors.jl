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

Base.convert(::Type{T},D::T) where {T<:TensorStorage} = D

