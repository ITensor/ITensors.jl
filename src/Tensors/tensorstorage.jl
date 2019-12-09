export data,
       TensorStorage,
       randn!

# TODO: define as
# abstract type TensorStorage{El} end <: AbstractVector{El}
abstract type TensorStorage{ElT} end

data(S::TensorStorage) = S.data

Base.@propagate_inbounds Base.getindex(S::TensorStorage,
                                       i::Integer) = getindex(data(S),i)
Base.@propagate_inbounds Base.setindex!(S::TensorStorage,v,
                                        i::Integer) = setindex!(data(S),v,i)

Random.randn!(S::TensorStorage) = randn!(data(S))
Base.fill!(S::TensorStorage,v) = fill!(data(S),v)

Base.convert(::Type{T},D::T) where {T<:TensorStorage} = D

