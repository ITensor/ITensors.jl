# This is a file to write generic fills for NDTensors.
#  This includes random fills, zeros, ...

function generic_randn(
  StoreT::Type{<:Dense{ElT,DataT}}, dim::Integer=0
) where {DataT<:AbstractArray{ElT}} where {ElT}
  data = generic_randn(DataT, dim)
  return StoreT(data)
end

function generic_randn(StoreT::Type{<:Dense}, dim::Integer=0)
  return generic_randn(StoreT{default_eltype(),default_datatype(default_eltype())}, dim)
end

function generic_randn(DataT::Type{<:AbstractArray{ElT}}, dim::Integer=0) where {ElT}
  data = DataT(undef, dim)
  for i in 1:dim
    @inbounds data[i] = randn(ElT)
  end
  return data
end

function generic_randn(DataT::Type{<:AbstractArray}, dim::Integer=0)
  return generic_randn(DataT{default_eltype()}, dim)
end

##TODO replace randn in ITensors with generic_randn
## and replace zeros with generic_zeros
function randn(
  StoreT::Type{<:Dense{ElT,VecT}}, dim::Integer
) where {VecT<:AbstractArray{ElT}} where {ElT<:Number}
  return StoreT(randn(ElT, dim))
end

function generic_zeros(
  StoreT::Type{<:Dense{Any,DataT}}, dim::Integer=0
) where {DataT<:AbstractArray}
  data = generic_zeros(DataT, dim)
  return StoreT(data)
end

function generic_zeros(StoreT::Type{<:Dense}, dim::Integer=0)
  return generic_zeros(StoreT{default_eltype(),default_datatype(default_eltype())}, dim)
end

function generic_zeros(DataT::Type{<:AbstractArray{ElT}}, dim::Integer=0) where {ElT}
  return fill!(DataT(undef, dim), zero(ElT))
end

function generic_zeros(DataT::Type{<:AbstractArray}, dim::Integer=0)
  return generic_zeros(DataT{default_eltype()}, dim)
end

