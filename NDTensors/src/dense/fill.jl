##TODO replace randn in ITensors with generic_randn
## and replace zeros with generic_zeros

# This is a file to write generic fills for NDTensors.
#  This includes random fills, zeros, ...

function generic_randn(
  StoreT::Type{<:Dense{ElT,DataT}}, dim::Integer=0
) where {DataT<:AbstractArray,ElT}
  @assert ElT == eltype(DataT)
  data = generic_randn(DataT, dim)
  StoreT = set_datatype(StoreT, typeof(data))
  return StoreT(data)
end

function generic_randn(StoreT::Type{<:Dense{ElT}}, dim::Integer=0) where {ElT}
  return generic_randn(default_storagetype(ElT), dim)
end

function generic_randn(StoreT::Type{<:Dense}, dim::Integer=0)
  return generic_randn(default_storagetype(), dim)
end

function generic_zeros(
  StoreT::Type{<:Dense{ElT,DataT}}, dim::Integer=0
) where {DataT<:AbstractArray,ElT}
  @assert ElT == eltype(DataT)
  data = generic_zeros(DataT, dim)
  StoreT = set_datatype(StoreT, typeof(data))
  return StoreT(data)
end

function generic_zeros(StoreT::Type{<:Dense{ElT}}, dim::Integer=0) where {ElT}
  return generic_zeros(default_storagetype(ElT), dim)
end

function generic_zeros(StoreT::Type{<:Dense}, dim::Integer=0)
  return generic_zeros(default_storagetype(), dim)
end
