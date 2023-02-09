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

function generic_randn(DataT::Type{<:AbstractArray}, dim::Integer=0)
  DataT = set_eltype_if_unspecified(DataT)
  data = similar(DataT, dim)
  ElT = eltype(DataT)
  for i in 1:length(data)
    data[i] = randn(ElT)
  end
  return data
end

##TODO replace randn in ITensors with generic_randn
## and replace zeros with generic_zeros
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

function generic_zeros(DataT::Type{<:AbstractArray}, dim::Integer=0)
  DataT = set_eltype_if_unspecified(DataT)
  ElT = eltype(DataT)
  return fill!(similar(DataT, dim), zero(ElT))
end
