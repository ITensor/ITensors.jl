# This is a file to write generic fills for NDTensors.
#  This includes random fills, zeros, ...

function generic_randn(
  StoreT::Type{<:Dense{<:Number,DataT}}, dim::Integer=0
) where {DataT<:AbstractArray}
  ElT = eltype(StoreT)
  ElTD = eltype(DataT);
  if ElTD != Any && ElTD != ElT
    println("Warning, Element provided to Dense does not match the datatype. Defaulting to Dense eltype.")
  end
  typedDataT = set_eltype_if_unspecified(DataT, ElT)
  data = generic_randn(typedDataT, dim)
  StoreT = NDTensors.similartype(StoreT, ElT)
  return StoreT(data)
end

function generic_randn(StoreT::Type{<:Dense{ElT}}, dim::Integer = 0) where {ElT}
  return generic_randn(default_storagetype(ElT), dim)
end

function generic_randn(StoreT::Type{<:Dense}, dim::Integer=0)
  return generic_randn(StoreT{default_eltype(),default_datatype(default_eltype())}, dim)
end

function generic_randn(DataT::Type{<:AbstractArray}, dim::Integer=0)
  DataT = NDTensors.set_eltype_if_unspecified(DataT)
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
  StoreT::Type{<:Dense{<:Number,DataT}}, dim::Integer=0
) where {DataT<:AbstractArray}
  ElT = eltype(StoreT)
  ElTD = eltype(DataT);
  if ElTD != Any && ElTD != ElT
    println("Warning, Element provided to Dense does not match the datatype. Defaulting to Dense eltype.")
  end
  typedDataT = set_eltype_if_unspecified(DataT, ElT)
  data = generic_zeros(typedDataT, dim)
  StoreT = NDTensors.similartype(StoreT, ElT)
  return StoreT(data)
end

function generic_zeros(StoreT::Type{<:Dense{ElT}}, dim::Integer = 0) where ElT
  @show default_storagetype(eltype(StoreT))
  return generic_zeros(default_storagetype(eltype(StoreT)), dim)
end

function generic_zeros(StoreT::Type{<:Dense}, dim::Integer=0)
  return generic_zeros(default_storagetype(), dim)
end

function generic_zeros(DataT::Type{<:AbstractArray}, dim::Integer=0)
  DataT = set_eltype_if_unspecified(DataT)
  ElT = eltype(DataT)
  return fill!(similar(DataT, dim), zero(ElT))
end

