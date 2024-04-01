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

function generic_zeros(StoreT::Type{<:Dense}, dim::Integer)
  return generic_zeros(StoreT, (dim,))
end

using .TypeParameterAccessors:
  default_type_parameter,
  parenttype,
  set_eltype,
  specify_default_type_parameters,
  type_parameter
function generic_zeros(StoreT::Type{<:Dense}, dims::Tuple{Integer})
  StoreT = specify_default_type_parameters(StoreT)
  DataT = specify_type_parameter(type_parameter(StoreT, parenttype), eltype, eltype(StoreT))
  @assert eltype(StoreT) == eltype(DataT)

  data = generic_zeros(DataT, dim(dims))
  StoreT = set_datatype(StoreT, typeof(data))
  return StoreT(data)
end

function generic_zeros(::Type{<:Dense}, ::Tuple{Integer,Integer,Vararg{Integer}})
  return error("Can't make a multidimensional `Dense` object.")
end
