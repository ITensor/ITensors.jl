using TypeParameterAccessors:
  default_type_parameter,
  parenttype,
  set_eltype,
  specify_default_type_parameters,
  type_parameter
##TODO replace randn in ITensors with generic_randn
## and replace zeros with generic_zeros

# This is a file to write generic fills for NDTensors.
#  This includes random fills, zeros, ...

function generic_randn(StoreT::Type{<:Dense}, dims::Integer; rng=Random.default_rng())
  StoreT = specify_default_type_parameters(StoreT)
  DataT = specify_type_parameter(type_parameter(StoreT, parenttype), eltype, eltype(StoreT))
  @assert eltype(StoreT) == eltype(DataT)

  data = generic_randn(DataT, dims; rng=rng)
  StoreT = set_datatype(StoreT, typeof(data))
  return StoreT(data)
end

function generic_zeros(StoreT::Type{<:Dense}, dims::Integer)
  StoreT = specify_default_type_parameters(StoreT)
  DataT = specify_type_parameter(type_parameter(StoreT, parenttype), eltype, eltype(StoreT))
  @assert eltype(StoreT) == eltype(DataT)

  data = generic_zeros(DataT, dims)
  StoreT = set_datatype(StoreT, typeof(data))
  return StoreT(data)
end
