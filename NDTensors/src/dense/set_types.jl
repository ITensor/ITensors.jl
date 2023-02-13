function set_properties_if_unspecified(
  storetype::Type{<:Dense{ElT,DataT}}, datatype::Type{<:AbstractArray}=default_datatype(ElT)
) where {ElT,DataT<:AbstractArray{ElT}}
  return storetype
end
function set_properties_if_unspecified(
  storetype::Type{<:Dense{ElT,DataT}}, datatype::Type{<:AbstractArray}=default_datatype(ElT)
) where {ElT,DataT<:AbstractArray}
  return set_datatype(storetype, set_properties_if_unspecified(DataT, ElT))
end
function set_properties_if_unspecified(
  storetype::Type{<:Dense{ElT}}, datatype::Type{<:AbstractArray}=default_datatype(ElT)
) where {ElT}
  return default_storagetype(set_properties_if_unspecified(datatype, ElT), ())
end

function set_properties_if_unspecified(
  storetype::Type{<:Dense},
  datatype::Type{<:AbstractArray}=default_datatype(default_eltype()),
)
  return default_storagetype(datatype, ())
end
