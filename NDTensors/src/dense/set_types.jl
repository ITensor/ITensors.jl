## TODO verify unecessary and delete
# function SetParameters.specify_parameters(
#   storetype::Type{<:Dense{ElT,DataT}}, datatype::Type{<:AbstractArray}=default_datatype(ElT)
# ) where {ElT,DataT<:AbstractArray{ElT}}
#   return storetype
# end
# function SetParameters.specify_parameters(
#   storetype::Type{<:Dense{ElT,DataT}}, datatype::Type{<:AbstractArray}=default_datatype(ElT)
# ) where {ElT,DataT<:AbstractArray}
#   return set_datatype(storetype, specify_parameters(DataT, ElT))
# end
# function SetParameters.specify_parameters(
#   storetype::Type{<:Dense{ElT}}, datatype::Type{<:AbstractArray}=default_datatype(ElT)
# ) where {ElT}
#   return default_storagetype(specify_parameters(datatype, ElT), ())
# end

# function SetParameters.specify_parameters(
#   storetype::Type{<:Dense},
#   datatype::Type{<:AbstractArray}=default_datatype(default_eltype()),
# )
#   return default_storagetype(datatype, ())
# end

function set_datatype(storagetype::Type{<:Dense}, datatype::Type{<:AbstractVector})
  return Dense{eltype(datatype),datatype}
end

function set_datatype(storagetype::Type{<:Dense}, datatype::Type{<:AbstractArray})
  return error(
    "Setting the `datatype` of the storage type `$storagetype` to a $(ndims(datatype))-dimsional array of type `$datatype` is not currently supported, use an `AbstractVector` instead.",
  )
end

## TODO move to UnspecifiedArrays
# function set_eltype(
#   storagetype::Type{<:Dense{ElT,DataT}}, ElR::Type
# ) where {ElT,DataT<:AbstractArray}
#   return Dense{ElR,similartype(DataT, ElR)}
# end
