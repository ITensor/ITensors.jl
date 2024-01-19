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
