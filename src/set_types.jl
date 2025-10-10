using NDTensors.TypeParameterAccessors: unwrap_array_type
NDTensors.TypeParameterAccessors.parenttype(::ITensor) = typeof(tensor(T))
function NDTensors.TypeParameterAccessors.unwrap_array_type(T::ITensor)
    return unwrap_array_type(tensor(T))
end
