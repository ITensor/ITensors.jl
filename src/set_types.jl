using NDTensors.TypeParameterAccessors: unwrap_array_type
NDTensors.TypeParameterAccessors.parenttype(::ITensor) = typeof(tensor(T))
NDTensors.TypeParameterAccessors.unwrap_array_type(T::ITensor) = unwrap_array_type(tensor(T))