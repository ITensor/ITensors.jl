using NDTensors.TypeParameterAccessors: TypeParameterAccessors, type_parameter

function storagemode(object)
  return storagemode(typeof(object))
end
function storagemode(type::Type)
  return type_parameter(type, storagemode)
end

function cpu end
