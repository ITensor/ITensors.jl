using NDTensors.TypeParameterAccessors: TypeParameterAccessors, type_parameter, set_type_parameter

function storagemode(object)
  return storagemode(typeof(object))
end
function storagemode(type::Type)
  return type_parameter(type, storagemode)
end

function set_storagemode(type::Type, param)
  return set_type_parameter(type, storagemode, param)
end

function cpu end
