using ..Expose: Exposed, unexpose
using ..Vendored.TypeParameterAccessors: TypeParameterAccessors, type_parameters, set_type_parameters

function storagemode(object)
    return storagemode(typeof(object))
end
function storagemode(type::Type)
    return type_parameters(type, storagemode)
end

function set_storagemode(type::Type, param)
    return set_type_parameters(type, storagemode, param)
end

function cpu end

cpu(E::Exposed) = cpu(unexpose(E))
