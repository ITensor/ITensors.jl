using ..Vendored.TypeParameterAccessors:
    TypeParameterAccessors, unwrap_array_type, parenttype, type_parameters
struct Exposed{Unwrapped, Object}
    object::Object
end

expose(object) = Exposed{unwrap_array_type(object), typeof(object)}(object)

unexpose(E::Exposed) = E.object

## TODO remove TypeParameterAccessors when SetParameters is removed
TypeParameterAccessors.parenttype(type::Type{<:Exposed}) = type_parameters(type, parenttype)
function TypeParameterAccessors.position(::Type{<:Exposed}, ::typeof(parenttype))
    return TypeParameterAccessors.Position(1)
end
TypeParameterAccessors.unwrap_array_type(type::Type{<:Exposed}) = parenttype(type)
TypeParameterAccessors.unwrap_array_type(E::Exposed) = unwrap_array_type(typeof(E))
