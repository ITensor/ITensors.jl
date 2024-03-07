using NDTensors.TypeParameterAccessors: TypeParameterAccessors, unwrap_array_type, parameter, parenttype, type_parameter
struct Exposed{Unwrapped,Object}
  object::Object
end

expose(object) = Exposed{unwrap_array_type(object),typeof(object)}(object)

unexpose(E::Exposed) = E.object

TypeParameterAccessors.parenttype(type::Type{<:Exposed}) = parameter(type, parenttype)
TypeParameterAccessors.position(::Type{<:Exposed}, ::typeof(parenttype)) = TypeParameterAccessors.Position(1)
TypeParameterAccessors.unwrap_array_type(type::Type{<:Exposed}) = parenttype(type)
TypeParameterAccessors.unwrap_array_type(E::Exposed) = unwrap_array_type(typeof(E))