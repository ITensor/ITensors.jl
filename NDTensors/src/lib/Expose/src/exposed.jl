using TypeParameterAccessors:
  TypeParameterAccessors, unwrap_array_type, parenttype, type_parameters
struct Exposed{Unwrapped,Object}
  object::Object
end

expose(object) = Exposed{unwrap_array_type(object),typeof(object)}(object)

# This is a corner case that handles the fact that by convention,
# the storage of a uniform diagonaly tensor in NDTensors.jl is a number.
expose(object::Number) = Exposed{typeof(object),typeof(object)}(object)

unexpose(E::Exposed) = E.object

## TODO remove TypeParameterAccessors when SetParameters is removed
TypeParameterAccessors.parenttype(type::Type{<:Exposed}) = type_parameters(type, parenttype)
function TypeParameterAccessors.position(::Type{<:Exposed}, ::typeof(parenttype))
  return TypeParameterAccessors.Position(1)
end
TypeParameterAccessors.unwrap_array_type(type::Type{<:Exposed}) = parenttype(type)
TypeParameterAccessors.unwrap_array_type(E::Exposed) = unwrap_array_type(typeof(E))
