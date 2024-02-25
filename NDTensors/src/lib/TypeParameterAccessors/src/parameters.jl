# JULIA_INTERNALS: Warning! This code relies on undocumented
# internal details of the Julia language.
# It could break from version to version of Julia.

# The signature `parameters(::Type{type}) where {type}`
# doesn't work if `type` is a `DataType` with `TypeVar`s.
function _parameters(type::Type)
  return Tuple(Base.unwrap_unionall(type).parameters)
end
@generated function parameters(type_type::Type)
  type = only(type_type.parameters)
  return _parameters(type)
end
parameters(object) = parameters(typeof(object))

nparameters(type_or_object) = length(parameters(type_or_object))

eachposition(type_or_object) = ntuple(Position, Val(nparameters(type_or_object)))
