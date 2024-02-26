abstract type AbstractTypeParameter end
AbstractTypeParameter(param::AbstractTypeParameter) = param
wrapped_type_parameter(param) = AbstractTypeParameter(param)
wrapped_type_parameter(type::Type, pos) = AbstractTypeParameter(parameter(type, pos))

struct TypeParameter{Param} <: AbstractTypeParameter end
TypeParameter(param) = TypeParameter{param}()
TypeParameter(param::TypeParameter) = param
AbstractTypeParameter(param) = TypeParameter(param)

struct UnspecifiedTypeParameter <: AbstractTypeParameter end
AbstractTypeParameter(param::TypeVar) = UnspecifiedTypeParameter()
