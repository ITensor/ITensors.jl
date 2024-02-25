abstract type AbstractTypeParameter end

struct TypeParameter{Param} <: AbstractTypeParameter end
TypeParameter(param) = TypeParameter{param}()
TypeParameter(param::TypeParameter) = param
AbstractTypeParameter(param::AbstractTypeParameter) = param
AbstractTypeParameter(param) = TypeParameter(param)
AbstractTypeParameter(param::TypeVar) = UnspecifiedTypeParameter()
AbstractTypeParameter(type::Type, pos) = AbstractTypeParameter(parameter(type, pos))

struct UnspecifiedTypeParameter <: AbstractTypeParameter end
