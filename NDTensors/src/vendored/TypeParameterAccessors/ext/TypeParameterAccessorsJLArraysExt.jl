module TypeParameterAccessorsJLArraysExt

using JLArrays: JLArray
using NDTensors.Vendored.TypeParameterAccessors: TypeParameterAccessors, Position

TypeParameterAccessors.position(::Type{<:JLArray}, ::typeof(eltype)) = Position(1)
TypeParameterAccessors.position(::Type{<:JLArray}, ::typeof(ndims)) = Position(2)
TypeParameterAccessors.default_type_parameters(::Type{<:JLArray}) = (Float64, 1)

end
