module TypeParameterAccessorsStridedViewsExt

using NDTensors.Vendored.TypeParameterAccessors: Position, TypeParameterAccessors, parenttype
using StridedViews: StridedView

TypeParameterAccessors.position(::Type{<:StridedView}, ::typeof(eltype)) = Position(1)
TypeParameterAccessors.position(::Type{<:StridedView}, ::typeof(ndims)) = Position(2)
function TypeParameterAccessors.position(::Type{<:StridedView}, ::typeof(parenttype))
    return Position(3)
end
function TypeParameterAccessors.default_type_parameters(::Type{<:StridedView})
    return (Float64, 1, Vector{Float64}, typeof(identity))
end

end
