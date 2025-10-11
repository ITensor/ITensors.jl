module TypeParameterAccessorsFillArraysExt

using NDTensors.Vendored.TypeParameterAccessors: TypeParameterAccessors, Position
using FillArrays: Fill, Zeros, Ones

for T in (:Fill, :Zeros, :Ones)
    @eval begin
        TypeParameterAccessors.position(::Type{<:$T}, ::typeof(eltype)) = Position(1)
        TypeParameterAccessors.position(::Type{<:$T}, ::typeof(ndims)) = Position(2)
        TypeParameterAccessors.position(::Type{<:$T}, ::typeof(axes)) = Position(3)
        TypeParameterAccessors.default_type_parameters(::Type{<:$T}) = (
            Float64, 1, Tuple{Base.OneTo{Int}},
        )
    end
end

end
