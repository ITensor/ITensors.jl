using LinearAlgebra: Adjoint, Diagonal, Hermitian, LowerTriangular, Symmetric, Transpose,
    UnitLowerTriangular, UnitUpperTriangular, UpperTriangular

for wrapper in [
        :Transpose,
        :Adjoint,
        :Symmetric,
        :Hermitian,
        :UpperTriangular,
        :LowerTriangular,
        :UnitUpperTriangular,
        :UnitLowerTriangular,
        :Diagonal,
    ]
    @eval position(::Type{<:$wrapper}, ::typeof(eltype)) = Position(1)
    @eval position(::Type{<:$wrapper}, ::typeof(parenttype)) = Position(2)
end
