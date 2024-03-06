TypeParameterAccessors.position(::Type{<:DiagBlockSparse}, ::typeof(eltype)) = Position(1)
function TypeParameterAccessors.position(::Type{<:DiagBlockSparse}, ::typeof(parenttype))
  return Position(2)
end
function TypeParameterAccessors.position(::Type{<:DiagBlockSparse}, ::typeof(Base.ndims))
  return Position(3)
end
