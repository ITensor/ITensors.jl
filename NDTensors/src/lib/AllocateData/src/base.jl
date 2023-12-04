# TODO: Move to `AllocatedDataBaseExt`.

# `Base.UndefInitializer`
function allocate(arraytype::Type{<:AbstractArray}, ::Base.UndefInitializer, axes::Tuple)
  return allocate(arraytype, undef, axes)
end

# Work around limited `Array` constructors (don't accept `axes`)
function Base.Array{T}(::UndefInitializer, axes::Tuple{Vararg{AbstractUnitRange}}) where {T}
  return Array{T}(Base.undef, to_dim.(axes))
end
