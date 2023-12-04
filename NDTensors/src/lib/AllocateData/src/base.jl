# TODO: Move to `AllocatedDataBaseExt`.

function Base.Array{T}(::UndefInitializer, axes::Tuple) where {T}
  return Array{T}(Base.undef, to_dim.(axes))
end

# Work around limited Base Array constructors.
function allocate(arraytype::Type{<:AbstractArray}, ::Base.UndefInitializer, axes::Tuple)
  return allocate(arraytype, undef, axes)
end

function allocate(arraytype::Type{<:AbstractArray}, ::Base.UndefInitializer, axes::Tuple)
  return allocate(arraytype, undef, axes)
end
