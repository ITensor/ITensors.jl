# Allocate undefined memory.
function allocate(arraytype::Type{<:AbstractArray}, ::UndefInitializer, axes::Tuple{Vararg{AbstractUnitRange}})
  # Defaults to `undef` constructor, like `Base.similar`.
  return arraytype(undef, axes)
end

function allocate(arraytype::Type{<:AbstractArray}, initializer::AbstractInitializer, dims::Tuple)
  return allocate(arraytype, initializer, to_axis.(dims))
end

# TODO: Move to `allocate_zeros`.
# Allocate an array filled with zeros.
function allocate(arraytype::Type{<:AbstractArray}, ::ZeroInitializer, axes::Tuple)
  a = allocate(arraytype, axes)
  # TODO: Use `VectorInterface.zerovector!!`.
  a .= zero(eltype(a))
  return a
end

function allocate_zeros(arraytype::Type{<:AbstractArray}, axes::Tuple)
  return allocate(arraytype, zero_init, axes)
end

# Default initializes undefined memory
function allocate(arraytype::Type{<:AbstractArray}, axes::Tuple)
  return allocate(arraytype, default_initializer(arraytype), axes)
end

# Default initializes undefined memory
function allocate(axes::Tuple)
  return allocate(default_arraytype(), axes)
end
