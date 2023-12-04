# Convert dimension to axis
to_axis(dim::Int) = Base.OneTo(dim)
to_axis(dim::Integer) = to_axis(Int(dim))

function to_dim(axis::AbstractUnitRange)
  # Assume 1-based indexing
  @assert isone(first(axis))
  return length(axis)
end

# TODO: Move to `AllocateDataBlockArraysExt`.
using BlockArrays: blockedrange

# Blocked dimension/axis
to_axis(dim::Vector{Int}) = blockedrange(dim)
to_axis(dim::Vector{<:Integer}) = blockedrange(Int.(dim))
