using FillArrays: FillArrays, getindex_value
using NDTensors.SetParameters: set_parameters
@inline Base.axes(A::Union{<:UnallocatedFill,<:UnallocatedZeros}) = axes(parent(A))
Base.size(A::Union{<:UnallocatedFill,<:UnallocatedZeros}) = size(parent(A))
function FillArrays.getindex_value(A::Union{<:UnallocatedFill,<:UnallocatedZeros})
  return getindex_value(parent(A))
end

function Base.complex(A::Union{<:UnallocatedFill,<:UnallocatedZeros})
  return set_alloctype(
    complex(parent(A)), set_parameters(alloctype(A), Position{1}(), complex(eltype(A)))
  )
end

function Base.transpose(a::Union{<:UnallocatedFill,<:UnallocatedZeros})
  return set_alloctype(transpose(parent(a)), alloctype(a))
end

function Base.adjoint(a::Union{<:UnallocatedFill,<:UnallocatedZeros})
  return set_alloctype(adjoint(parent(a)), alloctype(a))
end

## TODO fix this because reshape loses alloctype
#FillArrays.reshape(a::Union{<:UnallocatedFill, <:UnallocatedZeros}, dims) = set_alloctype(reshape(parent(a), dims), allocate(a))

# function Adapt.adapt_storage(to::Type{<:AbstractArray}, x::Union{<:UnallocatedFill, <:UnallocatedZeros})
#   return set_alloctype(parent(x), to)
# end

# function Adapt.adapt_storage(to::Type{<:Number}, x::)
