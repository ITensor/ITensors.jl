using FillArrays: FillArrays, getindex_value
using NDTensors.SetParameters: set_parameters
using Adapt: adapt

const UnallocatedArray{ElT, N, AxesT, AllocT} = Union{<:UnallocatedFill{ElT, N, AxesT, AllocT},<:UnallocatedZeros{ElT, N, AxesT, AllocT}}

@inline Base.axes(A::UnallocatedArray) = axes(parent(A))
Base.size(A::UnallocatedArray) = size(parent(A))
function FillArrays.getindex_value(A::UnallocatedArray)
  return getindex_value(parent(A))
end

function Base.complex(A::UnallocatedArray)
  return set_alloctype(
    complex(parent(A)), set_parameters(alloctype(A), Position{1}(), complex(eltype(A)))
  )
end

function Base.transpose(a::UnallocatedArray)
  return set_alloctype(transpose(parent(a)), alloctype(a))
end

function Base.adjoint(a::UnallocatedArray)
  return set_alloctype(adjoint(parent(a)), alloctype(a))
end

function Base.similar(a::UnallocatedArray)
  return alloctype(a)(undef, size(a))
end

function set_alloctype(T::Type{<:UnallocatedArray}, alloc::Type{<:AbstractArray})
  return set_parameters(T, Position{4}(), alloc)
end

for STYPE in (:AbstractArray, :AbstractFill)
  @eval begin
      @inline $STYPE{T}(F::UnallocatedArray{T}) where T = F
      @inline $STYPE{T,N}(F::UnallocatedArray{T,N}) where {T,N} = F
  end
end

## TODO fix this because reshape loses alloctype
#FillArrays.reshape(a::Union{<:UnallocatedFill, <:UnallocatedZeros}, dims) = set_alloctype(reshape(parent(a), dims), allocate(a))

# function Adapt.adapt_storage(to::Type{<:AbstractArray}, x::Union{<:UnallocatedFill, <:UnallocatedZeros})
#   return set_alloctype(parent(x), to)
# end

# function Adapt.adapt_storage(to::Type{<:Number}, x::)
