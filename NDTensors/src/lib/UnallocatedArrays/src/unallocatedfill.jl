## TODO All constructors not fully implemented but so far it matches the 
## constructors found in `FillArrays`. Need to fix io
struct UnallocatedFill{ElT,N,Axes,Alloc<:AbstractArray{ElT,N}} <: AbstractFill{ElT,N,Axes}
  f::Fill{ElT,N,Axes}
end

## TODO use `set_parameters` as constructor to these types
function UnallocatedFill(f::Fill, alloc::Type{<:AbstractArray})
  return set_alloctype(
    set_axes(set_ndims(set_eltype(UnallocatedFill, eltype(f)), ndims(f)), typeof(axes(f))),
    alloc,
  )(
    f
  )
end

function set_alloctype(T::Type{<:UnallocatedFill}, alloc::Type{<:AbstractArray})
  return set_parameters(T, Position{4}(), alloc)
end

set_alloctype(f::Fill, alloc::Type{<:AbstractArray}) = UnallocatedFill(f, alloc)

Base.parent(F::UnallocatedFill) = F.f

## These functions are the same for UnallocatedX
function Base.complex(A::UnallocatedFill)
  return set_alloctype(
    complex(parent(A)), set_parameters(alloctype(A), Position{1}(), complex(eltype(A)))
  )
end

@inline Base.axes(A::UnallocatedFill) = axes(parent(A))
Base.size(A::UnallocatedFill) = size(parent(A))
function FillArrays.getindex_value(A::UnallocatedFill)
  return getindex_value(parent(A))
end
Base.copy(A::UnallocatedFill) = A