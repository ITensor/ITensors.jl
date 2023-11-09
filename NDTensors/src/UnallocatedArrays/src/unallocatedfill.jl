## TODO All constructors not fully implemented but so far it matches the 
## constructors found in `FillArrays`. Need to fix io
struct UnallocatedFill{ElT,N,Axes,Alloc<:AbstractArray{ElT,N}} <:
       FillArrays.AbstractFill{ElT,N,Axes}
  f::FillArrays.Fill{ElT,N,Axes}
  ## TODO use `set_parameters` as constructor to these types
end

function set_alloctype(f::Fill, alloc::Type{<:AbstractArray})
  return UnallocatedFill{eltype(f),ndims(f),typeof(axes(f)),alloc}(f)
end

Base.parent(F::UnallocatedFill) = F.f
