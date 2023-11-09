## TODO All constructors not fully implemented but so far it matches the 
## constructors found in `FillArrays`. Need to fix io
struct UnallocatedFill{ElT,N,Axes,Alloc<:AbstractArray{ElT, N}} <:
       FillArrays.AbstractFill{ElT,N,Axes}
  f::FillArrays.Fill{ElT,N,Axes}
  ## TODO use `set_parameters` as constructor to these types
end

Base.parent(F::UnallocatedFill) = F.f
