## TODO Should Alloc also be of ElT and N or should there be 
## More freedom there?
struct UnallocatedZeros{ElT,N,Axes,Alloc<:AbstractArray{ElT,N}} <:
       FillArrays.AbstractZeros{ElT,N,Axes} <: AbstractUnallocatedArray
  z::FillArrays.Zeros{ElT,N,Axes}
  ## TODO use `set_parameters` as constructor to these types
end

function set_alloctype(z::Zeros, alloc::Type{<:AbstractArray})
  return UnallocatedZeros{eltype(z),ndims(z),typeof(axes(z)),alloc}(z)
end

Base.parent(Z::UnallocatedZeros) = Z.z
