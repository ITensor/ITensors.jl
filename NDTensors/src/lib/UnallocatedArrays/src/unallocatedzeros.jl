## TODO Should Alloc also be of ElT and N or should there be 
## More freedom there?
struct UnallocatedZeros{ElT,N,Axes,Alloc<:AbstractArray{ElT,N}} <:
       FillArrays.AbstractZeros{ElT,N,Axes}
  z::FillArrays.Zeros{ElT,N,Axes}
  ## TODO use `set_parameters` as constructor to these types
end

function set_alloctype(z::Zeros, alloc::Type{<:AbstractArray})
  return UnallocatedZeros{eltype(z),ndims(z),typeof(axes(z)),alloc}(z)
end

Base.parent(Z::UnallocatedZeros) = Z.z

## Here I can't overload ::AbstractFill because this overwrites Base.complex in 
## Fill arrays which creates an infinite loop. Another option would be to write
## UnallocatedArrays.complex(::AbstractFill) which calls Base.complex on the parent
function Base.complex(A::UnallocatedZeros)
  return set_alloctype(
    complex(parent(A)), set_parameter(alloctype(A), Position{1}(), complex(eltype(A)))
  )
end
