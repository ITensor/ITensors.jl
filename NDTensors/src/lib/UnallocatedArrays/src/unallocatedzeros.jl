## TODO Should Alloc also be of ElT and N or should there be 
## More freedom there?
struct UnallocatedZeros{ElT,N,Axes,Alloc<:AbstractArray{ElT,N}} <: AbstractZeros{ElT,N,Axes}
  z::Zeros{ElT,N,Axes}
end

function UnallocatedZeros(f::Zeros, alloc::Type{<:AbstractArray})
  return set_alloctype(
    set_axes(set_ndims(set_eltype(UnallocatedZeros, eltype(f)), ndims(f)), typeof(axes(f))),
    alloc,
  )(
    f
  )
end

function set_alloctype(T::Type{<:UnallocatedZeros}, alloc::Type{<:AbstractArray})
  return set_parameters(T, Position{4}(), alloc)
end

set_alloctype(f::Zeros, alloc::Type{<:AbstractArray}) = UnallocatedZeros(f, alloc)

Base.parent(Z::UnallocatedZeros) = Z.z
