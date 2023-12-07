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

FillArrays.mult_zeros(a::UnallocatedZeros, b, elt, ax) = UnallocatedZeros(Zeros{elt}(ax), alloctype(a))
FillArrays.mult_zeros(a, b::UnallocatedZeros, elt, ax) = mult_zeros(b,a, elt, ax)
function FillArrays.mult_zeros(a::UnallocatedZeros, b::UnallocatedZeros, elt, ax)
  @assert(alloctype(a) == alloctype(b))
  return UnallocatedZeros(Zeros{elt}(ax), alloctype(a))
end

FillArrays.broadcasted_zeros(f, a::UnallocatedZeros, elt, ax) = UnallocatedZeros(Zeros{elt}(ax), alloctype(a))
function FillArrays.broadcasted_zeros(f, a::UnallocatedZeros, b::UnallocatedZeros, elt, ax)
  @assert(alloctype(a) == alloctype(b)) 
  return UnallocatedZeros(Zeros{elt}(ax), alloctype(a))
end

FillArrays.broadcasted_zeros(f, a::UnallocatedZeros, b, elt, ax) = UnallocatedZeros(Zeros{elt}(ax), alloctype(a))
FillArrays.broadcasted_zeros(f, a, b::UnallocatedZeros, elt, ax) = broadcasted_zeros(f, b, a, elt, ax)
# kron_zeros(a, b, elt, ax) = Zeros{elt}(ax)