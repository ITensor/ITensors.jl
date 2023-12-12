using FillArrays: FillArrays, AbstractZeros, Zeros, broadcasted_zeros, kron_fill, kron_zeros, mult_zeros
using NDTensors.SetParameters: Position, set_parameters
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

Base.convert(::Type{<:UnallocatedZeros}, A::UnallocatedZeros) = A

#############################################
# Arithmatic

function FillArrays.mult_zeros(a::UnallocatedZeros, b, elt, ax)
  return UnallocatedZeros(Zeros{elt}(ax), alloctype(a))
end
FillArrays.mult_zeros(a, b::UnallocatedZeros, elt, ax) = mult_zeros(b, a, elt, ax)
function FillArrays.mult_zeros(a::UnallocatedZeros, b::UnallocatedZeros, elt, ax)
  @assert(alloctype(a) == alloctype(b))
  return UnallocatedZeros(Zeros{elt}(ax), alloctype(a))
end

function FillArrays.broadcasted_zeros(f, a::UnallocatedZeros, elt, ax)
  return UnallocatedZeros(Zeros{elt}(ax), alloctype(a))
end
function FillArrays.broadcasted_zeros(f, a::UnallocatedZeros, b::UnallocatedZeros, elt, ax)
  @assert(alloctype(a) == alloctype(b))
  return UnallocatedZeros(Zeros{elt}(ax), alloctype(a))
end

function FillArrays.broadcasted_zeros(f, a::UnallocatedZeros, b, elt, ax)
  return UnallocatedZeros(Zeros{elt}(ax), alloctype(a))
end
function FillArrays.broadcasted_zeros(f, a, b::UnallocatedZeros, elt, ax)
  return broadcasted_zeros(f, b, a, elt, ax)
end

function FillArrays.kron_zeros(a::UnallocatedZeros, b::UnallocatedZeros, elt, ax)
  @assert alloctype(a) == alloctype(b)
  return UnallocatedZeros(Zeros{elt}(ax), alloctype(a))
end

function FillArrays.kron_fill(a::UnallocatedZeros, b::UnallocatedFill, val, ax)
  @assert alloctype(a) == alloctype(b)
  elt = typeof(val)
  return UnallocatedZeros(Zeros{elt}(ax), alloctype(a))
end

function FillArrays.kron_fill(a::UnallocatedFill, b::UnallocatedZeros, val, ax)
  return kron_fill(b, a, val, ax)
end
