using FillArrays: FillArrays, AbstractFill, Fill, broadcasted_fill, kron_fill, mult_fill
using NDTensors.SetParameters: Position, set_parameters

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

Base.convert(::Type{<:UnallocatedFill}, A::UnallocatedFill) = A

#############################################
# Arithmatic

# mult_fill(a, b, val, ax) = Fill(val, ax)
function FillArrays.mult_fill(a::UnallocatedFill, b, val, ax)
  return UnallocatedFill(Fill(val, ax), alloctype(a))
end
FillArrays.mult_fill(a, b::UnallocatedFill, val, ax) = mult_fill(b, a, val, ax)
function FillArrays.mult_fill(a::UnallocatedFill, b::UnallocatedFill, val, ax)
  @assert(alloctype(a) == alloctype(b))
  return UnallocatedFill(Fill(val, ax), alloctype(a))
end

function FillArrays.broadcasted_fill(f, a::UnallocatedFill, val, ax)
  return UnallocatedFill(Fill(val, ax), alloctype(a))
end
function FillArrays.broadcasted_fill(f, a::UnallocatedFill, b::UnallocatedFill, val, ax)
  @assert(alloctype(a) == alloctype(b))
  return UnallocatedFill(Fill(val, ax), alloctype(a))
end

function FillArrays.broadcasted_fill(f, a::UnallocatedFill, b, val, ax)
  return UnallocatedFill(Fill(val, ax), alloctype(a))
end
function FillArrays.broadcasted_fill(f, a, b::UnallocatedFill, val, ax)
  return broadcasted_fill(f, b, a, val, ax)
end

function FillArrays.kron_fill(a::UnallocatedFill, b::UnallocatedFill, val, ax)
  @assert alloctype(a) == alloctype(b)
  return UnallocatedFill(Fill(val, ax), alloctype(a))
end
