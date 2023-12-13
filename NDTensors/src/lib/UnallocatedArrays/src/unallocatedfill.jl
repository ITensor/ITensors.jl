using FillArrays: FillArrays, AbstractFill, Fill, broadcasted_fill, kron_fill, mult_fill
using NDTensors.SetParameters: Position, set_parameters
import Base: +

struct UnallocatedFill{ElT,N,Axes,Alloc} <: AbstractFill{ElT,N,Axes}
  f::Fill{ElT,N,Axes}
  alloc::Alloc

  function UnallocatedFill{ElT,N,Axes}(
    f::Fill, alloc::Type{<:AbstractArray{ElT,N}}
  ) where {ElT,N,Axes}
    return new{ElT,N,Axes,Type{alloc}}(f, alloc)
  end
end

function UnallocatedFill{ElT,N}(f::Fill, alloc) where {ElT,N}
  return UnallocatedFill{ElT,N,typeof(axes(f))}(f, alloc)
end

function UnallocatedFill{ElT}(f::Fill, alloc) where {ElT}
  return UnallocatedFill{ElT,ndims(f)}(f, alloc)
end

function UnallocatedFill(f::Fill, alloc)
  return UnallocatedFill{eltype(f)}(f, alloc)
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

function +(A::UnallocatedFill, B::UnallocatedFill)
  FillArrays.promote_shape(A, B)
  b = getindex_value(B)
  return A .+ b
end

# ####
# ## TODO use `set_parameters` as constructor to these types
# function UnallocatedFill(f::Fill, alloc::Type{<:AbstractArray})
#   ## set_axes -> set_axes_type
#   return set_alloctype(
#     set_axes(set_ndims(set_eltype(UnallocatedFill, eltype(f)), ndims(f)), typeof(axes(f))),
#     alloc,
#   )(
#     f
#   )
# end

## Things to fix
## in different Change syntax of set_xxx_if_unspecified
# adapt
