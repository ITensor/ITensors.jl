using FillArrays: FillArrays, getindex_value
using NDTensors.SetParameters: set_parameters
using Adapt: adapt

const UnallocatedArray{ElT,N,AxesT,AllocT} = Union{
  UnallocatedFill{ElT,N,AxesT,AllocT},UnallocatedZeros{ElT,N,AxesT,AllocT}
}

@inline Base.axes(A::UnallocatedArray) = axes(parent(A))
Base.size(A::UnallocatedArray) = size(parent(A))
function FillArrays.getindex_value(A::UnallocatedArray)
  return getindex_value(parent(A))
end

function Base.complex(A::UnallocatedArray)
  return complex(eltype(A)).(A)
end

function Base.transpose(a::UnallocatedArray)
  return set_alloctype(transpose(parent(a)), alloctype(a))
end

function Base.adjoint(a::UnallocatedArray)
  return set_alloctype(adjoint(parent(a)), alloctype(a))
end

function set_alloctype(T::Type{<:UnallocatedArray}, alloc::Type{<:AbstractArray})
  return set_parameters(T, Position{4}(), alloc)
end

## This overloads the definition defined in `FillArrays.jl`
for STYPE in (:AbstractArray, :AbstractFill)
  @eval begin
    @inline $STYPE{T}(F::UnallocatedArray{T}) where {T} = F
    @inline $STYPE{T,N}(F::UnallocatedArray{T,N}) where {T,N} = F
  end
end

function allocate(f::UnallocatedArray)
  a = similar(f)
  fill!(a, getindex_value(f))
  return a
end

function allocate(arraytype::Type{<:AbstractArray}, elt::Type, axes::Tuple)
  ## TODO rewrite this using set_eltype and set_ndims functions
  ## currently these functions are defined in `NDTensors`
  ## In the future they should be defined in `SetParameters`
  ArrayT = set_parameters(arraytype, Position{1}(), elt)
  return similar(ArrayT, axes)
end

function Base.similar(f::UnallocatedArray, elt::Type, axes::Tuple{Int64, Vararg{Int64}})
  return allocate(alloctype(f), elt, axes)
end

## TODO fix this because reshape loses alloctype
#FillArrays.reshape(a::Union{<:UnallocatedFill, <:UnallocatedZeros}, dims) = set_alloctype(reshape(parent(a), dims), allocate(a))

# function Adapt.adapt_storage(to::Type{<:AbstractArray}, x::Union{<:UnallocatedFill, <:UnallocatedZeros})
#   return set_alloctype(parent(x), to)
# end

# function Adapt.adapt_storage(to::Type{<:Number}, x::)
