#
# Used to adapt `EmptyStorage` types
#

NDTensors.cu(eltype::Type{<:Number}, x) = Functors.fmap(x -> Adapt.adapt(CuArray{eltype}, x), x)
NDTensors.cu(x) = Functors.fmap(x -> Adapt.adapt(CuArray, x), x)

to_vector_type(arraytype::Type{CuArray}) = CuVector
to_vector_type(arraytype::Type{CuArray{T}}) where {T} = CuVector{T}

function adapt_storagetype(to::CUDA.CuArrayAdaptor, x::Type{<:TensorStorage})
  @show to
  return set_datatype(x, set_eltype_if_unspecified(to_vector_type(to), eltype(x)))
end

function adapt_storagetype(to::CUDA.CuArrayAdaptor, x::Type{<:EmptyStorage})
  return emptytype(adapt_storagetype(to, NDTensors.fulltype(x)))
end

function NDTensors.set_eltype_if_unspecified(
  arraytype::Type{CuVector{T}}, eltype::Type
) where {T}
  return arraytype
end
function NDTensors.set_eltype_if_unspecified(arraytype::Type{CuVector}, eltype::Type)
  return CuVector{eltype}
end

# Overload `CUDA.cu` for convenience
# const ITensorType = Union{
#   TensorStorage,Tensor,ITensor,Array{ITensor},Array{<:Array{ITensor}},MPS,MPO
# }
CUDA.cu(x::TensorStorage) = NDTensors.cu(x)
CUDA.cu(eltype::Type{<:Number}, x::TensorStorage) = NDTensors.cu(eltype, x)
