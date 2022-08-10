#
# Used to adapt `EmptyStorage` types
#

NDTensors.cu(eltype::Type{<:Number}, x) = fmap(x -> adapt(CuArray{eltype}, x), x)
NDTensors.cu(x) = fmap(x -> adapt(CuArray, x), x)

NDTensors.to_vector_type(arraytype::Type{CuArray}) = CuVector
NDTensors.to_vector_type(arraytype::Type{CuArray{T}}) where {T} = CuVector{T}

function NDTensors.set_eltype_if_unspecified(
  arraytype::Type{CuVector{T}}, eltype::Type
) where {T}
  return arraytype
end
function NDTensors.set_eltype_if_unspecified(arraytype::Type{CuVector}, eltype::Type)
  return CuVector{eltype}
end

# Overload `CUDA.cu` for convenience
const ITensorType = Union{
  TensorStorage,Tensor,ITensor,Array{ITensor},Array{<:Array{ITensor}},MPS,MPO
}
CUDA.cu(x::ITensorType) = NDTensors.cu(x)
CUDA.cu(eltype::Type{<:Number}, x::ITensorType) = NDTensors.cu(eltype, x)
