#
# Used to adapt `EmptyStorage` types
#

NDTensors.cu(eltype::Type{<:Number}, x) = fmap(x -> adapt(CuVector{eltype, default_buffertype()}, x), x)
NDTensors.cu(x) = fmap(x -> adapt(CuVector, x), x)

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

function NDTensors.adapt_storagetype(
  to::Type{<:CUDA.CuArray}, x::Type{<:NDTensors.EmptyStorage}
)
  store = NDTensors.storagetype(x)
  return NDTensors.emptytype(
    NDTensors.set_datatype(store, CuVector{eltype(store), default_buffertype()})
  )
end