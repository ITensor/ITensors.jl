#
# Used to adapt `EmptyStorage` types
#

NDTensors.mtl(eltype::Type{<:Number}, x) = fmap(x -> adapt(MtlArray{eltype, 1}, x), x)
NDTensors.mtl(x) = Functors.fmap(x -> Adapt.adapt(MtlArray, x), x)

NDTensors.to_vector_type(arraytype::Type{MtlArray}) = MtlVector
NDTensors.to_vector_type(arraytype::Type{MtlArray{T}}) where {T} = MtlVector{T}

function NDTensors.set_eltype_if_unspecified(
  arraytype::Type{MtlArray{T}}, eltype::Type
) where {T}
  return arraytype
end
function NDTensors.set_eltype_if_unspecified(arraytype::Type{MtlArray}, eltype::Type)
  return MtlVector{eltype}
end

# Overload `CUDA.cu` for convenience
# const ITensorType = Union{
#   TensorStorage,Tensor,ITensor,Array{ITensor},Array{<:Array{ITensor}},MPS,MPO
# }

Metal.mtl(x::TensorStorage) = typeof(x)(NDTensors.mtl(data(x)))
Metal.mtl(x::Type{<:Dense}) = Dense{eltype(x), MtlArray{eltype(x)}}

#CUDA.cu(eltype::Type{<:Number}, x::ITensorType) = NDTensors.cu(eltype, x)
