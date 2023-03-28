#
# Used to adapt `EmptyStorage` types
#

to_vector_type(arraytype::Type{CuArray}) = CuVector
to_vector_type(arraytype::Type{CuArray{T}}) where {T} = CuVector{T}

@inline function NDTensors.cu(xs; unified::Bool=false)
  return CUDA.cu(xs; unified)
end

@inline function NDTensors.cu(xs::AbstractArray; unified::Bool=false)
  ElT = eltype(xs)
  N = ndims(xs)
  return NDTensors.adapt_structure(
    CuArray{ElT, N, (unified ? CUDA.Mem.UnifiedBuffer : CUDA.Mem.DeviceBuffer)}, xs
  )
end
