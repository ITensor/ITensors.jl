#
# Used to adapt `EmptyStorage` types
#

struct NDTensorCuArrayAdaptor{B} end
## TODO make this work for unified. This works but overwrites CUDA's adapt_storage. This fails for emptystorage...
@inline function NDTensors.cu(xs; unified::Bool=false)
  return fmap(
    x -> adapt(NDTensorCuArrayAdaptor{unified ? Mem.UnifiedBuffer : Mem.DeviceBuffer}(), x),
    xs,
  )
end

function Adapt.adapt_storage(
  ::NDTensorCuArrayAdaptor{B}, xs::AbstractArray{T,N}
) where {T,N,B}
  return isbits(xs) ? xs : CuArray{T,N,B}(xs)
end

# @inline function NDTensors.cu(xs::AbstractArray; unified::Bool=false)
#   ElT = eltype(xs)
#   N = ndims(xs)
#   return NDTensors.adapt_structure(
#     CuArray{ElT,N,(unified ? CUDA.Mem.UnifiedBuffer : CUDA.Mem.DeviceBuffer)}, xs
#   )
# end

# @inline function NDTensors.cu(xs::Tensor; unified::Bool=false)
#   ElT = eltype(xs)
#   return NDTensors.adapt_structure(
#     CuVector{ElT,(unified ? CUDA.Mem.UnifiedBuffer : CUDA.Mem.DeviceBuffer)}, xs
#   )
# end
