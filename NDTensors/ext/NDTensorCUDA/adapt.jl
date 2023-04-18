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
  return isbits(xs) ? xs : CuArray{T,1,B}(xs)
end

function NDTensors.adapt_storagetype(
  ::NDTensorCuArrayAdaptor{B}, xs::Type{EmptyStorage{ElT,StoreT}}
) where {ElT,StoreT,B}
  return NDTensors.emptytype(NDTensors.adapt_storagetype(CuVector{ElT,B}, StoreT))
end

## In house patch to deal issue of calling ndims with an Array of unspecified eltype
## https://github.com/JuliaLang/julia/pull/40682
if VERSION < v"1.7"
  ndims(::Type{<:CuArray{<:Any,N}}) where {N} = N
end