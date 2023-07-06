import Metal: MtlArrayAdaptor, Private, Shared;
@inline NDTensors.mtl(xs; storage=Private) = NDTensors.adapt(MtlArray, xs)

## In the future, to preserve the value of `storage` we need to use `MtlArrayAdaptor`
# @inline NDTensors.mtl(xs; storage=Private) = NDTensors.adapt(MtlArrayAdaptor{storage}(), xs)

# function NDTensors.adapt_storagetype(
#   adaptor::Metal.MtlArrayAdaptor, xs::Type{EmptyStorage{ElT,StoreT}}
# ) where {ElT,StoreT}
#   @show NDTensors.adapt_storage(adaptor, StoreT)
#   return NDTensors.emptytype(NDTensors.adapt(adaptor, StoreT))
# end
