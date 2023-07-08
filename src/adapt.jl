adapt_structure(to, x::ITensor) = itensor(adapt(to, tensor(x)))

# @inline function NDTensors.cu(x::ITensor; unified::Bool=false)
#   return itensor(NDTensors.cu(storage(x); unified=unified), inds(x))
# end
