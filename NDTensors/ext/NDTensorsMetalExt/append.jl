# This circumvents an issues that `MtlArray` can't call `resize!`.
# TODO: Raise an issue with Metal.jl.
function NDTensors.append!!(::Type{<:MtlArray}, collection, collections...)
  return vcat(collection, collections...)
end

