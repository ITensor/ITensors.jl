#
# EmptyStorage ITensor constructors
#

"""
    emptyITensor([::Type{ElT} = NDTensors.default_eltype(), ]inds)
    emptyITensor([::Type{ElT} = NDTensors.default_eltype(), ]inds::Index...)

Construct an ITensor with storage type `NDTensors.EmptyStorage`, indices `inds`, and element type `ElT`. If the element type is not specified, it defaults to `NDTensors.default_eltype()`, which represents a number type that can take on any value (for example, the type of the first value it is set to).
"""
function emptyITensor(elt::Type{<:Number}, is::Indices)
  z = NDTensors.Zeros{elt,1,NDTensors.default_datatype(elt)}(Tuple(dim(is)))
  return itensor(z, is)
end

function emptyITensor(elt::Type{<:Number}, is...)
  return emptyITensor(elt, indices(is...))
end

emptyITensor(is::Indices) = emptyITensor(NDTensors.default_eltype(), is)

emptyITensor(is...) = emptyITensor(NDTensors.default_eltype(), indices(is...))
