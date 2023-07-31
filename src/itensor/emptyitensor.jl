#
# EmptyStorage ITensor constructors
#

# TODO: Deprecated!
"""
    emptyITensor([::Type{ElT} = NDTensors.default_eltype(), ]inds)
    emptyITensor([::Type{ElT} = NDTensors.default_eltype(), ]inds::Index...)

Construct an ITensor with storage type `NDTensors.EmptyStorage`, indices `inds`, and element type `ElT`. If the element type is not specified, it defaults to `NDTensors.default_eltype()`, which represents a number type that can take on any value (for example, the type of the first value it is set to).
"""
function emptyITensor(::Type{ElT}, is::Indices) where {ElT<:Number}
  return itensor(NDTensors.Zeros{ElT,1,NDTensors.default_datatype(ElT)}(is), is)
end

function emptyITensor(::Type{ElT}, is...) where {ElT<:Number}
  return emptyITensor(ElT, indices(is...))
end

emptyITensor(is::Indices) = emptyITensor(NDTensors.default_eltype(), is)

emptyITensor(is...) = emptyITensor(NDTensors.default_eltype(), indices(is...))

function emptyITensor(::Type{ElT}=NDTensors.default_eltype()) where {ElT<:Number}
  return itensor(NDTensors.Zeros{ElT,1,NDTensors.default_datatype(ElT)}(()), ())
end
