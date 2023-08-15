# TODO: Deprecated.
"""
    emptyITensor([::Type{ElT} = EmptyNumber, ]inds)
    emptyITensor([::Type{ElT} = EmptyNumber, ]inds::QNIndex...)

Construct an ITensor with `NDTensors.BlockSparse` storage of element type `ElT` with the no blocks.

If `ElT` is not specified it defaults to `NDTensors.EmptyNumber`.
"""
function emptyITensor(::Type{ElT}, inds::QNIndices) where {ElT<:Number}
  return itensor(EmptyBlockSparseTensor(ElT, inds))
end
emptyITensor(inds::QNIndices) = emptyITensor(EmptyNumber, inds)

function emptyITensor(eltype::Type{<:Number}, flux::QN, is...)
  return error(
    "Trying to create an empty ITensor with flux $flux, cannot create empty ITensor with a specified flux.",
  )
end
emptyITensor(flux::QN, is...) = emptyITensor(EmptyNumber, flux, is...)
