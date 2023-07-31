# Scalar identity ITensor
# TODO: Implement as a new `Scalar` storage type.
struct OneITensor end

inds(::OneITensor) = ()

# This is to help with generic promote_type code
# in eltype(::AbstractProjMPO)
eltype(::OneITensor) = Bool
dim(::OneITensor) = 1
isoneitensor(::OneITensor) = true
isoneitensor(::ITensor) = false

dag(t::OneITensor) = t

(::OneITensor * A::ITensor) = A
(A::ITensor * ::OneITensor) = A
*(t::OneITensor) = t
deepcontract(ts::Union{ITensor,OneITensor}...) = *(ts...)
