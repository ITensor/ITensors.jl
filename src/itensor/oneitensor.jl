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

### informally defined Scalar ITensors

# For now, it's not well defined to construct an ITensor without indices
# from a non-zero dimensional Array.
function ITensor(as::AliasStyle, ElT::Type{<:Number}, A::AbstractArray{<:Number}; kwargs...)
  if length(A) > 1
    error(
      "Trying to create an ITensor without any indices from Array $A of dimensions $(size(A)). Cannot construct an ITensor from an Array with more than one element without any indices.",
    )
  end
  return ITensor(as, ElT, A, Index(1); kwargs...)
end

function ITensor(eltype::Type{<:Number}, A::AbstractArray{<:Number}; kwargs...)
  return ITensor(NeverAlias(), eltype, A; kwargs...)
end

function ITensor(A::AbstractArray; kwargs...)
  return ITensor(NeverAlias(), eltype(A), A; kwargs...)
end

function emptyITensor(::Type{ElT}=NDTensors.default_eltype()) where {ElT<:Number}
  return emptyITensor(ElT, Index(0))
end
