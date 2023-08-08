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

function ITensor(as::AliasStyle, elt::Type{<:Number}, A::AbstractArray{<:Number}, inds::Tuple{}; kwargs...)
  data = set_eltype(typeof(A), elt)(as, A)
  return ITensor(as, NDTensors.default_storagetype(elt, inds)(data), inds)
end

# For now, it's not well defined to construct an ITensor without indices
# from a non-zero dimensional Array.
function ITensor(as::AliasStyle, elt::Type{<:Number}, A::AbstractArray{<:Number}; kwargs...)
  if length(A) > 1
    error(
      "Trying to create an ITensor without any indices from Array $A of dimensions $(size(A)). Cannot construct an ITensor from an Array with more than one element without any indices.",
    )
  end
  return ITensor(as, elt, A, (); kwargs...)
end

function ITensor(elt::Type{<:Number}, A::AbstractArray{<:Number}; kwargs...)
  return ITensor(NeverAlias(), elt, A; kwargs...)
end

function ITensor(A::AbstractArray; kwargs...)
  return ITensor(NeverAlias(), eltype(A), A; kwargs...)
end

function ITensor(::Type{ElT}, inds::Tuple{}) where {ElT<:Number}
  return ITensor(AllowAlias(), ElT, NDTensors.default_datatype(ElT)(), ())
end

# To fix ambiguity with QN Index version
# TODO: define as `emptyITensor(ElT)`
ITensor(ElT::Type{<:Number}=NDTensors.default_eltype()) = ITensor(ElT, ())

# function emptyITensor(elt::Type{<:Number}=NDTensors.default_eltype())
#   return emptyITensor(elt, ())
# end

# To fix ambiguity with QN version
function randomITensor(::Type{ElT}, is::Tuple{}) where {ElT<:Number}
  return randomITensor(Random.default_rng(), ElT, is)
end

# To fix ambiguity with QN version
function randomITensor(rng::AbstractRNG, ::Type{ElT}, is::Tuple{}; kwargs...) where {ElT<:Number}
  v = randn(rng, ElT, dim(is))
  return ITensor(AllowAlias(), ElT, v, is; kwargs...)
end

# To fix ambiguity with QN version
function randomITensor(is::Tuple{})
  return randomITensor(Random.default_rng(), ())
end

# To fix ambiguity with QN version
function randomITensor(rng::AbstractRNG, is::Tuple{})
  return randomITensor(rng, NDTensors.default_eltype(), ())
end

# To fix ambiguity errors with QN version
function randomITensor(::Type{ElT}) where {ElT<:Number}
  return randomITensor(Random.default_rng(), ElT, ())
end

# To fix ambiguity errors with QN version
function randomITensor(rng::AbstractRNG, ::Type{ElT}) where {ElT<:Number}
  return randomITensor(rng, ElT, ())
end

# To fix ambiguity errors with QN version
randomITensor() = randomITensor(Random.default_rng())

# To fix ambiguity errors with QN version
randomITensor(rng::AbstractRNG) = randomITensor(rng, Float64, ())
