using ChainRulesCore
using ChainRulesTestUtils
using FiniteDifferences
using ITensors
using Random

#
# For ITensor compatibility with FiniteDifferences
#

function FiniteDifferences.to_vec(A::ITensor)
  # TODO: generalize to sparse tensors
  # TODO: define `itensor([1.0])` as well
  # as `itensor([1.0], ())` to help with generic code.
  function vec_to_ITensor(x)
    return isempty(inds(A)) ? ITensor(x[]) : itensor(x, inds(A))
  end
  return vec(array(A)), vec_to_ITensor
end

function FiniteDifferences.to_vec(x::Index)
  return (Bool[], _ -> x)
end

function FiniteDifferences.to_vec(x::Tuple{Vararg{Index}})
  return (Bool[], _ -> x)
end

function FiniteDifferences.to_vec(x::Pair{<:Tuple{Vararg{Index}},<:Tuple{Vararg{Index}}})
  return (Bool[], _ -> x)
end

function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, A::ITensor)
  # TODO: generalize to sparse tensors
  return isempty(inds(A)) ? ITensor(randn(eltype(A))) : randomITensor(eltype(A), inds(A))
end

function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::Index)
  return NoTangent()
end

function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::Tuple{Vararg{Index}})
  return NoTangent()
end

function ChainRulesTestUtils.rand_tangent(
  rng::AbstractRNG, x::Pair{<:Tuple{Vararg{Index}},<:Tuple{Vararg{Index}}}
)
  return NoTangent()
end
