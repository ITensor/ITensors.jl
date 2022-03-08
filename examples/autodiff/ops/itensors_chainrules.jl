using ChainRulesCore
using ITensors
using ITensors.NDTensors

using ITensors: Indices

import ChainRulesCore: rrule

function rrule(::Type{ITensor}, x1::Tensor)
  y = ITensor(x1)
  function ITensor_pullback(ȳ)
    x̄1 = Tensor(ȳ)
    return (NoTangent(), x̄1)
  end
  return y, ITensor_pullback
end

function rrule(::typeof(itensor), x1::Tensor)
  y = itensor(x1)
  function itensor_pullback(ȳ)
    x̄1 = tensor(ȳ)
    return (NoTangent(), x̄1)
  end
  return y, itensor_pullback
end

function rrule(f::Type{<:Tensor}, x1::ITensor)
  y = f(x1)
  function Tensor_pullback(ȳ)
    x̄1 = ITensor(ȳ)
    return (NoTangent(), x̄1)
  end
  return y, Tensor_pullback
end

function rrule(::typeof(tensor), x1::ITensor)
  y = tensor(x1)
  function tensor_pullback(ȳ)
    x̄1 = ITensor(typeof(storage(x1))(ȳ.storage.data), inds(x1))
    return (NoTangent(), x̄1)
  end
  return y, tensor_pullback
end

function rrule(::typeof(tensor), x1::TensorStorage, x2::Tuple)
  y = tensor(x1, x2)
  function tensor_pullback(ȳ)
    x̄1 = storage(ȳ)
    x̄2 = NoTangent()
    return (NoTangent(), x̄1, x̄2)
  end
  return y, tensor_pullback
end

function rrule(f::Type{<:Tensor}, x1::TensorStorage, x2::Tuple)
  y = f(x1, x2)
  function tensor_pullback(ȳ)
    x̄1 = copy(storage(x1))
    x̄2 = NoTangent()
    return (NoTangent(), x̄1, x̄2)
  end
  return y, tensor_pullback
end

function rrule(f::Type{<:Dense}, x1::AbstractVector)
  y = f(x1)
  function Dense_pullback(ȳ)
    x̄1 = ȳ.data
    return (NoTangent(), x̄1)
  end
  return y, Dense_pullback
end

@non_differentiable has_fermion_string(::AbstractString, ::Index)
@non_differentiable permute(::Indices, ::Indices)
@non_differentiable combiner(::Indices)
