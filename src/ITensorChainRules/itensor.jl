function rrule(::Type{ITensor}, x1::AllowAlias, x2::TensorStorage, x3)
  y = ITensor(x1, x2, x3)
  function ITensor_pullback(ȳ)
    x̄1 = NoTangent()
    x̄2 = ȳ.tensor.storage
    x̄3 = NoTangent()
    return (NoTangent(), x̄1, x̄2, x̄3)
  end
  return y, ITensor_pullback
end

function rrule(::Type{ITensor}, x1::AllowAlias, x2::Tensor)
  y = ITensor(x1, x2)
  function ITensor_pullback(ȳ)
    x̄1 = NoTangent()
    x̄2 = Tensor(x1, ȳ)
    return (NoTangent(), x̄1, x̄2)
  end
  return y, ITensor_pullback
end

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

function rrule(f::Type{<:Tensor}, x1::AllowAlias, x2::ITensor)
  y = f(x1, x2)
  function Tensor_pullback(ȳ)
    x̄1 = NoTangent()
    x̄2 = ITensor(x1, ȳ)
    return (NoTangent(), x̄1, x̄2)
  end
  return y, Tensor_pullback
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

# Special case for contracting a pair of ITensors
function ChainRulesCore.rrule(::typeof(contract), x1::ITensor, x2::ITensor)
  y = x1 * x2
  function contract_pullback(ȳ)
    x̄1 = ȳ * dag(x2)
    x̄2 = dag(x1) * ȳ
    return (NoTangent(), x̄1, x̄2)
  end
  return y, contract_pullback
end

@non_differentiable ITensors.optimal_contraction_sequence(::Any)

function ChainRulesCore.rrule(::typeof(*), x1::Number, x2::ITensor)
  y = x1 * x2
  function contract_pullback(ȳ)
    x̄1 = ȳ * dag(x2)
    x̄2 = dag(x1) * ȳ
    return (NoTangent(), x̄1[], x̄2)
  end
  return y, contract_pullback
end

function ChainRulesCore.rrule(::typeof(*), x1::ITensor, x2::Number)
  y = x1 * x2
  function contract_pullback(ȳ)
    x̄1 = ȳ * dag(x2)
    x̄2 = dag(x1) * ȳ
    return (NoTangent(), x̄1, x̄2[])
  end
  return y, contract_pullback
end

function ChainRulesCore.rrule(::typeof(+), x1::ITensor, x2::ITensor)
  y = x1 + x2
  function add_pullback(ȳ)
    return (NoTangent(), ȳ, ȳ)
  end
  return y, add_pullback
end

function ChainRulesCore.rrule(::typeof(itensor), x::Array, a...)
  y = itensor(x, a...)
  function itensor_pullback(ȳ)
    uȳ = permute(unthunk(ȳ), a...)
    x̄ = reshape(array(uȳ), size(x))
    ā = broadcast_notangent(a)
    return (NoTangent(), x̄, ā...)
  end
  return y, itensor_pullback
end

function ChainRulesCore.rrule(::Type{ITensor}, x::Array{<:Number}, a...)
  y = ITensor(x, a...)
  function ITensor_pullback(ȳ)
    # TODO: define `Array(::ITensor)` directly
    uȳ = Array(unthunk(ȳ), a...)
    x̄ = reshape(uȳ, size(x))
    ā = broadcast_notangent(a)
    return (NoTangent(), x̄, ā...)
  end
  return y, ITensor_pullback
end

function ChainRulesCore.rrule(::Type{ITensor}, x::Number)
  y = ITensor(x)
  function ITensor_pullback(ȳ)
    x̄ = ȳ[]
    return (NoTangent(), x̄)
  end
  return y, ITensor_pullback
end

function ChainRulesCore.rrule(::typeof(dag), x)
  y = dag(x)
  function dag_pullback(ȳ)
    x̄ = dag(unthunk(ȳ))
    return (NoTangent(), x̄)
  end
  return y, dag_pullback
end

function ChainRulesCore.rrule(::typeof(permute), x::ITensor, a...)
  y = permute(x, a...)
  function permute_pullback(ȳ)
    x̄ = permute(unthunk(ȳ), inds(x))
    ā = broadcast_notangent(a)
    return (NoTangent(), x̄, ā...)
  end
  return y, permute_pullback
end

@non_differentiable combiner(::Indices)
