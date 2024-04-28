using ITensors.NDTensors

using ITensors.NDTensors: AllowAlias

function rrule(f::Type{<:Tensor}, x1::AllowAlias, x2::TensorStorage, x3::Tuple)
  y = f(x1, x2, x3)
  function Tensor_pullback(ȳ)
    x̄1 = NoTangent()
    x̄2 = ȳ.storage
    x̄3 = NoTangent()
    return (NoTangent(), x̄1, x̄2, x̄3)
  end
  return y, Tensor_pullback
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
