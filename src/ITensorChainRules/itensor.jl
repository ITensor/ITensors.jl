using ITensors: Indices

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

@non_differentiable combiner(::Indices)
