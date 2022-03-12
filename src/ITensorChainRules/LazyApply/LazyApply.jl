using ITensors.LazyApply: Applied, AppliedTupleVector

function rrule(f::Type{<:Applied}, x1, x2::Tuple)
  y = f(x1, x2)
  function Applied_pullback(ȳ)
    x̄1 = NoTangent()
    x̄2 = ȳ.args
    return (NoTangent(), x̄1, x̄2)
  end
  return (y, Applied_pullback)
end

function rrule(f::Type{<:AppliedTupleVector}, x1::Vector)
  y = f(x1)
  function Applied_pullback(ȳ)
    x̄1 = ȳ.args[1]
    return (NoTangent(), x̄1)
  end
  function Applied_pullback(ȳ::Vector)
    x̄1 = ȳ
    return (NoTangent(), x̄1)
  end
  function Applied_pullback(ȳ::ZeroTangent)
    x̄1 = ȳ
    return (NoTangent(), x̄1)
  end
  return (y, Applied_pullback)
end
