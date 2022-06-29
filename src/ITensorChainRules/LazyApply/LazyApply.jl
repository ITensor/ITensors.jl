function rrule(::Type{Applied}, x1, x2::Tuple, x3::NamedTuple)
  y = Applied(x1, x2, x3)
  function Applied_pullback(ȳ)
    x̄1 = ȳ.f
    x̄2 = ȳ.args
    x̄3 = ȳ.kwargs
    return (NoTangent(), x̄1, x̄2, x̄3)
  end
  function Applied_pullback(ȳ::Vector)
    x̄1 = NoTangent()
    x̄2 = (ȳ,)
    x̄3 = NoTangent()
    return (NoTangent(), x̄1, x̄2, x̄3)
  end
  return y, Applied_pullback
end
