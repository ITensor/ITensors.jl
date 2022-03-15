function rrule(f::Type{<:Dense}, x1::AbstractVector)
  y = f(x1)
  function Dense_pullback(ȳ)
    x̄1 = ȳ.data
    return (NoTangent(), x̄1)
  end
  return y, Dense_pullback
end
