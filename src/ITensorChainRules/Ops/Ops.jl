function Tangent_to_NamedTuple(t)
  return NamedTuple((k => t[k] for k in keys(t)))
end

Tangent_to_NamedTuple(::ZeroTangent) = ZeroTangent()

function rrule(::Type{Op}, x1, x2, x3)
  y = Op(x1, x2, x3)
  function Op_pullback(ȳ)
    x̄1 = x1
    x̄2 = x2
    t = ȳ.params
    x̄3 = Tangent_to_NamedTuple(t)
    return (NoTangent(), x̄1, x̄2, x̄3)
  end
  return y, Op_pullback
end

@non_differentiable Ops.sites(::Any)
