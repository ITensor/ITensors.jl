function op(I::UniformScaling, s::Index...)
  return I.λ * op("Id", s...)
end

function ITensor(o::Op, s::Vector{<:Index})
  return op(o.which_op, map(n -> s[n], o.sites)...; o.params...)
end

function ITensor(o::Scaled, s::Vector{<:Index})
  return coefficient(o) * ITensor(argument(o), s)
end

function ITensor(o::Prod, s::Vector{<:Index})
  T = ITensor(true)
  for a in o.args[1]
    Tₙ = ITensor(a, s)
    # TODO: Implement this logic inside `apply`
    if hascommoninds(T, Tₙ)
      T = T(Tₙ)
    else
      T *= Tₙ
    end
  end
  return T
end

function ITensor(o::Sum, s::Vector{<:Index})
  T = ITensor()
  for a in o.args[1]
    T += ITensor(a, s)
  end
  return T
end

function ITensor(o::Exp, s::Vector{<:Index})
  return exp(ITensor(argument(o), s))
end

function ITensor(o::LazyApply.Adjoint, s::Vector{<:Index})
  return swapprime(dag(ITensor(o', s)), 0 => 1)
end

function Sum{ITensor}(o::Sum, s::Vector{<:Index})
  return Applied(sum, map(oₙ -> ITensor(oₙ, s), o))
end

function Prod{ITensor}(o::Prod, s::Vector{<:Index})
  return Applied(prod, map(oₙ -> ITensor(oₙ, s), o))
end

function apply(o::Prod{ITensor}, v::ITensor)
  ov = v
  for oₙ in only(o.args)
    ov = apply(oₙ, ov)
  end
  return ov
end

function (o::Prod{ITensor})(v::ITensor)
  return apply(o, v)
end
