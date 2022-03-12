using ChainRulesCore
using ITensors
using ITensors.LazyApply
using ITensors.Ops

using ITensors.LazyApply: Applied, AppliedTupleVector
using ITensors.Ops: which_op, WhichOp

import ChainRulesCore: rrule

function Tangent_to_NamedTuple(t)
  return NamedTuple((k => t[k] for k in keys(t)))
end

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

@non_differentiable Ops.sites(::Any)
