using ChainRulesCore
using ITensors
using Zygote

function ITensors.ITensor(m::ITensors.OpTerm, s...)
  os = OpSum()
  add!(os, ITensors.MPOTerm(1.0, m))
  return prod(MPO(os, s...))
end

function ITensors.ITensor(m::ITensors.MPOTerm, s...)
  c = ITensors.coef(m)
  c = isreal(c) ? real(c) : c
  T = ITensor(ITensors.ops(m), s...)
  return c * T
end

function ITensors.ITensor(os::OpSum, s...)
  return sum([ITensor(os[n], s...) for n in 1:length(os)])
end

function _convert_tangent(x::OpSum, ȳ)
  (; coef, ops) = only(ȳ.data)
  return OpSum() + (coef, "Id", 1)
end

function _convert_tangent(x::Tuple, ȳ)
  (; coef, ops) = only(ȳ.data)
  return (coef, x[2:end]...)
end

function ChainRulesCore.rrule(::typeof(+), x1::OpSum, x2)
  y = x1 + x2

  @show y

  function add_pullback(ȳ)
    @show ȳ

    x̄1 = _convert_tangent(x1, ȳ)
    x̄2 = _convert_tangent(x2, ȳ)

    @show x̄1
    @show x̄2

    return (NoTangent(), x̄1, x̄2)
  end
  return y, add_pullback
end

## @non_differentiable ITensor(::ITensors.OpTerm, s...)

function ChainRulesCore.rrule(::Type{ITensor}, x1::ITensors.MPOTerm, xs...)
  y = ITensor(x1, xs...)
  c = ITensors.coef(x1)
  c = isreal(c) ? real(c) : c
  ops = ITensors.ops(x1)

  @show y
  @show c
  @show ops

  function ITensor_pullback(ȳ)
    # XXX: Make this grab the first nonzero value!!!
    c̄ = ȳ[1]
    x̄1 = ITensors.MPOTerm(c̄, ops)
    x̄s = ITensors.ITensorChainRules.broadcast_notangent(xs)

    @show ȳ
    @show c̄
    @show x̄1
    @show x̄s

    return (NoTangent(), x̄1, x̄s...)
  end
  return y, ITensor_pullback
end

function f(x)
  os = OpSum()
  os += (cos(x), "Sz", 1)
  return real(ITensors.coef(os[1]))
end

x = 1.2
@show x
@show f(x)
@show f'(x)

function g(x)
  os = OpSum()
  os += (cos(x), "Sz", 1)
  o = ITensor(os, s)
  return o[1, 1]
end

s = siteinds("S=1/2", 1)
@show x
@show g(x)
@show g'(x)
