using ITensors
using Test

using ChainRulesCore: rrule_via_ad

include("utils/chainrulestestutils.jl")

using Zygote: ZygoteRuleConfig, gradient

@testset "ChainRules rrules: Ops" begin
  s = siteinds("S=1/2", 4)

  x = 2.4
  V = randomITensor(s[1], s[2])

  f = function (x)
    y = ITensor(Op("Ry", 1; θ=x), s)
    return y[1, 1]
  end
  args = (x,)
  test_rrule(
    ZygoteRuleConfig(),
    f,
    args...;
    rrule_f=rrule_via_ad,
    check_inferred=false,
    rtol=1.0e-7,
    atol=1.0e-7,
  )

  f = function (x)
    y = exp(ITensor(Op("Ry", 1; θ=x), s))
    return y[1, 1]
  end
  args = (x,)
  test_rrule(
    ZygoteRuleConfig(),
    f,
    args...;
    rrule_f=rrule_via_ad,
    check_inferred=false,
    rtol=1.0e-7,
    atol=1.0e-7,
  )

  f = function (x)
    y = Op("Ry", 1; θ=x) + Op("Ry", 1; θ=x)
    return y[1].params.θ
  end
  args = (x,)
  test_rrule(
    ZygoteRuleConfig(),
    f,
    args...;
    rrule_f=rrule_via_ad,
    check_inferred=false,
    rtol=1.0e-7,
    atol=1.0e-7,
  )

  f = function (x)
    y = ITensor(Op("Ry", 1; θ=x) + Op("Ry", 1; θ=x), s)
    return y[1, 1]
  end
  args = (x,)
  test_rrule(
    ZygoteRuleConfig(),
    f,
    args...;
    rrule_f=rrule_via_ad,
    check_inferred=false,
    rtol=1.0e-7,
    atol=1.0e-7,
  )

  f = function (x)
    y = exp(ITensor(Op("Ry", 1; θ=x) + Op("Ry", 1; θ=x), s))
    return y[1, 1]
  end
  args = (x,)
  test_rrule(
    ZygoteRuleConfig(),
    f,
    args...;
    rrule_f=rrule_via_ad,
    check_inferred=false,
    rtol=1.0e-7,
    atol=1.0e-7,
  )

  f = function (x)
    y = ITensor(exp(Op("Ry", 1; θ=x) + Op("Ry", 1; θ=x)), s)
    return y[1, 1]
  end
  args = (x,)
  test_rrule(
    ZygoteRuleConfig(),
    f,
    args...;
    rrule_f=rrule_via_ad,
    check_inferred=false,
    rtol=1.0e-7,
    atol=1.0e-7,
  )

  f = function (x)
    y = ITensor(2 * Op("Ry", 1; θ=x), s)
    return y[1, 1]
  end
  args = (x,)
  test_rrule(
    ZygoteRuleConfig(),
    f,
    args...;
    rrule_f=rrule_via_ad,
    check_inferred=false,
    rtol=1.0e-7,
    atol=1.0e-7,
  )

  f = function (x)
    y = ITensor(2 * (Op("Ry", 1; θ=x) + Op("Ry", 1; θ=x)), s)
    return y[1, 1]
  end
  args = (x,)
  test_rrule(
    ZygoteRuleConfig(),
    f,
    args...;
    rrule_f=rrule_via_ad,
    check_inferred=false,
    rtol=1.0e-7,
    atol=1.0e-7,
  )

  f = function (x)
    y = ITensor(Op("Ry", 1; θ=x) * Op("Ry", 2; θ=x), s)
    return y[1, 1]
  end
  args = (x,)
  test_rrule(
    ZygoteRuleConfig(),
    f,
    args...;
    rrule_f=rrule_via_ad,
    check_inferred=false,
    rtol=1.0e-7,
    atol=1.0e-7,
  )

  f = function (x)
    y = ITensor(exp(-x * Op("X", 1) * Op("X", 2)), s)
    return norm(y)
  end
  args = (x,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

  f = function (x)
    y = exp(-x * Op("X", 1) * Op("X", 2))
    y *= exp(-x * Op("X", 1) * Op("X", 2))
    U = Prod{ITensor}(y, s)
    return norm(U(V))
  end
  args = (x,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

  f = function (x)
    y = exp(-x * (Op("X", 1) + Op("Z", 1) + Op("Z", 1)); alg=Trotter{1}(1))
    U = Prod{ITensor}(y, s)
    return norm(U(V))
  end
  args = (x,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

  ## ## XXX: Error in vcat!
  ## f = function (x)
  ##   y = -x * (Op("X", 1) * Op("X", 2) + Op("Z", 1) * Op("Z", 2))
  ##   U = ITensor(y, s)
  ##   return norm(U * V)
  ## end
  ## 
  ## ## XXX: Error in vcat!
  ## f = function (x)
  ##   y = exp(-x * (Op("X", 1) * Op("X", 2) + Op("Z", 1) * Op("Z", 2)); alg=Trotter{1}(1))
  ##   U = ITensor(y, s)
  ##   return norm(U * V)
  ## end
end
