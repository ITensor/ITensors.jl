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

  function sometimes_broken_test()
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
    return nothing
  end

  @static if VERSION > v"1.8"
    @test_skip sometimes_broken_test()
  else
    sometimes_broken_test()
  end

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
    U = ITensor(y, s)
    return norm(U)
  end
  args = (x,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

  U1(θ) = Op("Ry", 1; θ)
  U2(θ) = Op("Ry", 2; θ)

  f = function (x)
    return ITensor(U1(x), s)[1, 1]
  end
  args = (x,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

  f = function (x)
    return ITensor(U1(x) * U2(x), s)[1, 1]
  end
  args = (x,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

  f = function (x)
    return ITensor(1.2 * U1(x), s)[1, 1]
  end
  args = (x,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

  f = function (x)
    return ITensor(exp(1.2 * U1(x)), s)[1, 1]
  end
  args = (x,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

  f = function (x)
    return ITensor(exp(x * U1(1.2)), s)[1, 1]
  end
  args = (x,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

  function H(x1, x2)
    os = Ops.OpSum()
    os += x1 * Op("X", 1)
    os += x2 * Op("X", 2)
    return os
  end

  # These are broken in versions of Zygote after 0.6.43,
  # See: https://github.com/FluxML/Zygote.jl/issues/1304
  @test_skip begin
    f = function (x)
      return ITensor(exp(1.5 * H(x, x); alg=Trotter{1}(1)), s)[1, 1]
    end
    args = (x,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

    f = function (x)
      return ITensor(exp(1.5 * H(x, x); alg=Trotter{2}(1)), s)[1, 1]
    end
    args = (x,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

    f = function (x)
      return ITensor(exp(1.5 * H(x, x); alg=Trotter{2}(2)), s)[1, 1]
    end
    args = (x,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

    f = function (x)
      return ITensor(exp(x * H(x, x); alg=Trotter{2}(2)), s)[1, 1]
    end
    args = (x,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end

  f = function (x)
    y = -x * (Op("X", 1) * Op("X", 2) + Op("Z", 1) * Op("Z", 2))
    U = ITensor(y, s)
    return norm(U * V)
  end
  args = (x,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

  f = function (x)
    y = exp(-x * (Op("X", 1) * Op("X", 2) + Op("Z", 1) * Op("Z", 2)); alg=Trotter{1}(1))
    U = ITensor(y, s)
    return norm(U * V)
  end
  args = (x,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

  ## XXX: Fix
  f = function (x)
    y = exp(-x * Op("X", 1) * Op("X", 2))
    y *= exp(-x * Op("X", 1) * Op("X", 2))
    U = Prod{ITensor}(y, s)
    return norm(U(V))
  end
  args = (x,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

  ## XXX: Fix
  f = function (x)
    y = exp(-x * (Op("X", 1) + Op("Z", 1) + Op("Z", 1)); alg=Trotter{1}(1))
    U = Prod{ITensor}(y, s)
    return norm(U(V))
  end
  args = (x,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
end
