using ITensors
using Random
using Test

using ChainRulesCore: rrule_via_ad

include("utils/chainrulestestutils.jl")

using Zygote: ZygoteRuleConfig, gradient

Random.seed!(1234)

@testset "ChainRules rrules: basic ITensor operations" begin
  i = Index(2, "i")
  j = Index(2, "j")
  A = randomITensor(i', dag(i))
  V = randomITensor(i)
  Ac = randomITensor(ComplexF64, i', dag(i))
  B = randomITensor(i', dag(i))
  C = ITensor(3.4)
  D = randomITensor(i', j)

  @testset "getindex, priming, tagging, ITensor constructors, dag, etc." begin
    test_rrule(getindex, ITensor(3.4); check_inferred=false)
    test_rrule(getindex, A, 1, 2; check_inferred=false)
    test_rrule(contract, A', A; check_inferred=false)
    test_rrule(*, 3.2, A; check_inferred=false)
    test_rrule(*, A, 4.3; check_inferred=false)
    test_rrule(+, A, B; check_inferred=false)
    test_rrule(prime, A; check_inferred=false)
    test_rrule(prime, A, 2; check_inferred=false)
    test_rrule(prime, A; fkwargs=(; tags="i"), check_inferred=false)
    test_rrule(prime, A; fkwargs=(; tags="x"), check_inferred=false)
    test_rrule(setprime, D, 2; check_inferred=false)
    test_rrule(noprime, D; check_inferred=false)
    test_rrule(replaceprime, A, 1 => 2; check_inferred=false)
    test_rrule(replaceprime, A, 1, 2; check_inferred=false)
    test_rrule(swapprime, A, 0 => 1; check_inferred=false)
    test_rrule(swapprime, A, 0, 1; check_inferred=false)
    test_rrule(addtags, A, "x"; check_inferred=false)
    test_rrule(addtags, A, "x"; fkwargs=(; plev=1), check_inferred=false)
    test_rrule(removetags, A, "i"; check_inferred=false)
    test_rrule(replacetags, A, "i" => "j"; check_inferred=false)
    test_rrule(replacetags, A, "i", "j"; check_inferred=false)
    test_rrule(settags, A, "x"; check_inferred=false)
    test_rrule(settags, A, "x"; fkwargs=(; plev=1), check_inferred=false)
    test_rrule(
      swaptags,
      randomITensor(Index(2, "i"), Index(2, "j")),
      "i" => "j";
      check_inferred=false,
    )
    test_rrule(
      swaptags, randomITensor(Index(2, "i"), Index(2, "j")), "i", "j"; check_inferred=false
    )
    test_rrule(replaceind, A, i' => sim(i); check_inferred=false)
    test_rrule(replaceind, A, i', sim(i); check_inferred=false)
    test_rrule(replaceinds, A, (i, i') => (sim(i), sim(i)); check_inferred=false)
    test_rrule(replaceinds, A, (i, i'), (sim(i), sim(i)); check_inferred=false)
    test_rrule(swapind, A, i', i; check_inferred=false)
    test_rrule(swapinds, A, (i',), (i,); check_inferred=false)
    test_rrule(itensor, randn(2, 2), i', i; check_inferred=false)
    test_rrule(itensor, randn(2, 2), [i', i]; check_inferred=false)
    test_rrule(itensor, randn(4), i', i; check_inferred=false)
    test_rrule(ITensor, randn(2, 2), i', i; check_inferred=false)
    test_rrule(ITensor, randn(2, 2), [i', i]; check_inferred=false)
    test_rrule(ITensor, randn(4), i', i; check_inferred=false)
    test_rrule(ITensor, 2.3; check_inferred=false)
    test_rrule(dag, A; check_inferred=false)
    test_rrule(permute, A, reverse(inds(A)); check_inferred=false)
  end

  @testset "apply, contract" begin
    test_rrule(ZygoteRuleConfig(), apply, A, V; rrule_f=rrule_via_ad, check_inferred=false)
    f = function (A, B)
      AT = ITensor(A, i, j)
      BT = ITensor(B, j, i)
      return (BT * AT)[1]
    end
    args = (rand(2, 2), rand(2, 2))
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    args = (rand(4), rand(4))
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    args = (rand(4), rand(2, 2))
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end

  @testset "contraction sequence" begin
    a, b, k, l, m, n, u, v = Index.([2, 3, 2, 3, 2, 3, 2, 3])
    args = (
      randomITensor(a, b, k),
      randomITensor(a, l, m),
      randomITensor(b, u, n),
      randomITensor(u, v),
      randomITensor(k, v),
      randomITensor(l, m, n),
    )
    f = (args...) -> contract([args...])[] # Left associative
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    seq = ITensors.optimal_contraction_sequence([args...])
    f = (args...) -> contract([args...]; sequence=seq)[] # sequence
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end

  @testset "construction and contraction" begin
    f = function (x)
      b = itensor([0, 0, 1, 1], i, j)
      k = itensor([0, 1, 0, 0], i, j)
      T = itensor([0 x x^2 1; 0 0 sin(x) 0; 0 cos(x) 0 exp(x); x 0 0 0], i', j', i, j)
      return x * real((b' * T * k)[])
    end
    args = (0.3,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

    #f = function (x)
    #  b = itensor([0, 0, 1, 1], i, j)
    #  k = itensor([0, 1, 0, 0], i, j)
    #  T = itensor([0 x x^2 1; 0 0 sin(x) 0; 0 cos(x) 0 exp(x); x 0 0 0], i, j, i', j')
    #  return x * real((b' * T * k)[])
    #end
    #args = (0.3,)
    #test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end

  @testset "scalar operations" begin
    f = x -> sin(scalar(x)^3)
    args = (C,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

    f = x -> sin(x[]^3)
    args = (C,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end

  @testset "adjoint" begin
    f = adjoint
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end

  @testset "contraction, priming, tagging + getindex" begin
    f = (x, y) -> (x * y)[1, 1]
    args = (A', A)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

    f = x -> prime(x, 2)[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

    f = x -> x'[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

    f = x -> addtags(x, "x")[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

    f = x -> (x' * x)[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

    f = x -> (prime(x) * x)[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

    f = x -> ((x'' * x') * x)[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

    f = x -> (x'' * (x' * x))[1, 1]
    args = (A,)

    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = (x, y, z) -> (x * y * z)[1, 1]
    args = (A'', A', A)

    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> (x'' * x' * x)[1, 1]
    args = (A,)

    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> (x''' * x'' * x' * x)[1, 1]
    args = (A,)

    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> (x''' * x'' * x' * x)[1, 1]
    args = (A,)

    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = (x, y) -> (x + y)[1, 1]
    args = (A, B)

    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> (x + x)[1, 1]
    args = (A,)

    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> (2x)[1, 1]
    args = (A,)

    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> (x + 2x)[1, 1]
    args = (A,)

    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

    f = x -> (x + 2 * mapprime(x' * x, 2 => 1))[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

    f = (x, y) -> (x * y)[]
    args = (A, δ(dag(inds(A))))
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

    f = x -> (x * x)[]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

    f = x -> (x * δ(dag(inds(x))))[]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end

  @testset "delta contractions" begin
    f = function (x)
      y = x' * x
      tr = δ(dag(inds(y)))
      return (y * tr)[]
    end
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

    f = function (x)
      y = x'' * x' * x
      tr = δ(dag(inds(y)))
      return (y * tr)[]
    end
    args = (A,)

    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> (x^2 * δ((i', i)))[1, 1]
    args = (6.2,)

    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> (x^2 * δ(i', i))[1, 1]
    args = (5.2,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end

  @testset "ITensor constructors" begin
    f = x -> itensor([x^2 x; x^3 x^4], i', i)
    args = (2.54,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

    f = x -> ITensor([x^2 x; x^3 x^4], i', i)
    args = (2.1,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

    f = x -> ITensor(x)
    args = (2.12,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end

  @testset "ITensor constructor and contraction" begin
    f = function (x)
      T = itensor([x^2 sin(x); x^2 exp(-2x)], j', dag(j))
      return real((dag(T) * T)[])
    end
    args = (2.8,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

    args = (2.8 + 3.1im,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

    f = function (x)
      v = itensor([exp(-3.2x), cos(2x^2)], j)
      T = itensor([x^2 sin(x); x^2 exp(-2x)], j', dag(j))
      return real((dag(v') * T * v)[])
    end
    args = (2.8,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    #args = (2.8 + 3.1im,)
    #test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = function (x)
      return real((x^3 * ITensor([sin(x) exp(-2x); 3x^3 x+x^2], j', dag(j)))[1, 1])
    end
    args = (3.4 + 2.3im,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end

  @testset "priming" begin
    f = x -> prime(permute(x, reverse(inds(x))))[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

    f = x -> prime(x; plev=1)[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end

  @testset "ITensor inner" begin
    W = itensor([1 1] / √2, i)
    f = x -> inner(W', exp(x), W)
    args = (A,)
    test_rrule(
      ZygoteRuleConfig(),
      f,
      args...;
      rrule_f=rrule_via_ad,
      check_inferred=false,
      rtol=1e-3,
      atol=1e-3,
    )

    f = x -> inner(V', exp(x), V)
    args = (A,)
    test_rrule(
      ZygoteRuleConfig(),
      f,
      args...;
      rrule_f=rrule_via_ad,
      check_inferred=false,
      rtol=1e-4,
      atol=1e-4,
    )
  end

  @testset "issue 933" begin
    # https://github.com/ITensor/ITensors.jl/issues/933
    f2 = function (x, a)
      y = a + im * x
      return real(dag(y) * y)[]
    end
    a = randomITensor()
    f_itensor = x -> f2(x, a)
    f_number = x -> f2(x, a[])
    x = randomITensor()
    @test f_number(x[]) ≈ f_itensor(x)
    @test f_number'(x[]) ≈ f_itensor'(x)[]
    @test isreal(f_itensor'(x))
  end

  @testset "issue 969" begin
    i = Index(2)
    j = Index(3)
    A = randomITensor(i)
    B = randomITensor(j)
    f = function (x, y)
      d = δ(ind(x, 1), ind(y, 1))
      return (x * d * y)[]
    end
    args = (A, B)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end
end

@testset "ChainRules rrules: op" begin
  s = siteinds("Qubit", 4)

  # RX
  args = (0.2,)
  for σ in [1, 2], σ′ in [1, 2]
    f = x -> op("Rx", s, 1; θ=x)[σ, σ′]
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end
  # RY
  args = (0.2,)
  for σ in [1, 2], σ′ in [1, 2]
    f = x -> op("Ry", s, 1; θ=x)[σ, σ′]
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end
  # RZ
  args = (0.2,)
  for σ in [1, 2], σ′ in [1, 2]
    f = x -> op("Rz", s, 1; ϕ=x)[σ, σ′]
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end
  # Rn
  args = (0.2, 0.3, 0.4)
  for σ in [1, 2], σ′ in [1, 2]
    f = x -> op("Rn", s, 1; θ=x[1], ϕ=x[2], λ=x[3])[σ, σ′]
    test_rrule(ZygoteRuleConfig(), f, args; rrule_f=rrule_via_ad, check_inferred=false)
  end

  basis = vec(collect(Iterators.product(fill([1, 2], 2)...)))
  # CRx
  args = (0.2,)
  for σ in basis, σ′ in basis
    f = x -> op("CRx", s, (1, 2); θ=x)[σ..., σ′...]
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end

  # CRy
  args = (0.2,)
  for σ in basis, σ′ in basis
    f = x -> op("CRy", s, (1, 2); θ=x)[σ..., σ′...]
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end

  # CRz
  args = (0.2,)
  for σ in basis, σ′ in basis
    f = x -> op("CRz", s, (1, 2); ϕ=x)[σ..., σ′...]
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end
  # Rn
  args = (0.2, 0.3, 0.4)
  for σ in basis, σ′ in basis
    f = x -> op("CRn", s, (1, 2); θ=x[1], ϕ=x[2], λ=x[3])[σ..., σ′...]
    test_rrule(ZygoteRuleConfig(), f, args; rrule_f=rrule_via_ad, check_inferred=false)
  end

  # Rxx
  args = (0.2,)
  for σ in basis, σ′ in basis
    f = x -> op("Rxx", s, (1, 2); ϕ=x)[σ..., σ′...]
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end
  # Ryy
  args = (0.2,)
  for σ in basis, σ′ in basis
    f = x -> op("Ryy", s, (1, 2); ϕ=x)[σ..., σ′...]
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end
  # Rzz
  args = (0.2,)
  for σ in basis, σ′ in basis
    f = x -> op("Rzz", s, (1, 2); ϕ=x)[σ..., σ′...]
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end

  # algebra with non-parametric gates
  args = (0.2,)
  # addition
  for σ in [1, 2], σ′ in [1, 2]
    f = x -> x * op("H + Y", s[1])[σ, σ′]
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end
  #subtraction
  for σ in [1, 2], σ′ in [1, 2]
    f = x -> x * op("H - Y", s[1])[σ, σ′]
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end
  # product
  for σ in [1, 2], σ′ in [1, 2]
    f = x -> x * op("H * Y", s[1])[σ, σ′]
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end
  # composite
  for σ in [1, 2], σ′ in [1, 2]
    f = x -> x * op("H + X * Y", s[1])[σ, σ′]
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end

  ## algebra with parametric gates
  #args = (0.2,)
  ## addition
  #for σ in [1, 2], σ′ in [1, 2]
  #  f = x -> x * op("H + Rx", s[1]; θ = x)[σ, σ′]
  #  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  #end
  ##subtraction
  #for σ in [1, 2], σ′ in [1, 2]
  #  f = x -> x * op("H - Rx", s[1]; θ = x)[σ, σ′]
  #  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  #end
  ### product
  #for σ in [1, 2], σ′ in [1, 2]
  #  f = x -> x * op("Rx * Y", s[1]; θ = x)[σ, σ′]
  #  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  #end
  ## composite
  #for σ in [1, 2], σ′ in [1, 2]
  #  f = x -> x * op("Rx * Y - Ry", s[1]; θ = x)[σ, σ′]
  #  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  #end
  #
  ## two-qubit composite algebra with parametric gate
  #args = (0.2,)
  #for σ in basis, σ′ in basis
  #  f = x -> op("Rxx + CX * CZ - Ryy", s, (1, 2); ϕ = x)[σ..., σ′...]
  #  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  #end

  # functions
  f = x -> exp(ITensor(Op("Ry", 1; θ=x), q))[1, 1]

  # RX
  args = (0.2,)
  for σ in [1, 2], σ′ in [1, 2]
    f = x -> exp(ITensor(Op("Rx", 1; θ=x), s))[σ, σ′]
    test_rrule(
      ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false, atol=1e-6
    )
  end

  # RY
  args = (0.2,)
  for σ in [1, 2], σ′ in [1, 2]
    f = x -> exp(ITensor(Op("Ry", 1; θ=x), s))[σ, σ′]
    test_rrule(
      ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false, atol=1e-6
    )
  end
end
