using ITensors
using Test

using ChainRulesCore: rrule_via_ad

include("utils/chainrulestestutils.jl")

using Zygote: ZygoteRuleConfig, gradient

@testset "ChainRules rrules: basic ITensor operations" begin
  i = Index(2, "i")
  A = randomITensor(i', dag(i))
  V = randomITensor(i)
  Ac = randomITensor(ComplexF64, i', dag(i))
  B = randomITensor(i', dag(i))
  C = ITensor(3.4)

  test_rrule(getindex, ITensor(3.4); check_inferred=false)
  test_rrule(getindex, A, 1, 2; check_inferred=false)
  test_rrule(*, A', A; check_inferred=false)
  test_rrule(*, 3.2, A; check_inferred=false)
  test_rrule(*, A, 4.3; check_inferred=false)
  test_rrule(+, A, B; check_inferred=false)
  test_rrule(prime, A; check_inferred=false)
  test_rrule(prime, A, 2; check_inferred=false)
  test_rrule(prime, A; fkwargs=(; tags="i"), check_inferred=false)
  test_rrule(prime, A; fkwargs=(; tags="x"), check_inferred=false)
  test_rrule(replaceprime, A, 1 => 2; check_inferred=false)
  test_rrule(swapprime, A, 0 => 1; check_inferred=false)
  test_rrule(addtags, A, "x"; check_inferred=false)
  test_rrule(addtags, A, "x"; fkwargs=(; plev=1), check_inferred=false)
  test_rrule(removetags, A, "i"; check_inferred=false)
  test_rrule(replacetags, A, "i" => "j"; check_inferred=false)
  test_rrule(
    swaptags, randomITensor(Index(2, "i"), Index(2, "j")), "i" => "j"; check_inferred=false
  )
  test_rrule(replaceind, A, i' => sim(i); check_inferred=false)
  test_rrule(replaceinds, A, (i, i') => (sim(i), sim(i)); check_inferred=false)
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

  test_rrule(ZygoteRuleConfig(), apply, A, V; rrule_f=rrule_via_ad, check_inferred=false)
  function f(A, B)
    i = Index(2)
    j = Index(2)
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

  f = function (x)
    j = Index(2, "j")
    b = itensor([0, 0, 1, 1], i, j)
    k = itensor([0, 1, 0, 0], i, j)
    T = itensor([0 x x^2 1; 0 0 sin(x) 0; 0 cos(x) 0 exp(x); x 0 0 0], i', j', i, j)
    return x * real((b' * T * k)[])
  end
  args = (0.3,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

  f = function (x)
    j = Index(2, "j")
    b = itensor([0, 0, 1, 1], i, j)
    k = itensor([0, 1, 0, 0], i, j)
    T = itensor([0 x x^2 1; 0 0 sin(x) 0; 0 cos(x) 0 exp(x); x 0 0 0], i, j, i', j')
    return x * real((b' * T * k)[])
  end
  args = (0.3,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)

  f = x -> sin(scalar(x)^3)
  args = (C,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> sin(x[]^3)
  args = (C,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  f = adjoint
  args = (A,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
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
  f = x -> itensor([x^2 x; x^3 x^4], i', i)
  args = (2.54,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> ITensor([x^2 x; x^3 x^4], i', i)
  args = (2.1,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> ITensor(x)
  args = (2.12,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  f = function (x)
    j = Index(2)
    T = itensor([x^2 sin(x); x^2 exp(-2x)], j', dag(j))
    return real((dag(T) * T)[])
  end
  args = (2.8,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  args = (2.8 + 3.1im,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  f = function f(x)
    j = Index(2)
    v = itensor([exp(-3.2x), cos(2x^2)], j)
    T = itensor([x^2 sin(x); x^2 exp(-2x)], j', dag(j))
    return real((dag(v') * T * v)[])
  end
  args = (2.8,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  args = (2.8 + 3.1im,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  f = function (x)
    j = Index(2)
    return real((x^3 * ITensor([sin(x) exp(-2x); 3x^3 x+x^2], j', dag(j)))[1, 1])
  end
  args = (3.4 + 2.3im,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> prime(permute(x, reverse(inds(x))))[1, 1]
  args = (A,)
  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> prime(x; plev=1)[1, 1]
  args = (A,)
  @test_throws ErrorException f'(args...)
end

@testset "MPS rrules" begin
  s = siteinds("S=1/2", 2)
  ψ = randomMPS(s)

  args = (ψ,)
  f = x -> inner(x', x')
  # TODO: Need to make MPS type compatible with FiniteDifferences.
  #test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  d_args = gradient(f, args...)
  @test norm(d_args[1] - 2 * args[1]) ≈ 0 atol = 1e-13

  args = (ψ,)
  f = x -> inner(prime(x), prime(x))
  # TODO: Need to make MPS type compatible with FiniteDifferences.
  #test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  d_args = gradient(f, args...)
  @test norm(d_args[1] - 2 * args[1]) ≈ 0 atol = 1e-13
end

@testset "ChainRules rrules: op" begin

  s = siteinds("Qubit", 4)
 
  # RX
  args = (0.2,)
  for σ in [1,2], σ′ in [1,2]
    f = x -> op("Rx", s, 1; θ=x)[σ, σ′]
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end
  # RY
  args = (0.2,)
  for σ in [1,2], σ′ in [1,2]
    f = x -> op("Ry", s, 1; θ=x)[σ, σ′]
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end
  # RZ
  args = (0.2,)
  for σ in [1,2], σ′ in [1,2]
    f = x -> op("Rz", s, 1; ϕ=x)[σ, σ′]
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  end
  # Rn
  args = (0.2,0.3,0.4)
  for σ in [1,2], σ′ in [1,2]
    f = x -> op("Rn", s, 1; θ = x[1], ϕ=x[2], λ=x[3])[σ, σ′]
    test_rrule(ZygoteRuleConfig(), f, args; rrule_f=rrule_via_ad, check_inferred=false)
  end

  # XXX: need #843
  #basis = vec(Iterators.product(fill([1,2],2)...)|>collect)
  ## CRx
  #args = (0.2,) 
  #for σ in basis, σ′ in basis
  #  f = x -> op("CRx", s, (1,2); θ = x)[σ...,σ′...]
  #  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  #end

  ## CRy
  #args = (0.2,) 
  #for σ in basis, σ′ in basis
  #  f = x -> op("CRy", s, (1,2); θ = x)[σ...,σ′...]
  #  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  #end

  ## CRz
  #args = (0.2,) 
  #for σ in basis, σ′ in basis
  #  f = x -> op("CRz", s, (1,2); ϕ = x)[σ...,σ′...]
  #  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  #end
  ## Rn
  #args = (0.2,0.3,0.4)
  #for σ in basis, σ′ in basis
  #  f = x -> op("CRn", s, (1,2); θ = x[1], ϕ=x[2], λ=x[3])[σ..., σ′...]
  #  test_rrule(ZygoteRuleConfig(), f, args; rrule_f=rrule_via_ad, check_inferred=false)
  #end
  #
  ## Rxx
  #args = (0.2,) 
  #for σ in basis, σ′ in basis
  #  f = x -> op("Rxx", s, (1,2); ϕ = x)[σ...,σ′...]
  #  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  #end
  ## Ryy
  #args = (0.2,) 
  #for σ in basis, σ′ in basis
  #  f = x -> op("Ryy", s, (1,2); ϕ = x)[σ...,σ′...]
  #  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  #end
  ## Rzz
  #args = (0.2,) 
  #for σ in basis, σ′ in basis
  #  f = x -> op("Rzz", s, (1,2); ϕ = x)[σ...,σ′...]
  #  test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
  #end
end

@testset "MPS ($ElType)" for ElType in (Float64, ComplexF64)
  n = 4
  s = siteinds("S=1/2", n; conserve_qns=true)
  function heisenberg(n)
    os = OpSum()
    for j in 1:(n - 1)
      os += 0.5, "S+", j, "S-", j + 1
      os += 0.5, "S-", j, "S+", j + 1
      os += "Sz", j, "Sz", j + 1
    end
    return os
  end
  H = MPO(heisenberg(n), s)
  ψ = randomMPS(s, n -> isodd(n) ? "Up" : "Dn"; linkdims=2)

  f = x -> inner(x, x)
  args = (ψ,)
  d_args = gradient(f, args...)
  @test norm(d_args[1] - 2 * args[1]) ≈ 0 atol = 1e-13

  f = x -> inner(x', H, x)
  args = (ψ,)
  d_args = gradient(f, args...)
  @test norm(d_args[1]' - 2 * H * args[1]) ≈ 0 atol = 1e-13

  f = x -> inner(x', x)
  args = (ψ,)
  @test_throws ErrorException gradient(f, args...)

  f = x -> inner(x, H, x)
  args = (ψ,)
  @test_throws ErrorException gradient(f, args...)
end
