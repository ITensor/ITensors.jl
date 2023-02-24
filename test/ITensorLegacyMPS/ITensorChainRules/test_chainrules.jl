using ITensors
using Random
using Test
using Zygote

Random.seed!(1234)

@testset "ChainRules/Zygote AD tests for MPS/MPO" begin
  @testset "issue 936" begin
    # https://github.com/ITensor/ITensors.jl/issues/936
    n = 2
    s = siteinds("S=1/2", n)
    x = (x -> outer(x', x))(randomMPS(s))
    f1 = x -> tr(x)
    f2 = x -> 2tr(x)
    f3 = x -> -tr(x)
    @test f1'(x) ≈ MPO(s, "I")
    @test f2'(x) ≈ 2MPO(s, "I")
    @test f3'(x) ≈ -MPO(s, "I")
  end

  @testset "MPS ($ElType)" for ElType in (Float64, ComplexF64)
    Random.seed!(1234)
    n = 4
    ϵ = 1e-8
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

    # apply on MPS 
    s = siteinds("S=1/2", n)
    ϕ = randomMPS(ElType, s)
    ψ = randomMPS(ElType, s)
    f = function (x)
      U = [op("Ry", s[2]; θ=x), op("CX", s[1], s[2]), op("Rx", s[3]; θ=x)]
      ψθ = apply(U, ψ)
      return abs2(inner(ϕ, ψθ))
    end
    θ = 0.5
    ∇f = f'(θ)
    ∇num = (f(θ + ϵ) - f(θ)) / ϵ
    @test ∇f ≈ ∇num atol = 1e-5
  end

  @testset "MPS rrules" begin
    Random.seed!(1234)
    s = siteinds("S=1/2", 4)
    ψ = randomMPS(s)
    args = (ψ,)
    f = x -> inner(x, x)
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

    ψ = randomMPS(ComplexF64, s)
    ψtensors = ITensors.data(ψ)
    ϕ = randomMPS(ComplexF64, s)
    f = function (x)
      ψ̃tensors = [x^j * ψtensors[j] for j in 1:length(ψtensors)]
      ψ̃ = MPS(ψ̃tensors)
      return abs2(inner(ϕ, ψ̃))
    end
    x = 0.5
    ϵ = 1e-10
    @test f'(x) ≈ (f(x + ϵ) - f(x)) / ϵ atol = 1e-6

    ρ = randomMPO(s)
    f = function (x)
      ψ̃tensors = [x^j * ψtensors[j] for j in 1:length(ψtensors)]
      ψ̃ = MPS(ψ̃tensors)
      return real(inner(ψ̃', ρ, ψ̃))
    end
    @test f'(x) ≈ (f(x + ϵ) - f(x)) / ϵ atol = 1e-6
  end

  #@testset "MPO rules" begin
  #  Random.seed!(1234)
  #  s = siteinds("S=1/2", 2)
  #  
  #  #ρ = randomMPO(s)
  #  #ρtensors = ITensors.data(ρ)
  #  #ϕ = randomMPS(ComplexF64, s)
  #  #f = function (x)
  #  #  ρ̃tensors  = [2 * x * ρtensors[1],  log(x) * ρtensors[2]] 
  #  #  ρ̃ = MPO(ρ̃tensors)
  #  #  #@show typeof(ρ̃)
  #  #  return real(inner(ϕ', ρ̃, ϕ))
  #  #end
  #  #x = 3.0
  #  #ϵ = 1e-8
  #  #@show (f(x+ϵ) - f(x)) / ϵ
  #  #@show f'(x)
  #  ##@test f'(x) ≈ (f(x+ϵ) - f(x)) / ϵ atol = 1e-6 
  #  #
  #
  #  #ϕ = randomMPO(s)
  #  #f = function (x)
  #  #  ψ̃tensors  = [2 * x * ψtensors[1],  log(x) * ψtensors[2]] 
  #  #  ψ̃ = MPS(ψ̃tensors)
  #  #  return abs2(inner(ϕ, ψ̃))
  #  #end
  #  #x = 3.0
  #  #ϵ = 1e-8
  #  #@test f'(x) ≈ (f(x+ϵ) - f(x)) / ϵ atol = 1e-6 
  #
  #  #ρ = randomMPO(s)
  #end
  @testset "MPO: apply" begin
    Random.seed!(1234)
    ϵ = 1e-8
    n = 3
    s = siteinds("Qubit", n)
    function ising(n, h)
      os = OpSum()
      for j in 1:(n - 1)
        os += -1, "Z", j, "Z", j + 1
        os += -h, "X", j
      end
      os += -h, "X", n
      return os
    end
    H = MPO(ising(n, 1.0), s)

    # apply on MPO with apply_dag=true
    ϕ = randomMPS(ComplexF64, s; linkdims=10)
    f = function (x)
      U = [op("Ry", s[2]; θ=x), op("CX", s[1], s[2]), op("Rx", s[3]; θ=x)]
      Hθ = apply(U, H; apply_dag=true)
      return real(inner(ϕ', Hθ, ϕ))
    end
    θ = 0.5
    ∇f = f'(θ)
    ∇num = (f(θ + ϵ) - f(θ)) / ϵ
    @test ∇f ≈ ∇num atol = 1e-5

    # apply on MPO with apply_dag=false
    f = function (x)
      U = [op("Ry", s[2]; θ=x), op("CX", s[1], s[2]), op("Rx", s[3]; θ=x)]
      Hθ = apply(U, H; apply_dag=false)
      return real(inner(ϕ', Hθ, ϕ))
    end
    θ = 0.5
    ∇f = f'(θ)
    ∇num = (f(θ + ϵ) - f(θ)) / ϵ
    @test ∇f ≈ ∇num atol = 1e-5

    # multiply two MPOs
    V = randomMPO(s)
    f = function (x)
      U = [op("Ry", s[2]; θ=x), op("CX", s[1], s[2]), op("Rx", s[3]; θ=x)]
      Hθ = apply(U, H; apply_dag=false)
      X = replaceprime(V' * Hθ, 2 => 1)
      return real(inner(ϕ', X, ϕ))
    end

    θ = 0.5
    ∇f = f'(θ)
    ∇num = (f(θ + ϵ) - f(θ)) / ϵ
    @test ∇f ≈ ∇num atol = 1e-5

    # trace(MPO) 
    V1 = randomMPO(s)
    V2 = randomMPO(s)
    f = function (x)
      U = [op("Ry", s[2]; θ=x), op("CX", s[1], s[2]), op("Rx", s[3]; θ=x)]
      Hθ = apply(U, H; apply_dag=false)
      X = V1''' * Hθ'' * V2' * Hθ
      return real(tr(X; plev=4 => 0))
    end

    θ = 0.5
    ∇f = f'(θ)
    ∇num = (f(θ + ϵ) - f(θ)) / ϵ
    @test ∇f ≈ ∇num atol = 1e-5
  end

  @testset "contract/apply MPOs" begin
    n = 2
    s = siteinds("S=1/2", n)
    x = (x -> outer(x', x))(randomMPS(s; linkdims=4))
    x_itensor = contract(x)

    f = x -> tr(apply(x, x))
    @test f(x) ≈ f(x_itensor)
    @test contract(f'(x)) ≈ f'(x_itensor)

    f = x -> tr(replaceprime(contract(x', x), 2 => 1))
    @test f(x) ≈ f(x_itensor)
    @test contract(f'(x)) ≈ f'(x_itensor)

    f = x -> tr(replaceprime(*(x', x), 2 => 1))
    @test f(x) ≈ f(x_itensor)
    @test contract(f'(x)) ≈ f'(x_itensor)
  end

  @testset "contract/apply MPOs on MPSs" begin
    n = 2
    s = siteinds("S=1/2", n)
    x = (x -> outer(x', x))(randomMPS(s; linkdims=4))
    x_itensor = contract(x)
    y = randomMPS(s; linkdims=4)
    y_itensor = contract(y)

    f = x -> inner(apply(x, y), apply(x, y))
    g = x -> inner(apply(x, y_itensor), apply(x, y_itensor))
    @test f(x) ≈ g(x_itensor)
    @test contract(f'(x)) ≈ g'(x_itensor)

    f = y -> inner(apply(x, y), apply(x, y))
    g = y -> inner(apply(x_itensor, y), apply(x_itensor, y))
    @test f(y) ≈ g(y_itensor)
    @test contract(f'(y)) ≈ g'(y_itensor)

    f =
      x -> inner(replaceprime(contract(x, y), 2 => 1), replaceprime(contract(x, y), 2 => 1))
    g =
      x -> inner(
        replaceprime(contract(x, y_itensor), 2 => 1),
        replaceprime(contract(x, y_itensor), 2 => 1),
      )
    @test f(x) ≈ g(x_itensor)
    @test contract(f'(x)) ≈ g'(x_itensor)

    f =
      y -> inner(replaceprime(contract(x, y), 2 => 1), replaceprime(contract(x, y), 2 => 1))
    g =
      y -> inner(
        replaceprime(contract(x_itensor, y), 2 => 1),
        replaceprime(contract(x_itensor, y), 2 => 1),
      )
    @test f(y) ≈ g(y_itensor)
    @test contract(f'(y)) ≈ g'(y_itensor)

    f = x -> inner(replaceprime(*(x, y), 2 => 1), replaceprime(*(x, y), 2 => 1))
    g =
      x ->
        inner(replaceprime(*(x, y_itensor), 2 => 1), replaceprime(*(x, y_itensor), 2 => 1))
    @test f(x) ≈ g(x_itensor)
    @test contract(f'(x)) ≈ g'(x_itensor)

    f = y -> inner(replaceprime(*(x, y), 2 => 1), replaceprime(*(x, y), 2 => 1))
    g =
      y ->
        inner(replaceprime(*(x_itensor, y), 2 => 1), replaceprime(*(x_itensor, y), 2 => 1))
    @test f(y) ≈ g(y_itensor)
    @test contract(f'(y)) ≈ g'(y_itensor)
  end
  @testset "Calling apply multiple times (ITensors #924 regression test)" begin
    n = 1
    θ = 3.0
    p = 2

    s = siteinds("S=1/2", n)

    ψ₀ₘₚₛ = MPS(s, "↑")
    ψ₀ = contract(ψ₀ₘₚₛ)

    U(θ) = [θ * op("Z", s, 1)]

    function f(θ, ψ)
      ψθ = ψ
      Uθ = U(θ)
      for _ in 1:p
        ψθ = apply(Uθ, ψθ)
      end
      return inner(ψ, ψθ)
    end

    function g(θ, ψ)
      Uθ = U(θ)
      Utot = Uθ
      for _ in 2:p
        Utot = [Utot; Uθ]
      end
      ψθ = apply(Utot, ψ)
      return inner(ψ, ψθ)
    end

    f_itensor(θ) = f(θ, ψ₀)
    f_mps(θ) = f(θ, ψ₀ₘₚₛ)
    g_itensor(θ) = g(θ, ψ₀)
    g_mps(θ) = g(θ, ψ₀ₘₚₛ)

    @test f_itensor(θ) ≈ θ^p
    @test f_mps(θ) ≈ θ^p
    @test f_itensor'(θ) ≈ p * θ^(p - 1)
    @test f_mps'(θ) ≈ p * θ^(p - 1)
    @test g_itensor(θ) ≈ θ^p
    @test g_mps(θ) ≈ θ^p
    @test g_itensor'(θ) ≈ p * θ^(p - 1)
    @test g_mps'(θ) ≈ p * θ^(p - 1)
  end
end
