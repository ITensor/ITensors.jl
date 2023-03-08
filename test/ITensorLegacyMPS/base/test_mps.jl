using Combinatorics
using ITensors
using Random
using LinearAlgebra: diag
using Test

Random.seed!(1234)

include(joinpath(@__DIR__, "utils", "util.jl"))

@testset "MPS Basics" begin
  sites = [Index(2, "Site") for n in 1:10]
  psi = MPS(sites)
  @test length(psi) == length(psi)
  @test length(MPS()) == 0
  @test linkdims(psi) == fill(1, length(psi) - 1)
  @test isnothing(flux(psi))

  psi = MPS(sites; linkdims=3)
  @test length(psi) == length(psi)
  @test length(MPS()) == 0
  @test linkdims(psi) == fill(3, length(psi) - 1)
  @test isnothing(flux(psi))

  str = split(sprint(show, psi), '\n')
  @test str[1] == "MPS"
  @test length(str) == length(psi) + 2

  @test siteind(psi, 2) == sites[2]
  @test findfirstsiteind(psi, sites[2]) == 2
  @test findfirstsiteind(psi, sites[4]) == 4
  @test findfirstsiteinds(psi, IndexSet(sites[5])) == 5
  @test hasind(psi[3], linkind(psi, 2))
  @test hasind(psi[3], linkind(psi, 3))

  @test isnothing(linkind(psi, length(psi)))
  @test isnothing(linkind(psi, length(psi) + 1))
  @test isnothing(linkind(psi, 0))
  @test isnothing(linkind(psi, -1))
  @test linkind(psi, 3) == commonind(psi[3], psi[4])

  psi[1] = ITensor(sites[1])
  @test hasind(psi[1], sites[1])

  @testset "N=1 MPS" begin
    sites1 = [Index(2, "Site,n=1")]
    psi = MPS(sites1)
    @test length(psi) == 1
    @test siteind(psi, 1) == sites1[1]
    @test siteinds(psi)[1] == sites1[1]
  end

  @testset "Missing links" begin
    psi = MPS([randomITensor(sites[i]) for i in 1:10])
    @test isnothing(linkind(psi, 1))
    @test isnothing(linkind(psi, 5))
    @test isnothing(linkind(psi, length(psi)))
    @test maxlinkdim(psi) == 1
    @test psi ⋅ psi ≈ *(dag(psi)..., psi...)[]
  end

  @testset "productMPS" begin
    @testset "vector of string input" begin
      sites = siteinds("S=1/2", 10)
      state = fill("", length(sites))
      for j in 1:length(sites)
        state[j] = isodd(j) ? "Up" : "Dn"
      end
      psi = MPS(sites, state)
      for j in 1:length(psi)
        sign = isodd(j) ? +1.0 : -1.0
        @test (psi[j] * op(sites, "Sz", j) * dag(prime(psi[j], "Site")))[] ≈ sign / 2
      end
      psi = productMPS(sites, state)
      for j in 1:length(psi)
        sign = isodd(j) ? +1.0 : -1.0
        @test (psi[j] * op(sites, "Sz", j) * dag(prime(psi[j], "Site")))[] ≈ sign / 2
      end
      @test_throws DimensionMismatch MPS(sites, fill("", length(psi) - 1))
      @test_throws DimensionMismatch productMPS(sites, fill("", length(psi) - 1))
    end

    @testset "String input" begin
      sites = siteinds("S=1/2", 10)
      psi = MPS(sites, "Dn")
      for j in 1:length(psi)
        sign = -1.0
        @test (psi[j] * op(sites, "Sz", j) * dag(prime(psi[j], "Site")))[] ≈ sign / 2
      end
      psi = productMPS(sites, "Dn")
      for j in 1:length(psi)
        sign = -1.0
        @test (psi[j] * op(sites, "Sz", j) * dag(prime(psi[j], "Site")))[] ≈ sign / 2
      end

      psi = MPS(sites, "X+")
      for j in 1:length(psi)
        @test (psi[j] * op(sites, "X", j) * dag(prime(psi[j], "Site")))[] ≈ 1.0
      end
    end

    @testset "Int input" begin
      sites = siteinds("S=1/2", 10)
      psi = MPS(sites, 2)
      for j in 1:length(psi)
        sign = -1.0
        @test (psi[j] * op(sites, "Sz", j) * dag(prime(psi[j], "Site")))[] ≈ sign / 2
      end
      psi = productMPS(sites, 2)
      for j in 1:length(psi)
        sign = -1.0
        @test (psi[j] * op(sites, "Sz", j) * dag(prime(psi[j], "Site")))[] ≈ sign / 2
      end
    end

    @testset "vector of int input" begin
      sites = siteinds("S=1/2", 10)
      state = fill(0, length(sites))
      for j in 1:length(sites)
        state[j] = isodd(j) ? 1 : 2
      end
      psi = MPS(sites, state)
      for j in 1:length(psi)
        sign = isodd(j) ? +1.0 : -1.0
        @test (psi[j] * op(sites, "Sz", j) * dag(prime(psi[j], "Site")))[] ≈ sign / 2
      end
      psi = productMPS(sites, state)
      for j in 1:length(psi)
        sign = isodd(j) ? +1.0 : -1.0
        @test (psi[j] * op(sites, "Sz", j) * dag(prime(psi[j], "Site")))[] ≈ sign / 2
      end
    end

    @testset "vector of ivals input" begin
      sites = siteinds("S=1/2", 10)
      states = fill(0, length(sites))
      for j in 1:length(sites)
        states[j] = isodd(j) ? 1 : 2
      end
      ivals = [sites[n] => states[n] for n in 1:length(sites)]
      psi = MPS(ivals)
      for j in 1:length(psi)
        sign = isodd(j) ? +1.0 : -1.0
        @test (psi[j] * op(sites, "Sz", j) * dag(prime(psi[j], "Site")))[] ≈ sign / 2
      end
      psi = productMPS(ivals)
      for j in 1:length(psi)
        sign = isodd(j) ? +1.0 : -1.0
        @test (psi[j] * op(sites, "Sz", j) * dag(prime(psi[j], "Site")))[] ≈ sign / 2
      end

      @testset "ComplexF64 eltype" begin
        sites = siteinds("S=1/2", 10)
        psi = MPS(ComplexF64, sites, fill(1, length(sites)))
        for j in 1:length(psi)
          @test eltype(psi[j]) == ComplexF64
        end
        psi = productMPS(ComplexF64, sites, fill(1, length(psi)))
        for j in 1:length(psi)
          @test eltype(psi[j]) == ComplexF64
        end
        @test eltype(psi) == ITensor
        @test ITensors.promote_itensor_eltype(psi) == ComplexF64
      end
    end

    @testset "N=1 case" begin
      site = Index(2, "Site,n=1")
      psi = MPS([site], [1])
      @test psi[1][1] ≈ 1.0
      @test psi[1][2] ≈ 0.0
      psi = productMPS([site], [1])
      @test psi[1][1] ≈ 1.0
      @test psi[1][2] ≈ 0.0
      psi = MPS([site], [2])
      @test psi[1][1] ≈ 0.0
      @test psi[1][2] ≈ 1.0
      psi = productMPS([site], [2])
      @test psi[1][1] ≈ 0.0
      @test psi[1][2] ≈ 1.0
    end
  end

  @testset "randomMPS with chi==1" begin
    phi = randomMPS(sites)
    phic = randomMPS(ComplexF64, sites)

    @test maxlinkdim(phi) == 1
    @test maxlinkdim(phic) == 1

    @test hasind(phi[1], sites[1])
    @test norm(phi[1]) ≈ 1.0
    @test norm(phic[1]) ≈ 1.0

    @test hasind(phi[4], sites[4])
    @test norm(phi[4]) ≈ 1.0
    @test norm(phic[4]) ≈ 1.0
  end

  @testset "randomMPS with chi>1" for linkdims in [1, 4]
    phi = randomMPS(Float32, sites; linkdims)
    @test LinearAlgebra.promote_leaf_eltypes(phi) === Float32
    @test all(x -> eltype(x) === Float32, phi)
    @test maxlinkdim(phi) == linkdims
    phic = randomMPS(ComplexF32, sites; linkdims)
    @test LinearAlgebra.promote_leaf_eltypes(phic) === ComplexF32
    @test maxlinkdim(phic) == linkdims
    @test all(x -> eltype(x) === ComplexF32, phic)
  end

  @testset "randomMPS with nonuniform dimensions" begin
    _linkdims = [2, 3, 4, 2, 4, 3, 2, 2, 2]
    phi = randomMPS(sites; linkdims=_linkdims)
    @test linkdims(phi) == _linkdims
  end

  @testset "QN randomMPS" begin
    s = siteinds("S=1/2", 5; conserve_qns=true)
    ψ = randomMPS(s, n -> isodd(n) ? "↑" : "↓"; linkdims=2)
    @test linkdims(ψ) == [2, 2, 2, 2]
    ψ = randomMPS(s, n -> isodd(n) ? "↑" : "↓"; linkdims=[2, 3, 2, 2])
    @test linkdims(ψ) == [2, 3, 2, 2]
  end

  @testset "inner different MPS" begin
    phi = randomMPS(sites)
    psi = randomMPS(sites)
    phipsi = dag(phi[1]) * psi[1]
    for j in 2:length(psi)
      phipsi *= dag(phi[j]) * psi[j]
    end
    @test phipsi[] ≈ inner(phi, psi)

    badsites = [Index(2) for n in 1:(length(psi) + 1)]
    badpsi = randomMPS(badsites)
    @test_throws DimensionMismatch inner(phi, badpsi)
  end

  @testset "loginner" begin
    n = 4
    c = 2

    s = siteinds("S=1/2", n)
    ψ = c .* randomMPS(s; linkdims=4)
    @test exp(loginner(ψ, ψ)) ≈ c^(2n)
    @test exp(loginner(ψ, -ψ)) ≈ -c^(2n)

    α = randn(ComplexF64)
    @test exp(loginner(ψ, α * ψ)) ≈ α * c^(2n)
  end

  @testset "broadcasting" begin
    psi = randomMPS(sites)
    orthogonalize!(psi, 1)
    @test ortho_lims(psi) == 1:1
    @test dim.(psi) == fill(2, length(psi))
    psi′ = prime.(psi)
    @test ortho_lims(psi′) == 1:length(psi′)
    @test ortho_lims(psi) == 1:1
    for n in 1:length(psi)
      @test prime(psi[n]) == psi′[n]
    end
    psi_copy = copy(psi)
    psi_copy .= addtags(psi_copy, "x")
    @test ortho_lims(psi_copy) == 1:length(psi_copy)
    @test ortho_lims(psi) == 1:1
    for n in 1:length(psi)
      @test addtags(psi[n], "x") == psi_copy[n]
    end
  end

  @testset "copy and deepcopy" begin
    s = siteinds("S=1/2", 3)
    M1 = randomMPS(s; linkdims=3)
    @test norm(M1) ≈ 1

    M2 = deepcopy(M1)
    M2[1] .*= 2 # Modifies the tensor data
    @test norm(M1) ≈ 1
    @test norm(M2) ≈ 2

    M3 = copy(M1)
    M3[1] *= 3
    @test norm(M1) ≈ 1
    @test norm(M3) ≈ 3

    M4 = copy(M1)
    M4[1] .*= 4
    @test norm(M1) ≈ 4
    @test norm(M4) ≈ 4
  end

  @testset "inner same MPS" begin
    psi = randomMPS(sites)
    psidag = dag(psi)
    #ITensors.prime_linkinds!(psidag)
    psipsi = psidag[1] * psi[1]
    for j in 2:length(psi)
      psipsi *= psidag[j] * psi[j]
    end
    @test psipsi[] ≈ inner(psi, psi)
  end

  @testset "norm MPS" begin
    psi = randomMPS(sites; linkdims=10)
    psidag = sim(linkinds, dag(psi))
    psi² = ITensor(1)
    for j in 1:length(psi)
      psi² *= psidag[j] * psi[j]
    end
    @test psi²[] ≈ psi ⋅ psi
    @test sqrt(psi²[]) ≈ norm(psi)

    psi = randomMPS(sites; linkdims=10)
    psi .*= 1:length(psi)
    @test norm(psi) ≈ factorial(length(psi))

    psi = randomMPS(sites; linkdims=10)
    for j in 1:length(psi)
      psi[j] .*= j
    end
    # This fails because it modifies the MPS ITensors
    # directly, which ruins the orthogonality
    @test norm(psi) ≉ factorial(length(psi))
    reset_ortho_lims!(psi)
    @test norm(psi) ≈ factorial(length(psi))

    # Test complex
    psi = randomMPS(ComplexF64, sites; linkdims=10)

    norm_psi = norm(psi)
    @test norm_psi ≈ 1
    @test isreal(norm_psi)

    lognorm_psi = lognorm(psi)
    @test lognorm_psi ≈ 0 atol = 1e-15
    @test isreal(lognorm_psi)

    psi = psi .* 2

    norm_psi = norm(psi)
    @test norm_psi ≈ 2^length(psi)
    @test isreal(norm_psi)

    lognorm_psi = lognorm(psi)
    @test lognorm_psi ≈ log(2) * length(psi)
    @test isreal(lognorm_psi)
  end

  @testset "lognorm checking real tolerance error regression test" begin
    # Test that lognorm doesn't throw an error when the norm isn't real
    # up to a certain tolerance
    Random.seed!(1234)
    s = siteinds("S=1/2", 10)
    ψ = randomMPS(ComplexF64, s; linkdims=4)
    reset_ortho_lims!(ψ)
    @test exp(lognorm(ψ)) ≈ 1
  end

  @testset "normalize/normalize! MPS" begin
    psi = randomMPS(sites; linkdims=10)

    @test norm(psi) ≈ 1
    @test norm(normalize(psi)) ≈ 1

    α = 3.5
    phi = α * psi
    @test norm(phi) ≈ α
    @test norm(normalize(phi)) ≈ 1
    @test norm(psi) ≈ 1
    @test inner(phi, psi) ≈ α

    normalize!(phi)
    @test norm(phi) ≈ 1
    @test norm(normalize(phi)) ≈ 1
    @test norm(psi) ≈ 1
    @test inner(phi, psi) ≈ 1

    # Zero norm
    @test norm(0phi) == 0
    @test lognorm(0phi) == -Inf

    zero_phi = 0phi
    lognorm_zero_phi = []
    normalize!(zero_phi; (lognorm!)=lognorm_zero_phi)
    @test lognorm_zero_phi[1] == -Inf
    @test norm(zero_phi) == 0
    @test norm(normalize(0phi)) == 0

    # Large number of sites
    psi = randomMPS(siteinds("S=1/2", 1_000); linkdims=10)

    @test norm(psi) ≈ 1.0
    @test lognorm(psi) ≈ 0.0 atol = 1e-15

    α = 2
    phi = α .* psi

    @test isnan(norm(phi))
    @test lognorm(phi) ≈ length(psi) * log(α)

    phi = normalize(phi)

    @test norm(phi) ≈ 1

    # Test scaling only a subset of sites
    psi = randomMPS(siteinds("S=1/2", 10); linkdims=10)

    @test norm(psi) ≈ 1.0
    @test lognorm(psi) ≈ 0.0 atol = 1e-15

    α = 2
    r = (length(psi) ÷ 2 - 1):(length(psi) ÷ 2 + 1)
    phi = copy(psi)
    for n in r
      phi[n] = α * psi[n]
    end

    @test norm(phi) ≈ α^length(r)
    @test lognorm(phi) ≈ length(r) * log(α)

    phi = normalize(phi)

    @test norm(phi) ≈ 1

    # Output the lognorm
    α = 2
    psi = randomMPS(siteinds("S=1/2", 30); linkdims=10)
    psi = α .* psi
    @test norm(psi) ≈ α^length(psi)
    @test lognorm(psi) ≈ length(psi) * log(α)
    lognorm_psi = Float64[]
    phi = normalize(psi; (lognorm!)=lognorm_psi)
    @test lognorm_psi[end] ≈ lognorm(psi)
    @test norm(phi) ≈ 1
    @test lognorm(phi) ≈ 0 atol = 1e-14
  end

  @testset "lognorm MPS" begin
    psi = randomMPS(sites; linkdims=10)
    for j in eachindex(psi)
      psi[j] .*= j
    end
    psidag = sim(linkinds, dag(psi))
    psi² = ITensor(1)
    for j in eachindex(psi)
      psi² *= psidag[j] * psi[j]
    end
    @test psi²[] ≈ psi ⋅ psi
    @test 0.5 * log(psi²[]) ≉ lognorm(psi)
    @test lognorm(psi) ≉ log(factorial(length(psi)))
    # Need to manually change the orthogonality
    # limits back to 1:length(psi)
    reset_ortho_lims!(psi)
    @test 0.5 * log(psi²[]) ≈ lognorm(psi)
    @test lognorm(psi) ≈ log(factorial(length(psi)))
  end

  @testset "scaling MPS" begin
    psi = randomMPS(sites; linkdims=4)
    twopsidag = 2.0 * dag(psi)
    #ITensors.prime_linkinds!(twopsidag)
    @test inner(twopsidag, psi) ≈ 2.0 * inner(psi, psi)
  end

  @testset "flip sign of MPS" begin
    psi = randomMPS(sites)
    minuspsidag = -dag(psi)
    #ITensors.primelinkinds!(minuspsidag)
    @test inner(minuspsidag, psi) ≈ -inner(psi, psi)
  end

  @testset "add MPS" begin
    psi = randomMPS(sites)
    phi = deepcopy(psi)
    xi = add(psi, phi)
    @test inner(xi, xi) ≈ 4.0 * inner(psi, psi)
    # sum of many MPSs
    Ks = [randomMPS(sites) for i in 1:3]
    K12 = add(Ks[1], Ks[2])
    K123 = add(K12, Ks[3])
    @test inner(sum(Ks), K123) ≈ inner(K123, K123)
    # https://github.com/ITensor/ITensors.jl/issues/1000
    @test sum([psi]) ≈ psi
  end

  @testset "+ MPS" begin
    psi = randomMPS(sites)
    phi = deepcopy(psi)
    xi = psi + phi
    @test inner(xi, xi) ≈ 4.0 * inner(psi, psi)
    # sum of many MPSs
    Ks = [randomMPS(sites) for i in 1:3]
    K12 = Ks[1] + Ks[2]
    K123 = K12 + Ks[3]
    @test inner(sum(Ks), K123) ≈ inner(K123, K123)

    χ1 = 2
    χ2 = 3
    ψ1 = randomMPS(sites; linkdims=χ1)
    ψ2 = 0.0 * randomMPS(sites; linkdims=χ2)

    ϕ1 = +(ψ1, ψ2; alg="densitymatrix", cutoff=nothing)
    for j in 2:7
      @test linkdim(ϕ1, j) == χ1 + χ2
    end
    @test inner(ϕ1, ψ1) + inner(ϕ1, ψ2) ≈ inner(ϕ1, ϕ1)

    ϕ2 = +(ψ1, ψ2; alg="directsum")
    for j in 1:8
      @test linkdim(ϕ2, j) == χ1 + χ2
    end
    @test inner(ϕ2, ψ1) + inner(ϕ2, ψ2) ≈ inner(ϕ2, ϕ2)
  end

  @testset "+ MPS with coefficients" begin
    Random.seed!(1234)

    conserve_qns = true

    s = siteinds("S=1/2", 20; conserve_qns=conserve_qns)
    state = n -> isodd(n) ? "↑" : "↓"

    ψ₁ = randomMPS(s, state; linkdims=4)
    ψ₂ = randomMPS(s, state; linkdims=4)
    ψ₃ = randomMPS(s, state; linkdims=4)

    ψ = ψ₁ + ψ₂

    @test inner(ψ, ψ) ≈ inner_add(ψ₁, ψ₂)
    @test maxlinkdim(ψ) ≤ maxlinkdim(ψ₁) + maxlinkdim(ψ₂)

    ψ = +(ψ₁, ψ₂; cutoff=0.0)

    @test_throws ErrorException ψ₁ + ψ₂'

    @test inner(ψ, ψ) ≈ inner_add(ψ₁, ψ₂)
    @test maxlinkdim(ψ) ≤ maxlinkdim(ψ₁) + maxlinkdim(ψ₂)

    ψ = ψ₁ + (-ψ₂)

    @test inner(ψ, ψ) ≈ inner_add((1, ψ₁), (-1, ψ₂))
    @test maxlinkdim(ψ) ≤ maxlinkdim(ψ₁) + maxlinkdim(ψ₂)

    α₁ = 2.2
    α₂ = -4.1
    ψ = +(α₁ * ψ₁, α₂ * ψ₂; cutoff=1e-8)

    @test inner(ψ, ψ) ≈ inner_add((α₁, ψ₁), (α₂, ψ₂))
    @test maxlinkdim(ψ) ≤ maxlinkdim(ψ₁) + maxlinkdim(ψ₂)

    α₁ = 2 + 3im
    α₂ = -4 + 1im
    ψ = α₁ * ψ₁ + α₂ * ψ₂

    @test inner(ψ, ψ) ≈ inner_add((α₁, ψ₁), (α₂, ψ₂))
    @test maxlinkdim(ψ) ≤ maxlinkdim(ψ₁) + maxlinkdim(ψ₂)

    α₁ = 2 + 3im
    α₂ = -4 + 1im
    ψ = α₁ * ψ₁ + α₂ * ψ₂ + ψ₃

    @test inner(ψ, ψ) ≈ inner_add((α₁, ψ₁), (α₂, ψ₂), ψ₃)
    @test maxlinkdim(ψ) ≤ maxlinkdim(ψ₁) + maxlinkdim(ψ₂) + maxlinkdim(ψ₃)

    ψ = ψ₁ - ψ₂

    @test inner(ψ, ψ) ≈ inner_add(ψ₁, (-1, ψ₂))
    @test maxlinkdim(ψ) ≤ maxlinkdim(ψ₁) + maxlinkdim(ψ₂)
  end

  sites = siteinds(2, 10)
  psi = MPS(sites)
  @test length(psi) == 10 # just make sure this works
  @test length(siteinds(psi)) == length(psi)

  psi = randomMPS(sites)
  l0s = linkinds(psi)
  orthogonalize!(psi, length(psi) - 1)
  ls = linkinds(psi)
  for (l0, l) in zip(l0s, ls)
    @test tags(l0) == tags(l)
  end
  @test ITensors.leftlim(psi) == length(psi) - 2
  @test ITensors.rightlim(psi) == length(psi)
  orthogonalize!(psi, 2)
  @test ITensors.leftlim(psi) == 1
  @test ITensors.rightlim(psi) == 3
  psi = randomMPS(sites)
  ITensors.setrightlim!(psi, length(psi) + 1) # do this to test qr 
  # from rightmost tensor
  orthogonalize!(psi, div(length(psi), 2))
  @test ITensors.leftlim(psi) == div(length(psi), 2) - 1
  @test ITensors.rightlim(psi) == div(length(psi), 2) + 1

  @test isnothing(linkind(MPS(fill(ITensor(), length(psi)), 0, length(psi) + 1), 1))

  @testset "replacebond!" begin
    # make sure factorization preserves the bond index tags
    psi = randomMPS(sites)
    phi = psi[1] * psi[2]
    bondindtags = tags(linkind(psi, 1))
    replacebond!(psi, 1, phi)
    @test tags(linkind(psi, 1)) == bondindtags

    # check that replacebond! updates llim and rlim properly
    orthogonalize!(psi, 5)
    phi = psi[5] * psi[6]
    replacebond!(psi, 5, phi; ortho="left")
    @test ITensors.leftlim(psi) == 5
    @test ITensors.rightlim(psi) == 7

    phi = psi[5] * psi[6]
    replacebond!(psi, 5, phi; ortho="right")
    @test ITensors.leftlim(psi) == 4
    @test ITensors.rightlim(psi) == 6

    ITensors.setleftlim!(psi, 3)
    ITensors.setrightlim!(psi, 7)
    phi = psi[5] * psi[6]
    replacebond!(psi, 5, phi; ortho="left")
    @test ITensors.leftlim(psi) == 3
    @test ITensors.rightlim(psi) == 7
  end
end

@testset "orthogonalize! with QNs" begin
  sites = siteinds("S=1/2", 8; conserve_qns=true)
  init_state = [isodd(n) ? "Up" : "Dn" for n in 1:length(sites)]
  psi0 = MPS(sites, init_state)
  orthogonalize!(psi0, 4)
  @test ITensors.leftlim(psi0) == 3
  @test ITensors.rightlim(psi0) == 5
end

# Helper function for making MPS
function basicRandomMPS(N::Int; dim=4)
  sites = [Index(2, "Site") for n in 1:N]
  M = MPS(sites)
  links = [Index(dim, "n=$(n-1),Link") for n in 1:(N + 1)]
  for n in 1:N
    M[n] = randomITensor(links[n], sites[n], links[n + 1])
  end
  M[1] *= delta(links[1])
  M[N] *= delta(links[N + 1])
  M[1] /= sqrt(inner(M, M))
  return M
end

function test_correlation_matrix(psi::MPS, ops::Vector{Tuple{String,String}}; kwargs...)
  N = length(psi)
  s = siteinds(psi)
  for op in ops
    Cpm = correlation_matrix(psi, op[1], op[2]; kwargs...)
    # Check using OpSum:
    Copsum = 0.0 * Cpm
    for i in 1:N, j in 1:N
      a = OpSum()
      a += op[1], i, op[2], j
      Copsum[i, j] = inner(psi', MPO(a, s), psi)
    end
    @test Cpm ≈ Copsum rtol = 1E-11

    PM = expect(psi, op[1] * " * " * op[2])
    @test norm(PM - diag(Cpm)) < 1E-8
  end
end

@testset "MPS gauging and truncation" begin
  @testset "orthogonalize! method" begin
    c = 12
    M = basicRandomMPS(30)
    orthogonalize!(M, c)

    @test ITensors.leftlim(M) == c - 1
    @test ITensors.rightlim(M) == c + 1

    # Test for left-orthogonality
    L = M[1] * prime(M[1], "Link")
    l = linkind(M, 1)
    @test norm(L - delta(l, l')) < 1E-12
    for j in 2:(c - 1)
      L = L * M[j] * prime(M[j], "Link")
      l = linkind(M, j)
      @test norm(L - delta(l, l')) < 1E-12
    end

    # Test for right-orthogonality
    R = M[length(M)] * prime(M[length(M)], "Link")
    r = linkind(M, length(M) - 1)
    @test norm(R - delta(r, r')) < 1E-12
    for j in reverse((c + 1):(length(M) - 1))
      R = R * M[j] * prime(M[j], "Link")
      r = linkind(M, j - 1)
      @test norm(R - delta(r, r')) < 1E-12
    end

    @test norm(M[c]) ≈ 1.0
  end

  @testset "truncate! method" begin
    M = basicRandomMPS(10; dim=10)
    M0 = copy(M)
    truncate!(M; maxdim=5)

    @test ITensors.rightlim(M) == 2

    # Test for right-orthogonality
    R = M[length(M)] * prime(M[length(M)], "Link")
    r = linkind(M, length(M) - 1)
    @test norm(R - delta(r, r')) < 1E-12
    for j in reverse(2:(length(M) - 1))
      R = R * M[j] * prime(M[j], "Link")
      r = linkind(M, j - 1)
      @test norm(R - delta(r, r')) < 1E-12
    end

    @test inner(M, M0) > 0.1
  end

  @testset "truncate! with site_range" begin
    M = basicRandomMPS(10; dim=10)
    truncate!(M; site_range=3:7, maxdim=2)
    @test linkdims(M) == [2, 4, 2, 2, 2, 2, 8, 4, 2]
  end
end

@testset "Other MPS methods" begin
  @testset "sample! method" begin
    sites = [Index(3, "Site,n=$n") for n in 1:10]
    psi = randomMPS(sites; linkdims=3)
    nrm2 = inner(psi, psi)
    psi[1] *= (1.0 / sqrt(nrm2))

    s = sample!(psi)

    @test length(s) == length(psi)
    for n in 1:length(psi)
      @test 1 <= s[n] <= 3
    end

    # Throws becase not orthogonalized to site 1:
    orthogonalize!(psi, 3)
    @test_throws ErrorException sample(psi)

    # Throws because not normalized
    orthogonalize!(psi, 1)
    psi[1] *= (5.0 / norm(psi[1]))
    @test_throws ErrorException sample(psi)

    # Works when ortho & normalized:
    orthogonalize!(psi, 1)
    psi[1] *= (1.0 / norm(psi[1]))
    s = sample(psi)
    @test length(s) == length(psi)
  end

  @testset "randomMPS with chi > 1" begin
    chi = 8
    sites = siteinds(2, 20)
    M = randomMPS(sites; linkdims=chi)

    @test ITensors.leftlim(M) == 0
    @test ITensors.rightlim(M) == 2

    @test norm(M[1]) ≈ 1.0

    @test maxlinkdim(M) == chi

    # Test for right-orthogonality
    R = M[length(M)] * prime(M[length(M)], "Link")
    r = linkind(M, length(M) - 1)
    @test norm(R - delta(r, r')) < 1E-10
    for j in reverse(2:(length(M) - 1))
      R = R * M[j] * prime(M[j], "Link")
      r = linkind(M, j - 1)
      @test norm(R - delta(r, r')) < 1E-10
    end

    # Complex case
    Mc = randomMPS(sites; linkdims=chi)
    @test inner(Mc, Mc) ≈ 1.0 + 0.0im
  end

  @testset "randomMPS from initial state (QN case)" begin
    chi = 8
    sites = siteinds("S=1/2", 20; conserve_qns=true)

    # Make flux-zero random MPS
    state = [isodd(n) ? 1 : 2 for n in 1:length(sites)]
    M = randomMPS(sites, state; linkdims=chi)
    @test flux(M) == QN("Sz", 0)

    @test ITensors.leftlim(M) == 0
    @test ITensors.rightlim(M) == 2

    @test norm(M[1]) ≈ 1.0
    @test inner(M, M) ≈ 1.0

    @test maxlinkdim(M) == chi

    # Test making random MPS with different flux
    state[1] = 2
    M = randomMPS(sites, state; linkdims=chi)
    @test flux(M) == QN("Sz", -2)
    state[3] = 2
    M = randomMPS(sites, state; linkdims=chi)
    @test flux(M) == QN("Sz", -4)
  end

  @testset "expect Function" begin
    N = 8
    s = siteinds("S=1/2", N)
    psi = randomMPS(ComplexF64, s; linkdims=2)

    eSz = zeros(N)
    eSx = zeros(N)
    for j in 1:N
      orthogonalize!(psi, j)
      eSz[j] = real(scalar(dag(prime(psi[j], "Site")) * op("Sz", s[j]) * psi[j]))
      eSx[j] = real(scalar(dag(prime(psi[j], "Site")) * op("Sx", s[j]) * psi[j]))
    end

    res = expect(psi, "Sz")
    @test res ≈ eSz

    res = expect(psi, "Sz"; sites=2:4)
    @test res ≈ eSz[2:4]

    res = expect(psi, "Sz"; sites=[2, 4, 8])
    @test res[1] ≈ eSz[2]
    @test res[2] ≈ eSz[4]
    @test res[3] ≈ eSz[8]

    res = expect(psi, "Sz", "Sx")
    @test res[1] ≈ eSz
    @test res[2] ≈ eSx

    res = expect(psi, "Sz"; sites=3)
    @test res isa Float64
    @test res ≈ eSz[3]

    res = expect(psi, "Sz", "Sx"; sites=3)
    @test res isa Tuple{Float64,Float64}
    @test res[1] ≈ eSz[3]
    @test res[2] ≈ eSx[3]

    res = expect(psi, ("Sz", "Sx"))
    @test res isa Tuple{Vector{Float64},Vector{Float64}}
    @test res[1] ≈ eSz
    @test res[2] ≈ eSx

    res = expect(psi, ["Sz" "Sx"; "Sx" "Sz"]; sites=3:7)
    @test res isa Matrix{Vector{Float64}}
    @test res[1, 1] ≈ eSz[3:7]
    @test res[2, 1] ≈ eSx[3:7]
    @test res[1, 2] ≈ eSx[3:7]
    @test res[2, 2] ≈ eSz[3:7]
  end

  @testset "Expected value and Correlations" begin
    m = 2

    # Non-fermionic real case - spin system with QNs (very restrictive on allowed ops)
    s = siteinds("S=1/2", 4; conserve_qns=true)
    psi = randomMPS(s, n -> isodd(n) ? "Up" : "Dn"; linkdims=m)
    test_correlation_matrix(psi, [("S-", "S+"), ("S+", "S-")])

    @test correlation_matrix(psi, [1/2 0; 0 -1/2], [1/2 0; 0 -1/2]) ≈
      correlation_matrix(psi, "Sz", "Sz")
    @test expect(psi, [1/2 0; 0 -1/2]) ≈ expect(psi, "Sz")
    @test all(expect(psi, [1/2 0; 0 -1/2], [1/2 0; 0 -1/2]) .≈ expect(psi, "Sz", "Sz"))
    @test expect(psi, [[1/2 0; 0 -1/2], [1/2 0; 0 -1/2]]) ≈ expect(psi, ["Sz", "Sz"])

    s = siteinds("S=1/2", length(s); conserve_qns=false)
    psi = randomMPS(s, n -> isodd(n) ? "Up" : "Dn"; linkdims=m)
    test_correlation_matrix(
      psi,
      [
        ("Sz", "Sz"),
        ("iSy", "iSy"),
        ("Sx", "Sx"),
        ("Sz", "Sx"),
        ("S+", "S+"),
        ("S-", "S+"),
        ("S+", "S-"),
        ("Sx", "S+"),
        ("iSy", "iSy"),
        ("Sx", "iSy"),
      ],
    )

    test_correlation_matrix(psi, [("Sz", "Sz")]; ishermitian=false)
    test_correlation_matrix(psi, [("Sz", "Sz")]; ishermitian=true)
    test_correlation_matrix(psi, [("Sz", "Sx")]; ishermitian=false)
    # This will fail becuase Sz*Sx is not hermitian, so the below diagonla corr matrix
    # need to be calculated explicitely.
    #test_correlation_matrix(psi,[("Sz", "Sx")];ishermitian=true)

    #Test sites feature
    s = siteinds("S=1/2", 8; conserve_qns=false)
    psi = randomMPS(s, n -> isodd(n) ? "Up" : "Dn"; linkdims=m)
    PM = expect(psi, "S+ * S-")
    Cpm = correlation_matrix(psi, "S+", "S-")
    range = 3:7
    Cpm37 = correlation_matrix(psi, "S+", "S-"; sites=range)
    @test norm(Cpm37 - Cpm[range, range]) < 1E-8

    @test norm(PM[range] - expect(psi, "S+ * S-"; sites=range)) < 1E-8

    # With start_site, end_site arguments:
    s = siteinds("S=1/2", 8)
    psi = randomMPS(ComplexF64, s; linkdims=m)
    ss, es = 3, 6
    Nb = es - ss + 1
    Cpm = correlation_matrix(psi, "S+", "S-"; sites=ss:es)
    Czz = correlation_matrix(psi, "Sz", "Sz"; sites=ss:es)
    @test size(Cpm) == (Nb, Nb)
    # Check using OpSum:
    for i in ss:es, j in i:es
      a = OpSum()
      a += "S+", i, "S-", j
      @test inner(psi', MPO(a, s), psi) ≈ Cpm[i - ss + 1, j - ss + 1]
    end

    # Electron case
    s = siteinds("Electron", 8)
    psi = randomMPS(s; linkdims=m)
    test_correlation_matrix(
      psi, [("Cdagup", "Cup"), ("Cup", "Cdagup"), ("Cup", "Cdn"), ("Cdagdn", "Cdn")]
    )

    s = siteinds("Electron", 8; conserve_qns=false)
    psi = randomMPS(s; linkdims=m)
    test_correlation_matrix(
      psi,
      [
        ("Ntot", "Ntot"),
        ("Nup", "Nup"),
        ("Ndn", "Ndn"),
        ("Cdagup", "Cup"),
        ("Adagup", "Aup"),
        ("Cdn", "Cdagdn"),
        ("Adn", "Adagdn"),
        ("Sz", "Sz"),
        ("S+", "S-"),
      ],
    )
    # can't test ,("Cdn","Cdn") yet, because AutoMPO thinks this is antisymmetric 

    #trigger unsupported error
    let err = nothing
      try
        test_correlation_matrix(psi, [("Cup", "Aup")])
      catch err
      end

      @test err isa Exception
      @test sprint(showerror, err) ==
        "correlation_matrix: Mixed fermionic and bosonic operators are not supported yet."
    end

    # Fermion case
    s = siteinds("Fermion", 8)
    psi = randomMPS(s; linkdims=m)
    test_correlation_matrix(psi, [("N", "N"), ("Cdag", "C"), ("C", "Cdag"), ("C", "C")])

    s = siteinds("Fermion", 8; conserve_qns=false)
    psi = randomMPS(s; linkdims=m)
    test_correlation_matrix(psi, [("N", "N"), ("Cdag", "C"), ("C", "Cdag"), ("C", "C")])

    #
    # Test non-contiguous sites input
    #
    C = correlation_matrix(psi, "N", "N")
    non_contiguous = [1, 3, 8]
    Cs = correlation_matrix(psi, "N", "N"; sites=non_contiguous)
    for (ni, i) in enumerate(non_contiguous), (nj, j) in enumerate(non_contiguous)
      @test Cs[ni, nj] ≈ C[i, j]
    end

    C2 = correlation_matrix(psi, "N", "N"; sites=2)
    @test C2 isa Matrix
    @test C2[1, 1] ≈ C[2, 2]
  end #testset

  @testset "correlation_matrix with dangling bonds" begin
    l01 = Index(3)
    l̃01 = sim(l01)
    l12 = Index(3)
    l23 = Index(3)
    l34 = Index(3)
    l̃34 = sim(l34)
    s1 = Index(2, "Qubit")
    s2 = Index(2, "Qubit")
    s3 = Index(2, "Qubit")
    b01 = randomITensor(l01, l̃01)
    A1 = randomITensor(l̃01, s1, l12)
    A2 = randomITensor(l12, s2, l23)
    A3 = randomITensor(l23, s3, l̃34)
    b34 = randomITensor(l̃34, l34)
    ψ = MPS([b01, A1, A2, A3, b34])
    sites = 1:3
    C = correlation_matrix(ψ, "Z", "Z"; sites=(sites .+ 1))
    siteinds = [l01, s1, s2, s3, l34]
    ψψ = inner(ψ, ψ)
    zz = map(Iterators.product(sites .+ 1, sites .+ 1)) do I
      i, j = I
      ZiZj = MPO(OpSum() + ("Z", i, "Z", j), siteinds)
      return inner(ψ', ZiZj, ψ) / ψψ
    end
    for I in eachindex(C)
      @test C[I] ≈ zz[I]
    end
  end

  @testset "expect regression test for in-place modification of input MPS" begin
    s = siteinds("S=1/2", 5)
    psi = randomMPS(s; linkdims=3)
    orthogonalize!(psi, 1)
    expect_init = expect(psi, "Sz")
    norm_scale = 10
    psi[1] *= norm_scale
    @test ortho_lims(psi) == 1:1
    @test norm(psi) ≈ norm_scale
    expect_Sz = expect(psi, "Sz")
    @test all(≤(1 / 2), expect_Sz)
    @test expect_Sz ≈ expect_init
    @test ortho_lims(psi) == 1:1
    @test norm(psi) ≈ norm_scale
  end

  @testset "correlation_matrix regression test for in-place modification of input MPS" begin
    s = siteinds("S=1/2", 5)
    psi = randomMPS(s; linkdims=3)
    orthogonalize!(psi, 1)
    correlation_matrix_init = correlation_matrix(psi, "Sz", "Sz")
    norm_scale = 10
    psi[1] *= norm_scale
    @test ortho_lims(psi) == 1:1
    @test norm(psi) ≈ norm_scale
    correlation_matrix_SzSz = correlation_matrix(psi, "Sz", "Sz")
    @test all(≤(1 / 2), correlation_matrix_SzSz)
    @test correlation_matrix_SzSz ≈ correlation_matrix_init
    @test ortho_lims(psi) == 1:1
    @test norm(psi) ≈ norm_scale
  end

  @testset "swapbondsites" begin
    sites = siteinds("S=1/2", 5)
    ψ0 = randomMPS(sites)
    ψ = replacebond(ψ0, 3, ψ0[3] * ψ0[4]; swapsites=true, cutoff=1e-15)
    @test siteind(ψ, 1) == siteind(ψ0, 1)
    @test siteind(ψ, 2) == siteind(ψ0, 2)
    @test siteind(ψ, 4) == siteind(ψ0, 3)
    @test siteind(ψ, 3) == siteind(ψ0, 4)
    @test siteind(ψ, 5) == siteind(ψ0, 5)
    @test prod(ψ) ≈ prod(ψ0)
    @test maxlinkdim(ψ) == 1

    ψ = swapbondsites(ψ0, 4; cutoff=1e-15)
    @test siteind(ψ, 1) == siteind(ψ0, 1)
    @test siteind(ψ, 2) == siteind(ψ0, 2)
    @test siteind(ψ, 3) == siteind(ψ0, 3)
    @test siteind(ψ, 5) == siteind(ψ0, 4)
    @test siteind(ψ, 4) == siteind(ψ0, 5)
    @test prod(ψ) ≈ prod(ψ0)
    @test maxlinkdim(ψ) == 1
  end

  @testset "map!" begin
    s = siteinds("S=½", 5)
    M0 = MPS(s, "↑")

    # Test map! with limits getting set
    M = orthogonalize(M0, 1)
    @test ITensors.leftlim(M) == 0
    @test ITensors.rightlim(M) == 2
    map!(prime, M)
    @test ITensors.leftlim(M) == 0
    @test ITensors.rightlim(M) == length(M0) + 1

    # Test map! without limits getting set
    M = orthogonalize(M0, 1)
    map!(prime, M; set_limits=false)
    @test ITensors.leftlim(M) == 0
    @test ITensors.rightlim(M) == 2

    # Test prime! with limits getting set
    M = orthogonalize(M0, 1)
    @test ITensors.leftlim(M) == 0
    @test ITensors.rightlim(M) == 2
    prime!(M; set_limits=true)
    @test ITensors.leftlim(M) == 0
    @test ITensors.rightlim(M) == length(M0) + 1

    # Test prime! without limits getting set
    M = orthogonalize(M0, 1)
    prime!(M)
    @test ITensors.leftlim(M) == 0
    @test ITensors.rightlim(M) == 2
  end

  @testset "setindex!(::MPS, _, ::Colon)" begin
    s = siteinds("S=½", 4)
    ψ = randomMPS(s)
    ϕ = MPS(s, "↑")
    orthogonalize!(ϕ, 1)
    ψ[:] = ϕ
    @test ITensors.orthocenter(ψ) == 1
    @test inner(ψ, ϕ) ≈ 1

    ψ = randomMPS(s)
    ϕ = MPS(s, "↑")
    orthogonalize!(ϕ, 1)
    ψ[:] = ITensors.data(ϕ)
    @test ITensors.leftlim(ψ) == 0
    @test ITensors.rightlim(ψ) == length(ψ) + 1
    @test inner(ψ, ϕ) ≈ 1
  end

  @testset "findsite[s](::MPS/MPO, is)" begin
    s = siteinds("S=1/2", 5)
    ψ = randomMPS(s)
    l = linkinds(ψ)

    A = randomITensor(s[4]', s[2]', dag(s[4]), dag(s[2]))

    @test findsite(ψ, s[3]) == 3
    @test findsite(ψ, (s[3], s[5])) == 3
    @test findsite(ψ, l[2]) == 2
    @test findsite(ψ, A) == 2

    @test findsites(ψ, s[3]) == [3]
    @test findsites(ψ, (s[4], s[1])) == [1, 4]
    @test findsites(ψ, l[2]) == [2, 3]
    @test findsites(ψ, (l[2], l[3])) == [2, 3, 4]
    @test findsites(ψ, A) == [2, 4]

    M = randomMPO(s)
    lM = linkinds(M)

    @test findsite(M, s[4]) == 4
    @test findsite(M, s[4]') == 4
    @test findsite(M, (s[4]', s[4])) == 4
    @test findsite(M, (s[4]', s[3])) == 3
    @test findsite(M, lM[2]) == 2
    @test findsite(M, A) == 2

    @test findsites(M, s[4]) == [4]
    @test findsites(M, s[4]') == [4]
    @test findsites(M, (s[4]', s[4])) == [4]
    @test findsites(M, (s[4]', s[3])) == [3, 4]
    @test findsites(M, (lM[2], lM[3])) == [2, 3, 4]
    @test findsites(M, A) == [2, 4]
  end

  @testset "[first]siteind[s](::MPS/MPO, j::Int)" begin
    s = siteinds("S=1/2", 5)
    ψ = randomMPS(s)
    @test siteind(first, ψ, 3) == s[3]
    @test siteind(ψ, 4) == s[4]
    @test isnothing(siteind(ψ, 4; plev=1))
    @test siteinds(ψ, 3) == IndexSet(s[3])
    @test siteinds(ψ, 3; plev=1) == IndexSet()

    M = randomMPO(s)
    @test noprime(siteind(first, M, 4)) == s[4]
    @test siteind(first, M, 4; plev=0) == s[4]
    @test siteind(first, M, 4; plev=1) == s[4]'
    @test siteind(M, 4) == s[4]
    @test siteind(M, 4; plev=0) == s[4]
    @test siteind(M, 4; plev=1) == s[4]'
    @test isnothing(siteind(M, 4; plev=2))
    @test hassameinds(siteinds(M, 3), (s[3], s[3]'))
    @test siteinds(M, 3; plev=1) == IndexSet(s[3]')
    @test siteinds(M, 3; plev=0) == IndexSet(s[3])
    @test siteinds(M, 3; tags="n=2") == IndexSet()
  end

  @testset "movesites $N sites" for N in 1:4
    s0 = siteinds("S=1/2", N)
    for perm in permutations(1:N)
      s = s0[perm]
      ψ = MPS(s, rand(("↑", "↓"), N))
      ns′ = [findfirst(==(i), s0) for i in s]
      @test ns′ == perm
      ψ′ = movesites(ψ, 1:N .=> ns′; cutoff=1e-15)
      if N == 1
        @test maxlinkdim(ψ′) == 1
      else
        @test maxlinkdim(ψ′) == 1
      end
      for n in 1:N
        @test s0[n] == siteind(ψ′, n)
      end
      @test prod(ψ) ≈ prod(ψ′)
    end
  end

  @testset "Construct MPS from ITensor" begin
    N = 5
    s = siteinds("S=1/2", N)
    l = [Index(3, "left_$n") for n in 1:2]
    r = [Index(3, "right_$n") for n in 1:2]

    #
    # MPS
    #

    A = randomITensor(s...)
    ψ = MPS(A, s)
    @test prod(ψ) ≈ A
    @test ITensors.orthocenter(ψ) == N
    @test maxlinkdim(ψ) == 4

    ψ0 = MPS(s, "↑")
    A = prod(ψ0)
    ψ = MPS(A, s; cutoff=1e-15)
    @test prod(ψ) ≈ A
    @test ITensors.orthocenter(ψ) == N
    @test maxlinkdim(ψ) == 1

    ψ0 = randomMPS(s; linkdims=2)
    A = prod(ψ0)
    ψ = MPS(A, s; cutoff=1e-15, orthocenter=2)
    @test prod(ψ) ≈ A
    @test ITensors.orthocenter(ψ) == 2
    @test maxlinkdim(ψ) == 2

    A = randomITensor(s..., l[1], r[1])
    ψ = MPS(A, s; leftinds=l[1], orthocenter=3)
    ls = linkinds(ψ)
    @test hassameinds(ψ[1], (l[1], s[1], ls[1]))
    @test hassameinds(ψ[N], (r[1], s[N], ls[N - 1]))
    @test prod(ψ) ≈ A
    @test ITensors.orthocenter(ψ) == 3
    @test maxlinkdim(ψ) == 12

    A = randomITensor(s..., l..., r...)
    ψ = MPS(A, s; leftinds=l)
    ls = linkinds(ψ)
    @test hassameinds(ψ[1], (l..., s[1], ls[1]))
    @test hassameinds(ψ[N], (r..., s[N], ls[N - 1]))
    @test prod(ψ) ≈ A
    @test ITensors.orthocenter(ψ) == N
    @test maxlinkdim(ψ) == 36

    #
    # Construct from regular Julia tensor
    # 
    Ajt = randn(2, 2, 2, 2, 2) # Julia tensor
    ψ = MPS(Ajt, s)
    @test prod(ψ) ≈ ITensor(Ajt, s...)

    Ajv = randn(2^N) # Julia vector
    ψ = MPS(Ajv, s)
    @test prod(ψ) ≈ ITensor(Ajv, s...)
  end

  @testset "Set range of MPS tensors" begin
    N = 5
    s = siteinds("S=1/2", N)
    ψ0 = randomMPS(s; linkdims=3)

    ψ = orthogonalize(ψ0, 2)
    A = prod(ITensors.data(ψ)[2:(N - 1)])
    randn!(A)
    ϕ = MPS(A, s[2:(N - 1)]; orthocenter=1)
    ψ[2:(N - 1)] = ϕ
    @test prod(ψ) ≈ ψ[1] * A * ψ[N]
    @test maxlinkdim(ψ) == 4
    @test ITensors.orthocenter(ψ) == 2

    ψ = orthogonalize(ψ0, 1)
    A = prod(ITensors.data(ψ)[2:(N - 1)])
    randn!(A)
    @test_throws AssertionError ψ[2:(N - 1)] = A

    ψ = orthogonalize(ψ0, 2)
    A = prod(ITensors.data(ψ)[2:(N - 1)])
    randn!(A)
    ψ[2:(N - 1), orthocenter=3] = A
    @test prod(ψ) ≈ ψ[1] * A * ψ[N]
    @test maxlinkdim(ψ) == 4
    @test ITensors.orthocenter(ψ) == 3
  end

  @testset "movesites reverse sites" begin
    N = 6
    s = siteinds("S=1/2", N)
    ψ0 = randomMPS(s)
    ψ = movesites(ψ0, 1:N .=> reverse(1:N))
    for n in 1:N
      @test siteind(ψ, n) == s[N - n + 1]
    end
  end

  @testset "movesites subsets of sites" begin
    N = 6
    s = siteinds("S=1/2", N)
    ψ = randomMPS(s)

    for i in 1:N, j in 1:N
      ns = [i, j]
      !allunique(ns) && continue
      min_ns = minimum(ns)
      ns′ = collect(min_ns:(min_ns + length(ns) - 1))
      ψ′ = movesites(ψ, ns .=> ns′; cutoff=1e-15)
      @test siteind(ψ′, min_ns) == siteind(ψ, i)
      @test siteind(ψ′, min_ns + 1) == siteind(ψ, j)
      @test maxlinkdim(ψ′) == 1
      ψ̃ = movesites(ψ′, ns′ .=> ns; cutoff=1e-15)
      for n in 1:N
        @test siteind(ψ̃, n) == siteind(ψ, n)
      end
      @test maxlinkdim(ψ̃) == 1
    end

    for i in 1:N, j in 1:N, k in 1:N
      ns = [i, j, k]
      !allunique(ns) && continue
      min_ns = minimum(ns)
      ns′ = collect(min_ns:(min_ns + length(ns) - 1))
      ψ′ = movesites(ψ, ns .=> ns′; cutoff=1e-15)
      @test siteind(ψ′, min_ns) == siteind(ψ, i)
      @test siteind(ψ′, min_ns + 1) == siteind(ψ, j)
      @test siteind(ψ′, min_ns + 2) == siteind(ψ, k)
      @test maxlinkdim(ψ′) == 1
      ψ̃ = movesites(ψ′, ns′ .=> ns; cutoff=1e-15)
      for n in 1:N
        @test siteind(ψ̃, n) == siteind(ψ, n)
      end
      @test maxlinkdim(ψ̃) == 1
    end

    for i in 1:N, j in 1:N, k in 1:N, l in 1:N
      ns = [i, j, k, l]
      !allunique(ns) && continue
      min_ns = minimum(ns)
      ns′ = collect(min_ns:(min_ns + length(ns) - 1))
      ψ′ = movesites(ψ, ns .=> ns′; cutoff=1e-15)
      @test siteind(ψ′, min_ns) == siteind(ψ, i)
      @test siteind(ψ′, min_ns + 1) == siteind(ψ, j)
      @test siteind(ψ′, min_ns + 2) == siteind(ψ, k)
      @test siteind(ψ′, min_ns + 3) == siteind(ψ, l)
      @test maxlinkdim(ψ′) == 1
      ψ̃ = movesites(ψ′, ns′ .=> ns; cutoff=1e-15)
      for n in 1:N
        @test siteind(ψ̃, n) == siteind(ψ, n)
      end
      @test maxlinkdim(ψ̃) == 1
    end

    for i in 1:N, j in 1:N, k in 1:N, l in 1:N, m in 1:N
      ns = [i, j, k, l, m]
      !allunique(ns) && continue
      min_ns = minimum(ns)
      ns′ = collect(min_ns:(min_ns + length(ns) - 1))
      ψ′ = movesites(ψ, ns .=> ns′; cutoff=1e-15)
      for n in 1:length(ns)
        @test siteind(ψ′, min_ns + n - 1) == siteind(ψ, ns[n])
      end
      @test maxlinkdim(ψ′) == 1
      ψ̃ = movesites(ψ′, ns′ .=> ns; cutoff=1e-15)
      for n in 1:N
        @test siteind(ψ̃, n) == siteind(ψ, n)
      end
      @test maxlinkdim(ψ̃) == 1
    end
  end

  @testset "product(::Vector{ITensor}, ::MPS)" begin
    N = 6
    s = siteinds("Qubit", N)

    I = [op("Id", s, n) for n in 1:N]
    X = [op("X", s, n) for n in 1:N]
    Y = [op("Y", s, n) for n in 1:N]
    Z = [op("Z", s, n) for n in 1:N]
    H = [op("H", s, n) for n in 1:N]
    CX = [op("CX", s, n, m) for n in 1:N, m in 1:N]
    CY = [op("CY", s, n, m) for n in 1:N, m in 1:N]
    CZ = [op("CZ", s, n, m) for n in 1:N, m in 1:N]
    CCNOT = [op("CCNOT", s, n, m, k) for n in 1:N, m in 1:N, k in 1:N]
    CSWAP = [op("CSWAP", s, n, m, k) for n in 1:N, m in 1:N, k in 1:N]
    CCCNOT = [op("CCCNOT", s, n, m, k, l) for n in 1:N, m in 1:N, k in 1:N, l in 1:N]

    v0 = [onehot(s[n] => "0") for n in 1:N]
    v1 = [onehot(s[n] => "1") for n in 1:N]

    # Single qubit
    @test product(I[1], v0[1]) ≈ v0[1]
    @test product(I[1], v1[1]) ≈ v1[1]

    @test product(H[1], H[1]) ≈ I[1]
    @test product(H[1], v0[1]) ≈ 1 / sqrt(2) * (v0[1] + v1[1])
    @test product(H[1], v1[1]) ≈ 1 / sqrt(2) * (v0[1] - v1[1])

    @test product(X[1], v0[1]) ≈ v1[1]
    @test product(X[1], v1[1]) ≈ v0[1]

    @test product(Y[1], v0[1]) ≈ im * v1[1]
    @test product(Y[1], v1[1]) ≈ -im * v0[1]

    @test product(Z[1], v0[1]) ≈ v0[1]
    @test product(Z[1], v1[1]) ≈ -v1[1]

    @test product(X[1], X[1]) ≈ I[1]
    @test product(Y[1], Y[1]) ≈ I[1]
    @test product(Z[1], Z[1]) ≈ I[1]
    @test -im * product([Y[1], X[1]], Z[1]) ≈ I[1]

    @test dag(X[1]) ≈ -product([X[1], Y[1]], Y[1])
    @test dag(Y[1]) ≈ -product([Y[1], Y[1]], Y[1])
    @test dag(Z[1]) ≈ -product([Z[1], Y[1]], Y[1])

    @test product(X[1], Y[1]) - product(Y[1], X[1]) ≈ 2 * im * Z[1]
    @test product(Y[1], Z[1]) - product(Z[1], Y[1]) ≈ 2 * im * X[1]
    @test product(Z[1], X[1]) - product(X[1], Z[1]) ≈ 2 * im * Y[1]

    @test product([Y[1], X[1]], v0[1]) - product([X[1], Y[1]], v0[1]) ≈
      2 * im * product(Z[1], v0[1])
    @test product([Y[1], X[1]], v1[1]) - product([X[1], Y[1]], v1[1]) ≈
      2 * im * product(Z[1], v1[1])
    @test product([Z[1], Y[1]], v0[1]) - product([Y[1], Z[1]], v0[1]) ≈
      2 * im * product(X[1], v0[1])
    @test product([Z[1], Y[1]], v1[1]) - product([Y[1], Z[1]], v1[1]) ≈
      2 * im * product(X[1], v1[1])
    @test product([X[1], Z[1]], v0[1]) - product([Z[1], X[1]], v0[1]) ≈
      2 * im * product(Y[1], v0[1])
    @test product([X[1], Z[1]], v1[1]) - product([Z[1], X[1]], v1[1]) ≈
      2 * im * product(Y[1], v1[1])

    #
    # 2-qubit
    #

    @test product(I[1] * I[2], v0[1] * v0[2]) ≈ v0[1] * v0[2]

    @test product(CX[1, 2], v0[1] * v0[2]) ≈ v0[1] * v0[2]
    @test product(CX[1, 2], v0[1] * v1[2]) ≈ v0[1] * v1[2]
    @test product(CX[1, 2], v1[1] * v0[2]) ≈ v1[1] * v1[2]
    @test product(CX[1, 2], v1[1] * v1[2]) ≈ v1[1] * v0[2]

    @test product(CY[1, 2], v0[1] * v0[2]) ≈ v0[1] * v0[2]
    @test product(CY[1, 2], v0[1] * v1[2]) ≈ v0[1] * v1[2]
    @test product(CY[1, 2], v1[1] * v0[2]) ≈ im * v1[1] * v1[2]
    @test product(CY[1, 2], v1[1] * v1[2]) ≈ -im * v1[1] * v0[2]

    @test product(CZ[1, 2], v0[1] * v0[2]) ≈ v0[1] * v0[2]
    @test product(CZ[1, 2], v0[1] * v1[2]) ≈ v0[1] * v1[2]
    @test product(CZ[1, 2], v1[1] * v0[2]) ≈ v1[1] * v0[2]
    @test product(CZ[1, 2], v1[1] * v1[2]) ≈ -v1[1] * v1[2]

    #
    # 3-qubit
    #

    @test product(CCNOT[1, 2, 3], v0[1] * v0[2] * v0[3]) ≈ v0[1] * v0[2] * v0[3]
    @test product(CCNOT[1, 2, 3], v0[1] * v0[2] * v1[3]) ≈ v0[1] * v0[2] * v1[3]
    @test product(CCNOT[1, 2, 3], v0[1] * v1[2] * v0[3]) ≈ v0[1] * v1[2] * v0[3]
    @test product(CCNOT[1, 2, 3], v0[1] * v1[2] * v1[3]) ≈ v0[1] * v1[2] * v1[3]
    @test product(CCNOT[1, 2, 3], v1[1] * v0[2] * v0[3]) ≈ v1[1] * v0[2] * v0[3]
    @test product(CCNOT[1, 2, 3], v1[1] * v0[2] * v1[3]) ≈ v1[1] * v0[2] * v1[3]
    @test product(CCNOT[1, 2, 3], v1[1] * v1[2] * v0[3]) ≈ v1[1] * v1[2] * v1[3]
    @test product(CCNOT[1, 2, 3], v1[1] * v1[2] * v1[3]) ≈ v1[1] * v1[2] * v0[3]

    @test product(CSWAP[1, 2, 3], v0[1] * v0[2] * v0[3]) ≈ v0[1] * v0[2] * v0[3]
    @test product(CSWAP[1, 2, 3], v0[1] * v0[2] * v1[3]) ≈ v0[1] * v0[2] * v1[3]
    @test product(CSWAP[1, 2, 3], v0[1] * v1[2] * v0[3]) ≈ v0[1] * v1[2] * v0[3]
    @test product(CSWAP[1, 2, 3], v0[1] * v1[2] * v1[3]) ≈ v0[1] * v1[2] * v1[3]
    @test product(CSWAP[1, 2, 3], v1[1] * v0[2] * v0[3]) ≈ v1[1] * v0[2] * v0[3]
    @test product(CSWAP[1, 2, 3], v1[1] * v0[2] * v1[3]) ≈ v1[1] * v1[2] * v0[3]
    @test product(CSWAP[1, 2, 3], v1[1] * v1[2] * v0[3]) ≈ v1[1] * v0[2] * v1[3]
    @test product(CSWAP[1, 2, 3], v1[1] * v1[2] * v1[3]) ≈ v1[1] * v1[2] * v1[3]

    #
    # Apply to an MPS
    #

    ψ = MPS(s, "0")
    @test prod(product(X[1], ψ)) ≈ prod(MPS(s, n -> n == 1 ? "1" : "0"))
    @test prod(product(X[1], product(X[2], ψ))) ≈
      prod(MPS(s, n -> n == 1 || n == 2 ? "1" : "0"))
    @test prod(product(X[1] * X[2], ψ)) ≈ prod(MPS(s, n -> n == 1 || n == 2 ? "1" : "0"))
    @test prod(product([X[2], X[1]], ψ)) ≈ prod(MPS(s, n -> n == 1 || n == 2 ? "1" : "0"))
    @test prod(product(CX[1, 2], ψ)) ≈ prod(MPS(s, "0"))
    @test prod(product(CX[1, 2], product(X[1], ψ))) ≈
      prod(MPS(s, n -> n == 1 || n == 2 ? "1" : "0"))
    @test prod(product(product(CX[1, 2], X[1]), ψ)) ≈
      prod(MPS(s, n -> n == 1 || n == 2 ? "1" : "0"))
    @test prod(product([X[1], CX[1, 2]], ψ)) ≈
      prod(MPS(s, n -> n == 1 || n == 2 ? "1" : "0"))

    for i in 1:N, j in 1:N
      !allunique((i, j)) && continue
      # Don't move sites back
      CXij_ψ = product([X[i], CX[i, j]], ψ; move_sites_back=false, cutoff=1e-15)
      @test maxlinkdim(CXij_ψ) == 1
      @test prod(CXij_ψ) ≈ prod(MPS(s, n -> n == i || n == j ? "1" : "0"))

      # Move sites back
      CXij_ψ = product([X[i], CX[i, j]], ψ)
      for n in 1:N
        @test siteind(CXij_ψ, n) == siteind(ψ, n)
      end
      @test prod(CXij_ψ) ≈ prod(MPS(s, n -> n == i || n == j ? "1" : "0"))
    end

    for i in 1:N, j in 1:N, k in 1:N
      ns = (i, j, k)
      !allunique(ns) && continue
      # Don't move sites back
      CCNOTijk_ψ = product(
        [X[j], X[i], CCNOT[ns...]], ψ; move_sites_back=false, cutoff=1e-15
      )
      @test maxlinkdim(CCNOTijk_ψ) == 1
      @test prod(CCNOTijk_ψ) ≈ prod(MPS(s, n -> n ∈ ns ? "1" : "0"))

      # Move sites back
      CCNOTijk_ψ = product([X[j], X[i], CCNOT[ns...]], ψ; cutoff=1e-15)
      @test maxlinkdim(CCNOTijk_ψ) == 1
      for n in 1:N
        @test siteind(CCNOTijk_ψ, n) == siteind(ψ, n)
      end
      @test prod(CCNOTijk_ψ) ≈ prod(MPS(s, n -> n ∈ ns ? "1" : "0"))
    end

    for i in 1:N, j in i:N, k in 1:N, l in k:N
      ns = (i, j, k, l)
      !allunique(ns) && continue
      # Don't move sites back
      CCCNOTijkl_ψ = product(
        [X[i], X[j], X[k], CCCNOT[ns...]], ψ; move_sites_back=false, cutoff=1e-15
      )
      @test maxlinkdim(CCCNOTijkl_ψ) == 1
      @test prod(CCCNOTijkl_ψ) ≈ prod(MPS(s, n -> n ∈ ns ? "1" : "0"))

      # Move sites back
      CCCNOTijkl_ψ = product([X[i], X[j], X[k], CCCNOT[ns...]], ψ; cutoff=1e-15)
      @test maxlinkdim(CCCNOTijkl_ψ) == 1
      for n in 1:N
        @test siteind(CCCNOTijkl_ψ, n) == siteind(ψ, n)
      end
      @test prod(CCCNOTijkl_ψ) ≈ prod(MPS(s, n -> n ∈ ns ? "1" : "0"))
    end
  end

  @testset "product" begin
    @testset "Contraction order of operations" begin
      s = siteind("Qubit")
      Q = SiteType("Qubit")
      @test product(ops([s], [("Y", 1), ("X", 1)]), setelt(s => 1)) ≈
        itensor(op("X", Q) * op("Y", Q) * [1; 0], s)
      @test product(ops([s], [("Y", 1), ("Z", 1)]), setelt(s => 1)) ≈
        itensor(op("Z", Q) * op("Y", Q) * [1; 0], s)
      @test product(ops([s], [("X", 1), ("Y", 1)]), setelt(s => 1)) ≈
        itensor(op("Y", Q) * op("X", Q) * [1; 0], s)
    end

    @testset "Simple on-site state evolution" begin
      N = 3

      pos = [("Z", 3), ("Y", 2), ("X", 1)]

      s = siteinds("Qubit", N)
      gates = ops(s, pos)
      ψ0 = MPS(s, "0")

      # Apply the gates
      ψ = product(gates, ψ0)

      # Move site 1 to position 3
      ψ′ = movesite(ψ, 1 => 3)
      @test siteind(ψ′, 1) == s[2]
      @test siteind(ψ′, 2) == s[3]
      @test siteind(ψ′, 3) == s[1]
      @test prod(ψ) ≈ prod(ψ′)

      # Move the site back
      ψ′′ = movesite(ψ′, 3 => 1)
      @test siteind(ψ′′, 1) == s[1]
      @test siteind(ψ′′, 2) == s[2]
      @test siteind(ψ′′, 3) == s[3]
      @test prod(ψ) ≈ prod(ψ′′)
    end

    @testset "More complex evolution" begin
      N = 7

      osX = [("X", n) for n in 1:N]

      osZ = [("Z", n) for n in 1:N]

      osSw = [("SWAP", n, n + 1) for n in 1:(N - 2)]

      osCx = [("CX", n, n + 3) for n in 1:(N - 3)]

      osT = [("CCX", n, n + 1, n + 3) for n in 1:(N - 3)]

      osRx = [("Rx", n, (θ=π,)) for n in 1:N]

      osXX = [("Rxx", (n, n + 1), (ϕ=π / 8,)) for n in 1:(N - 1)]

      #os_noise = [("noise", n, n+2, n+4) for n in 1:N-4]

      os = vcat(osX, osXX, osSw, osRx, osZ, osCx, osT)
      s = siteinds("Qubit", N)
      gates = ops(os, s)

      @testset "Pure state evolution" begin
        ψ0 = MPS(s, "0")
        ψ = product(gates, ψ0; cutoff=1e-15)
        @test maxlinkdim(ψ) == 8
        prodψ = product(gates, prod(ψ0))
        @test prod(ψ) ≈ prodψ rtol = 1e-12
      end

      M0 = MPO(s, "Id")
      maxdim = prod(dim(siteinds(M0, j)) for j in 1:N)

      @testset "Mixed state evolution" begin
        M = product(gates, M0; cutoff=1e-15, maxdim=maxdim)
        @test maxlinkdim(M) == 24 || maxlinkdim(M) == 25
        sM0 = siteinds(M0)
        sM = siteinds(M)
        for n in 1:N
          @test hassameinds(sM[n], sM0[n])
        end
        @set_warn_order 15 begin
          prodM = product(gates, prod(M0))
          @test prod(M) ≈ prodM rtol = 1e-6
        end
      end

      #@testset "Mixed state noisy evolution" begin
      #  prepend!(os, os_noise)
      #  gates = ops(os, s)
      #  M = product(gates, M0; apply_dag = true,
      #              cutoff = 1e-15, maxdim = maxdim)
      #  @test maxlinkdim(M) == 64
      #  sM0 = siteinds(M0)
      #  sM = siteinds(M)
      #  for n in 1:N
      #    @test hassameinds(sM[n], sM0[n])
      #  end
      #  @set_warn_order 16 begin
      #    prodM = product(gates, prod(M0); apply_dag = true)
      #    @test prod(M) ≈ prodM rtol = 1e-7
      #  end
      #end

      #@testset "Mixed state noisy evolution" begin
      #  prepend!(os, os_noise)
      #  gates = ops(os, s)
      #  M = product(gates, M0;
      #              apply_dag = true, cutoff = 1e-15, maxdim = maxdim-1)
      #  @test maxlinkdim(M) == 64
      #  sM0 = siteinds(M0)
      #  sM = siteinds(M)
      #  for n in 1:N
      #    @test hassameinds(sM[n], sM0[n])
      #  end
      #  @set_warn_order 16 begin
      #    prodM = product(gates, prod(M0); apply_dag = true)
      #    @test prod(M) ≈ prodM rtol = 1e-7
      #  end
      #end

    end

    @testset "Gate evolution open system" begin
      N = 8
      osX = [("X", n) for n in 1:N]
      osZ = [("Z", n) for n in 1:N]
      osSw = [("SWAP", n, n + 2) for n in 1:(N - 2)]
      osCx = [("CX", n, n + 3) for n in 1:(N - 3)]
      osT = [("CCX", n, n + 1, n + 3) for n in 1:(N - 3)]
      osRx = [("Rx", n, (θ=π,)) for n in 1:N]
      os = vcat(osX, osSw, osRx, osZ, osCx, osT)

      s = siteinds("Qubit", N)
      gates = ops(os, s)

      M0 = MPO(s, "Id")

      # Apply the gates

      s0 = siteinds(M0)

      M = apply(gates, M0; apply_dag=true, cutoff=1e-15, maxdim=500, svd_alg="qr_iteration")

      s = siteinds(M)
      for n in 1:N
        @assert hassameinds(s[n], s0[n])
      end

      @set_warn_order 18 begin
        prodM = apply(gates, prod(M0); apply_dag=true)
        @test prod(M) ≈ prodM rtol = 1e-6
      end
    end

    @testset "Gate evolution state" begin
      N = 10

      osX = [("X", n) for n in 1:N]
      osZ = [("Z", n) for n in 1:N]
      osSw = [("SWAP", n, n + 1) for n in 1:(N - 1)]
      osCx = [("CX", n, n + 1) for n in 1:(N - 1)]
      osT = [("CCX", n, n + 2, n + 4) for n in 1:(N - 4)]
      os = vcat(osX, osSw, osZ, osCx, osT)

      s = siteinds("Qubit", N)
      gates = ops(os, s)

      ψ0 = MPS(s, "0")

      # Apply the gates
      ψ = apply(gates, ψ0; cutoff=1e-15, maxdim=100)

      prodψ = apply(gates, prod(ψ0))
      @test prod(ψ) ≈ prodψ rtol = 1e-4
    end

    @testset "With fermions" begin
      N = 3

      s = siteinds("Fermion", N; conserve_qns=true)

      # Ground state |000⟩
      ψ000 = MPS(s, "0")

      # Start state |011⟩
      ψ011 = MPS(s, n -> n == 2 || n == 3 ? "1" : "0")

      # Reference state |110⟩
      ψ110 = MPS(s, n -> n == 1 || n == 2 ? "1" : "0")

      function ITensors.op(::OpName"CdagC", ::SiteType, s1::Index, s2::Index)
        return op("Cdag", s1) * op("C", s2)
      end

      os = [("CdagC", 1, 3)]
      Os = ops(os, s)

      # Results in -|110⟩
      ψ1 = product(Os, ψ011; cutoff=1e-15)

      @test inner(ψ1, ψ110) == -1

      a = OpSum()
      a += "Cdag", 1, "C", 3
      H = MPO(a, s)

      # Results in -|110⟩
      ψ2 = noprime(contract(H, ψ011; cutoff=1e-15))

      @test inner(ψ2, ψ110) == -1
    end

    @testset "Spinless fermion (gate evolution)" begin
      N = 6

      s = siteinds("Fermion", N; conserve_qns=true)

      # Starting state
      ψ0 = MPS(s, n -> isodd(n) ? "0" : "1")

      t = 1.0
      U = 1.0
      ampo = OpSum()
      for b in 1:(N - 1)
        ampo .+= -t, "Cdag", b, "C", b + 1
        ampo .+= -t, "Cdag", b + 1, "C", b
        ampo .+= U, "N", b, "N", b + 1
      end
      H = MPO(ampo, s)

      sweeps = Sweeps(6)
      maxdim!(sweeps, 10, 20, 40)
      cutoff!(sweeps, 1E-12)
      energy, ψ0 = dmrg(H, ψ0, sweeps; outputlevel=0)

      function ITensors.op(::OpName"CdagC", ::SiteType, s1::Index, s2::Index)
        return op("Cdag", s1) * op("C", s2)
      end

      function ITensors.op(
        ::OpName"CCCC", ::SiteType, s1::Index, s2::Index, s3::Index, s4::Index
      )
        return -1 * op("Cdag", s1) * op("Cdag", s2) * op("C", s3) * op("C", s4)
      end

      for i in 1:(N - 1), j in (i + 1):N
        G1 = op("CdagC", s, i, j)

        @disable_warn_order begin
          G2 = op("Cdag", s, i)
          for n in (i + 1):(j - 1)
            G2 *= op("F", s, n)
          end
          G2 *= op("C", s, j)
        end

        ampo = OpSum()
        ampo += "Cdag", i, "C", j
        G3 = MPO(ampo, s)

        A_OP = prod(product(G1, ψ0; cutoff=1e-6))

        A_OPS = noprime(G2 * prod(ψ0))

        A_MPO = noprime(prod(contract(G3, ψ0; cutoff=1e-6)))

        @test A_OP ≈ A_OPS
        @test A_OP ≈ A_MPO
      end

      for i in 1:(N - 3), j in (i + 1):(N - 2), k in (j + 1):(N - 1), l in (k + 1):N
        G1 = op("CCCC", s, i, j, k, l)
        @disable_warn_order begin
          G2 = -1 * op("Cdag", s, i)
          for n in (i + 1):(j - 1)
            G2 *= op("F", s, n)
          end
          G2 *= op("Cdag", s, j)
          for n in (j + 1):(k - 1)
            G2 *= op("Id", s, n)
          end
          G2 *= op("C", s, k)
          for n in (k + 1):(l - 1)
            G2 *= op("F", s, n)
          end
          G2 *= op("C", s, l)

          ampo = OpSum()
          ampo += "Cdag", i, "Cdag", j, "C", k, "C", l
          G3 = MPO(ampo, s)

          A_OP = prod(product(G1, ψ0; cutoff=1e-16))

          A_OPS = noprime(G2 * prod(ψ0))

          A_MPO = noprime(prod(contract(G3, ψ0; cutoff=1e-16)))
        end
        @test A_OPS ≈ A_OP rtol = 1e-12
        @test A_MPO ≈ A_OP rtol = 1e-12
      end
    end

    @testset "Spinful Fermions (Electron) gate evolution" begin
      N = 8
      s = siteinds("Electron", N; conserve_qns=true)
      ψ0 = randomMPS(s, n -> isodd(n) ? "↑" : "↓")
      t = 1.0
      U = 1.0
      ampo = OpSum()
      for b in 1:(N - 1)
        ampo .+= -t, "Cdagup", b, "Cup", b + 1
        ampo .+= -t, "Cdagup", b + 1, "Cup", b
        ampo .+= -t, "Cdagdn", b, "Cdn", b + 1
        ampo .+= -t, "Cdagdn", b + 1, "Cdn", b
      end
      for n in 1:N
        ampo .+= U, "Nupdn", n
      end
      H = MPO(ampo, s)
      sweeps = Sweeps(6)
      maxdim!(sweeps, 10, 20, 40)
      cutoff!(sweeps, 1E-12)
      energy, ψ = dmrg(H, ψ0, sweeps; outputlevel=0)

      function ITensors.op(::OpName"CCup", ::SiteType"Electron", s1::Index, s2::Index)
        return op("Adagup * F", s1) * op("Aup", s2)
      end

      for i in 1:(N - 1), j in (i + 1):N
        ampo = OpSum()
        ampo += "Cdagup", i, "Cup", j
        G1 = MPO(ampo, s)
        G2 = op("CCup", s, i, j)
        A_MPO = prod(noprime(contract(G1, ψ; cutoff=1e-8)))
        A_OP = prod(product(G2, ψ; cutoff=1e-8))
        @test A_MPO ≈ A_OP atol = 1E-4
      end
    end
  end

  @testset "dense conversion of MPS" begin
    N = 4
    s = siteinds("S=1/2", N; conserve_qns=true)
    QM = randomMPS(s, ["Up", "Dn", "Up", "Dn"]; linkdims=4)
    qsz1 = scalar(QM[1] * op("Sz", s[1]) * dag(prime(QM[1], "Site")))

    M = dense(QM)
    @test !hasqns(M[1])
    sz1 = scalar(M[1] * op("Sz", removeqns(s[1])) * dag(prime(M[1], "Site")))
    @test sz1 ≈ qsz1
  end

  @testset "inner of MPS with more than one site Index" begin
    s = siteinds("S=½", 4)
    sout = addtags.(s, "out")
    sin = addtags.(s, "in")
    sinds = IndexSet.(sout, sin)
    Cs = combiner.(sinds)
    cinds = combinedind.(Cs)
    ψ = randomMPS(cinds)
    @test norm(ψ) ≈ 1
    @test inner(ψ, ψ) ≈ 1
    ψ .*= dag.(Cs)
    @test norm(ψ) ≈ 1
    @test inner(ψ, ψ) ≈ 1
  end

  @testset "inner(::MPS, ::MPO, ::MPS) with more than one site Index" begin
    N = 8
    s = siteinds("S=1/2", N)
    a = OpSum()
    for j in 1:(N - 1)
      a .+= 0.5, "S+", j, "S-", j + 1
      a .+= 0.5, "S-", j, "S+", j + 1
      a .+= "Sz", j, "Sz", j + 1
    end
    H = MPO(a, s)
    ψ = randomMPS(s, n -> isodd(n) ? "↑" : "↓"; linkdims=10)
    # Create MPO/MPS with pairs of sites merged
    H2 = MPO([H[b] * H[b + 1] for b in 1:2:N])
    ψ2 = MPS([ψ[b] * ψ[b + 1] for b in 1:2:N])
    @test inner(ψ, ψ) ≈ inner(ψ2, ψ2)
    @test inner(ψ', H, ψ) ≈ inner(ψ2', H2, ψ2)
    @test_throws ErrorException inner(ψ2, ψ2')
    @test_throws ErrorException inner(ψ2, H2, ψ2)
  end

  @testset "orthogonalize! on MPS with no link indices" begin
    N = 4
    s = siteinds("S=1/2", N)
    ψ = MPS([itensor(randn(ComplexF64, 2), s[n]) for n in 1:N])
    @test all(==(IndexSet()), linkinds(all, ψ))
    ϕ = orthogonalize(ψ, 2)
    @test ITensors.hasdefaultlinktags(ϕ)
    @test ortho_lims(ϕ) == 2:2
    @test ITensors.dist(ψ, ϕ) ≈ 0 atol = 1e-6
    # TODO: use this instead?
    # @test lognorm(ψ - ϕ) < -16
    @test norm(ψ - ϕ) ≈ 0 atol = 1e-6
  end

  @testset "MPO from MPS with no link indices" begin
    N = 4
    s = siteinds("S=1/2", N)
    ψ = MPS([itensor(randn(ComplexF64, 2), s[n]) for n in 1:N])
    ρ = outer(ψ', ψ)
    @test !ITensors.hasnolinkinds(ρ)
    @test inner(ρ, ρ) ≈ inner(ψ, ψ)^2
    @test inner(ψ', ρ, ψ) ≈ inner(ψ, ψ)^2

    # Deprecated syntax
    @test_deprecated outer(ψ, ψ)
    @test_deprecated inner(ψ, ψ')
    @test_deprecated inner(ψ, ρ, ψ)

    ρ = @test_deprecated MPO(ψ)
    @test !ITensors.hasnolinkinds(ρ)
    @test inner(ρ, ρ) ≈ inner(ψ, ψ)^2
    @test inner(ψ', ρ, ψ) ≈ inner(ψ, ψ)^2
  end

  @testset "Truncate MPO with no link indices" begin
    N = 4
    s = siteinds("S=1/2", N)
    M = MPO([itensor(randn(ComplexF64, 2, 2), s[n]', dag(s[n])) for n in 1:N])
    @test ITensors.hasnolinkinds(M)
    Mt = truncate(M; cutoff=1e-15)
    @test ITensors.hasdefaultlinktags(Mt)
    @test norm(M - Mt) ≈ 0 atol = 1e-12
  end
end

nothing
