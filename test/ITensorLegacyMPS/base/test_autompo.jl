using ITensors, Test, Random, JLD2

include(joinpath(@__DIR__, "utils", "util.jl"))

function components_to_opsum(comps, n; reverse::Bool=true)
  opsum = OpSum()
  for (factor, operators, sites) in comps
    # reverse ordering for compatibility
    sites = reverse ? (n + 1) .- sites : sites
    sites_and_ops = [[Matrix(operator), site] for (operator, site) in zip(operators, sites)]
    sites_and_ops = [vcat(sites_and_ops...)...]
    opsum += factor, sites_and_ops...
  end
  return opsum
end

function isingMPO(sites)::MPO
  H = MPO(sites)
  N = length(H)
  link = Vector{Index}(undef, N + 1)
  for n in 1:(N + 1)
    link[n] = Index(3, "Link,Ising,l=$(n-1)")
  end
  for n in 1:N
    s = sites[n]
    ll = link[n]
    rl = link[n + 1]
    H[n] = ITensor(dag(ll), dag(s), s', rl)
    H[n] += setelt(ll => 1) * setelt(rl => 1) * op(sites, "Id", n)
    H[n] += setelt(ll => 3) * setelt(rl => 3) * op(sites, "Id", n)
    H[n] += setelt(ll => 2) * setelt(rl => 1) * op(sites, "Sz", n)
    H[n] += setelt(ll => 3) * setelt(rl => 2) * op(sites, "Sz", n)
  end
  LE = ITensor(link[1])
  LE[3] = 1.0
  RE = ITensor(dag(link[N + 1]))
  RE[1] = 1.0
  H[1] *= LE
  H[N] *= RE
  return H
end

function heisenbergMPO(sites, h::Vector{Float64}, onsite::String="Sz")::MPO
  H = MPO(sites)
  N = length(H)
  link = Vector{Index}(undef, N + 1)
  for n in 1:(N + 1)
    link[n] = Index(5, "Link,Heis,l=$(n-1)")
  end
  for n in 1:N
    s = sites[n]
    ll = link[n]
    rl = link[n + 1]
    H[n] = ITensor(ll, s, s', rl)
    H[n] += setelt(ll => 1) * setelt(rl => 1) * op(sites, "Id", n)
    H[n] += setelt(ll => 5) * setelt(rl => 5) * op(sites, "Id", n)
    H[n] += setelt(ll => 2) * setelt(rl => 1) * op(sites, "S+", n)
    H[n] += setelt(ll => 3) * setelt(rl => 1) * op(sites, "S-", n)
    H[n] += setelt(ll => 4) * setelt(rl => 1) * op(sites, "Sz", n)
    H[n] += setelt(ll => 5) * setelt(rl => 2) * op(sites, "S-", n) * 0.5
    H[n] += setelt(ll => 5) * setelt(rl => 3) * op(sites, "S+", n) * 0.5
    H[n] += setelt(ll => 5) * setelt(rl => 4) * op(sites, "Sz", n)
    H[n] += setelt(ll => 5) * setelt(rl => 1) * op(sites, onsite, n) * h[n]
  end
  H[1] *= setelt(link[1] => 5)
  H[N] *= setelt(link[N + 1] => 1)
  return H
end

function NNheisenbergMPO(sites, J1::Float64, J2::Float64)::MPO
  H = MPO(sites)
  N = length(H)
  link = Vector{Index}(undef, N + 1)
  if hasqns(sites[1])
    for n in 1:(N + 1)
      link[n] = Index(
        [
          QN() => 1,
          QN("Sz", -2) => 1,
          QN("Sz", +2) => 1,
          QN() => 1,
          QN("Sz", -2) => 1,
          QN("Sz", +2) => 1,
          QN() => 2,
        ],
        "Link,H,l=$(n-1)",
      )
    end
  else
    for n in 1:(N + 1)
      link[n] = Index(8, "Link,H,l=$(n-1)")
    end
  end
  for n in 1:N
    s = sites[n]
    ll = dag(link[n])
    rl = link[n + 1]
    H[n] = ITensor(ll, dag(s), s', rl)
    H[n] += onehot(ll => 1) * onehot(rl => 1) * op(sites, "Id", n)
    H[n] += onehot(ll => 8) * onehot(rl => 8) * op(sites, "Id", n)

    H[n] += onehot(ll => 2) * onehot(rl => 1) * op(sites, "S-", n)
    H[n] += onehot(ll => 5) * onehot(rl => 2) * op(sites, "Id", n)
    H[n] += onehot(ll => 8) * onehot(rl => 2) * op(sites, "S+", n) * J1 / 2
    H[n] += onehot(ll => 8) * onehot(rl => 5) * op(sites, "S+", n) * J2 / 2

    H[n] += onehot(ll => 3) * onehot(rl => 1) * op(sites, "S+", n)
    H[n] += onehot(ll => 6) * onehot(rl => 3) * op(sites, "Id", n)
    H[n] += onehot(ll => 8) * onehot(rl => 3) * op(sites, "S-", n) * J1 / 2
    H[n] += onehot(ll => 8) * onehot(rl => 6) * op(sites, "S-", n) * J2 / 2

    H[n] += onehot(ll => 4) * onehot(rl => 1) * op(sites, "Sz", n)
    H[n] += onehot(ll => 7) * onehot(rl => 4) * op(sites, "Id", n)
    H[n] += onehot(ll => 8) * onehot(rl => 4) * op(sites, "Sz", n) * J1
    H[n] += onehot(ll => 8) * onehot(rl => 7) * op(sites, "Sz", n) * J2
  end
  H[1] *= onehot(link[1] => 8)
  H[N] *= onehot(dag(link[N + 1]) => 1)
  return H
end

function threeSiteIsingMPO(sites, h::Vector{Float64})::MPO
  H = MPO(sites)
  N = length(H)
  link = Vector{Index}(undef, N + 1)
  for n in 1:(N + 1)
    link[n] = Index(4, "Link,l=$(n-1)")
  end
  for n in 1:N
    s = sites[n]
    ll = link[n]
    rl = link[n + 1]
    H[n] = ITensor(ll, s, s', rl)
    H[n] += setelt(ll => 1) * setelt(rl => 1) * op(sites, "Id", n)
    H[n] += setelt(ll => 4) * setelt(rl => 4) * op(sites, "Id", n)
    H[n] += setelt(ll => 2) * setelt(rl => 1) * op(sites, "Sz", n)
    H[n] += setelt(ll => 3) * setelt(rl => 2) * op(sites, "Sz", n)
    H[n] += setelt(ll => 4) * setelt(rl => 3) * op(sites, "Sz", n)
    H[n] += setelt(ll => 4) * setelt(rl => 1) * op(sites, "Sx", n) * h[n]
  end
  H[1] *= setelt(link[1] => 4)
  H[N] *= setelt(link[N + 1] => 1)
  return H
end

function fourSiteIsingMPO(sites)::MPO
  H = MPO(sites)
  N = length(H)
  link = Vector{Index}(undef, N + 1)
  for n in 1:(N + 1)
    link[n] = Index(5, "Link,l=$(n-1)")
  end
  for n in 1:N
    s = sites[n]
    ll = link[n]
    rl = link[n + 1]
    H[n] = ITensor(ll, s, s', rl)
    H[n] += setelt(ll => 1) * setelt(rl => 1) * op(sites, "Id", n)
    H[n] += setelt(ll => 5) * setelt(rl => 5) * op(sites, "Id", n)
    H[n] += setelt(ll => 2) * setelt(rl => 1) * op(sites, "Sz", n)
    H[n] += setelt(ll => 3) * setelt(rl => 2) * op(sites, "Sz", n)
    H[n] += setelt(ll => 4) * setelt(rl => 3) * op(sites, "Sz", n)
    H[n] += setelt(ll => 5) * setelt(rl => 4) * op(sites, "Sz", n)
  end
  H[1] *= setelt(link[1] => 5)
  H[N] *= setelt(link[N + 1] => 1)
  return H
end

@testset "OpSum" begin
  N = 10

  @test !ITensors.using_auto_fermion()

  @testset "Show MPOTerm" begin
    os = OpSum()
    add!(os, "Sz", 1, "Sz", 2)
    @test length(sprint(show, os[1])) > 1
  end

  @testset "Multisite operator" begin
    os = OpSum()
    os += ("CX", 1, 2)
    os += (2.3, "R", 3, 4, "S", 2)
    os += ("X", 3)
    @test length(os) == 3
    @test coefficient(os[1]) == 1
    @test length(os[1]) == 1
    @test ITensors.which_op(os[1][1]) == "CX"
    @test ITensors.sites(os[1][1]) == (1, 2)
    @test coefficient(os[2]) == 2.3
    @test length(os[2]) == 2
    @test ITensors.which_op(os[2][1]) == "R"
    @test ITensors.sites(os[2][1]) == (3, 4)
    @test ITensors.which_op(os[2][2]) == "S"
    @test ITensors.sites(os[2][2]) == (2,)
    @test coefficient(os[3]) == 1
    @test length(os[3]) == 1
    @test ITensors.which_op(os[3][1]) == "X"
    @test ITensors.sites(os[3][1]) == (3,)

    os = OpSum() + ("CX", 1, 2)
    @test length(os) == 1
    @test coefficient(os[1]) == 1
    @test length(os[1]) == 1
    @test ITensors.which_op(os[1][1]) == "CX"
    @test ITensors.sites(os[1][1]) == (1, 2)

    # Coordinate
    os = OpSum() + ("X", (1, 2))
    @test length(os) == 1
    @test coefficient(os[1]) == 1
    @test length(os[1]) == 1
    @test ITensors.which_op(os[1][1]) == "X"
    @test ITensors.sites(os[1][1]) == ((1, 2),)

    os = OpSum() + ("CX", 1, 2, (ϕ=π / 3,))
    @test length(os) == 1
    @test coefficient(os[1]) == 1
    @test length(os[1]) == 1
    @test ITensors.which_op(os[1][1]) == "CX"
    @test ITensors.sites(os[1][1]) == (1, 2)
    @test ITensors.params(os[1][1]) == (ϕ=π / 3,)

    os = OpSum() + ("CX", 1, 2, (ϕ=π / 3,), "CZ", 3, 4, (θ=π / 2,))
    @test length(os) == 1
    @test coefficient(os[1]) == 1
    @test length(os[1]) == 2
    @test ITensors.which_op(os[1][1]) == "CX"
    @test ITensors.sites(os[1][1]) == (1, 2)
    @test ITensors.params(os[1][1]) == (ϕ=π / 3,)
    @test ITensors.which_op(os[1][2]) == "CZ"
    @test ITensors.sites(os[1][2]) == (3, 4)
    @test ITensors.params(os[1][2]) == (θ=π / 2,)

    os = OpSum() + ("CX", (ϕ=π / 3,), 1, 2, "CZ", (θ=π / 2,), 3, 4)
    @test length(os) == 1
    @test coefficient(os[1]) == 1
    @test length(os[1]) == 2
    @test ITensors.which_op(os[1][1]) == "CX"
    @test ITensors.sites(os[1][1]) == (1, 2)
    @test ITensors.params(os[1][1]) == (ϕ=π / 3,)
    @test ITensors.which_op(os[1][2]) == "CZ"
    @test ITensors.sites(os[1][2]) == (3, 4)
    @test ITensors.params(os[1][2]) == (θ=π / 2,)

    os = OpSum() + ("CX", 1, 2, (ϕ=π / 3,))
    @test length(os) == 1
    @test coefficient(os[1]) == 1
    @test length(os[1]) == 1
    @test ITensors.which_op(os[1][1]) == "CX"
    @test ITensors.sites(os[1][1]) == (1, 2)
    @test ITensors.params(os[1][1]) == (ϕ=π / 3,)

    os = OpSum() + (1 + 2im, "CRz", (ϕ=π / 3,), 1, 2)
    @test length(os) == 1
    @test coefficient(os[1]) == 1 + 2im
    @test length(os[1]) == 1
    @test ITensors.which_op(os[1][1]) == "CRz"
    @test ITensors.sites(os[1][1]) == (1, 2)
    @test ITensors.params(os[1][1]) == (ϕ=π / 3,)

    os = OpSum() + ("CRz", (ϕ=π / 3,), 1, 2)
    @test length(os) == 1
    @test coefficient(os[1]) == 1
    @test length(os[1]) == 1
    @test ITensors.which_op(os[1][1]) == "CRz"
    @test ITensors.sites(os[1][1]) == (1, 2)
    @test ITensors.params(os[1][1]) == (ϕ=π / 3,)
  end

  @testset "Show OpSum" begin
    os = OpSum()
    add!(os, "Sz", 1, "Sz", 2)
    add!(os, "Sz", 2, "Sz", 3)
    @test length(sprint(show, os)) > 1
  end

  @testset "OpSum algebra" begin
    n = 5
    sites = siteinds("S=1/2", n)
    O1 = OpSum()
    for j in 1:(n - 1)
      O1 += "Sz", j, "Sz", j + 1
    end
    O2 = OpSum()
    for j in 1:n
      O2 += "Sx", j
    end
    O = O1 + 2 * O2
    @test length(O) == 2 * n - 1
    H1 = MPO(O1, sites)
    H2 = MPO(O2, sites)
    H = H1 + 2 * H2
    @test prod(MPO(O, sites)) ≈ prod(H)

    O = O1 - 2 * O2
    @test length(O) == 2 * n - 1
    H1 = MPO(O1, sites)
    H2 = MPO(O2, sites)
    H = H1 - 2 * H2
    @test prod(MPO(O, sites)) ≈ prod(H)

    O = O1 - O2 / 2
    @test length(O) == 2 * n - 1
    H1 = MPO(O1, sites)
    H2 = MPO(O2, sites)
    H = H1 - H2 / 2
    @test prod(MPO(O, sites)) ≈ prod(H)
  end

  @testset "Single creation op" begin
    os = OpSum()
    add!(os, "Adagup", 3)
    sites = siteinds("Electron", N)
    W = MPO(os, sites)
    psi = makeRandomMPS(sites)
    cdu_psi = copy(psi)
    cdu_psi[3] = noprime(cdu_psi[3] * op(sites, "Adagup", 3))
    @test inner(psi', W, psi) ≈ inner(cdu_psi, psi)
  end

  @testset "Ising" begin
    os = OpSum()
    for j in 1:(N - 1)
      os += "Sz", j, "Sz", j + 1
    end
    sites = siteinds("S=1/2", N)
    Ha = MPO(os, sites)
    @test ITensors.scalartype(Ha) <: Float64
    He = isingMPO(sites)
    psi = makeRandomMPS(sites)
    Oa = inner(psi', Ha, psi)
    Oe = inner(psi', He, psi)
    @test Oa ≈ Oe

    H_complex = MPO(ComplexF64, os, sites)
    @test ITensors.scalartype(H_complex) <: ComplexF64
    @test H_complex ≈ Ha
  end

  @testset "Ising" begin
    os = OpSum()
    for j in 1:(N - 1)
      os -= "Sz", j, "Sz", j + 1
    end
    sites = siteinds("S=1/2", N)
    Ha = MPO(os, sites)
    He = -isingMPO(sites)
    psi = makeRandomMPS(sites)
    Oa = inner(psi', Ha, psi)
    Oe = inner(psi', He, psi)
    @test Oa ≈ Oe
  end

  @testset "Ising-Different Order" begin
    os = OpSum()
    for j in 1:(N - 1)
      os += "Sz", j, "Sz", j + 1
    end
    sites = siteinds("S=1/2", N)
    Ha = MPO(os, sites)
    He = isingMPO(sites)
    psi = makeRandomMPS(sites)
    Oa = inner(psi', Ha, psi)
    Oe = inner(psi', He, psi)
    @test Oa ≈ Oe
  end

  @testset "Heisenberg" begin
    os = OpSum()
    h = rand(N) #random magnetic fields
    for j in 1:(N - 1)
      os += "Sz", j, "Sz", j + 1
      os += 0.5, "S+", j, "S-", j + 1
      os += 0.5, "S-", j, "S+", j + 1
    end
    for j in 1:N
      os += h[j], "Sz", j
    end

    sites = siteinds("S=1/2", N)
    Ha = MPO(os, sites)
    He = heisenbergMPO(sites, h)
    psi = makeRandomMPS(sites)
    Oa = inner(psi', Ha, psi)
    Oe = inner(psi', He, psi)
    @test Oa ≈ Oe
  end

  @testset "Multiple Onsite Ops" begin
    sites = siteinds("S=1", N)
    os1 = OpSum()
    for j in 1:(N - 1)
      os1 += "Sz", j, "Sz", j + 1
      os1 += 0.5, "S+", j, "S-", j + 1
      os1 += 0.5, "S-", j, "S+", j + 1
    end
    for j in 1:N
      os1 += "Sz * Sz", j
    end
    Ha1 = MPO(os1, sites)

    os2 = OpSum()
    for j in 1:(N - 1)
      os2 += "Sz", j, "Sz", j + 1
      os2 += 0.5, "S+", j, "S-", j + 1
      os2 += 0.5, "S-", j, "S+", j + 1
    end
    for j in 1:N
      os2 += "Sz", j, "Sz", j
    end
    Ha2 = MPO(os2, sites)

    He = heisenbergMPO(sites, ones(N), "Sz * Sz")
    psi = makeRandomMPS(sites)
    Oe = inner(psi', He, psi)
    Oa1 = inner(psi', Ha1, psi)
    @test Oa1 ≈ Oe
    Oa2 = inner(psi', Ha2, psi)
    @test Oa2 ≈ Oe
  end

  @testset "Three-site ops" begin
    os = OpSum()
    # To test version of add! taking a coefficient
    add!(os, 1.0, "Sz", 1, "Sz", 2, "Sz", 3)
    @test length(os) == 1
    for j in 2:(N - 2)
      add!(os, "Sz", j, "Sz", j + 1, "Sz", j + 2)
    end
    h = ones(N)
    for j in 1:N
      add!(os, h[j], "Sx", j)
    end
    sites = siteinds("S=1/2", N)
    Ha = MPO(os, sites)
    He = threeSiteIsingMPO(sites, h)
    psi = makeRandomMPS(sites)
    Oa = inner(psi', Ha, psi)
    Oe = inner(psi', He, psi)
    @test Oa ≈ Oe
  end

  @testset "Four-site ops" begin
    os = OpSum()
    for j in 1:(N - 3)
      add!(os, "Sz", j, "Sz", j + 1, "Sz", j + 2, "Sz", j + 3)
    end
    sites = siteinds("S=1/2", N)
    Ha = MPO(os, sites)
    He = fourSiteIsingMPO(sites)
    psi = makeRandomMPS(sites)
    Oa = inner(psi', Ha, psi)
    Oe = inner(psi', He, psi)
    @test Oa ≈ Oe
  end

  @testset "Next-neighbor Heisenberg" begin
    os = OpSum()
    J1 = 1.0
    J2 = 0.5
    for j in 1:(N - 1)
      add!(os, J1, "Sz", j, "Sz", j + 1)
      add!(os, J1 * 0.5, "S+", j, "S-", j + 1)
      add!(os, J1 * 0.5, "S-", j, "S+", j + 1)
    end
    for j in 1:(N - 2)
      add!(os, J2, "Sz", j, "Sz", j + 2)
      add!(os, J2 * 0.5, "S+", j, "S-", j + 2)
      add!(os, J2 * 0.5, "S-", j, "S+", j + 2)
    end
    sites = siteinds("S=1/2", N)
    Ha = MPO(os, sites)

    He = NNheisenbergMPO(sites, J1, J2)
    psi = makeRandomMPS(sites)
    Oa = inner(psi', Ha, psi)
    Oe = inner(psi', He, psi)
    @test Oa ≈ Oe
    #@test maxlinkdim(Ha) == 8
  end

  @testset "Onsite Regression Test" begin
    sites = siteinds("S=1", 4)
    os = OpSum()
    add!(os, 0.5, "Sx", 1)
    add!(os, 0.5, "Sy", 1)
    H = MPO(os, sites)
    l = commonind(H[1], H[2])
    T = setelt(l => 1) * H[1]
    O = op(sites[1], "Sx") + op(sites[1], "Sy")
    @test norm(T - 0.5 * O) < 1E-8

    sites = siteinds("S=1", 2)
    os = OpSum()
    add!(os, 0.5im, "Sx", 1)
    add!(os, 0.5, "Sy", 1)
    H = MPO(os, sites)
    T = H[1] * H[2]
    O =
      im * op(sites[1], "Sx") * op(sites[2], "Id") + op(sites[1], "Sy") * op(sites[2], "Id")
    @test norm(T - 0.5 * O) < 1E-8
  end

  @testset "+ syntax" begin
    @testset "Single creation op" begin
      os = OpSum()
      os += "Adagup", 3
      sites = siteinds("Electron", N)
      W = MPO(os, sites)
      psi = makeRandomMPS(sites)
      cdu_psi = copy(psi)
      cdu_psi[3] = noprime(cdu_psi[3] * op(sites, "Adagup", 3))
      @test inner(psi', W, psi) ≈ inner(cdu_psi, psi)
    end

    @testset "Ising" begin
      os = OpSum()
      for j in 1:(N - 1)
        os += "Sz", j, "Sz", j + 1
      end
      sites = siteinds("S=1/2", N)
      Ha = MPO(os, sites)
      He = isingMPO(sites)
      psi = makeRandomMPS(sites)
      Oa = inner(psi', Ha, psi)
      Oe = inner(psi', He, psi)
      @test Oa ≈ Oe
    end

    @testset "Ising-Different Order" begin
      os = OpSum()
      for j in 1:(N - 1)
        os += "Sz", j + 1, "Sz", j
      end
      sites = siteinds("S=1/2", N)
      Ha = MPO(os, sites)
      He = isingMPO(sites)
      psi = makeRandomMPS(sites)
      Oa = inner(psi', Ha, psi)
      Oe = inner(psi', He, psi)
      @test Oa ≈ Oe
    end

    @testset "Heisenberg" begin
      os = OpSum()
      h = rand(N) #random magnetic fields
      for j in 1:(N - 1)
        os += "Sz", j, "Sz", j + 1
        os += 0.5, "S+", j, "S-", j + 1
        os += 0.5, "S-", j, "S+", j + 1
      end
      for j in 1:N
        os += h[j], "Sz", j
      end

      sites = siteinds("S=1/2", N)
      Ha = MPO(os, sites)
      He = heisenbergMPO(sites, h)
      psi = makeRandomMPS(sites)
      Oa = inner(psi', Ha, psi)
      Oe = inner(psi', He, psi)
      @test Oa ≈ Oe
    end

    @testset "Multiple Onsite Ops" begin
      sites = siteinds("S=1", N)
      os1 = OpSum()
      for j in 1:(N - 1)
        os1 += "Sz", j, "Sz", j + 1
        os1 += 0.5, "S+", j, "S-", j + 1
        os1 += 0.5, "S-", j, "S+", j + 1
      end
      for j in 1:N
        os1 += "Sz * Sz", j
      end
      Ha1 = MPO(os1, sites)

      os2 = OpSum()
      for j in 1:(N - 1)
        os2 += "Sz", j, "Sz", j + 1
        os2 += 0.5, "S+", j, "S-", j + 1
        os2 += 0.5, "S-", j, "S+", j + 1
      end
      for j in 1:N
        os2 += "Sz", j, "Sz", j
      end
      Ha2 = MPO(os2, sites)

      He = heisenbergMPO(sites, ones(N), "Sz * Sz")
      psi = makeRandomMPS(sites)
      Oe = inner(psi', He, psi)
      Oa1 = inner(psi', Ha1, psi)
      @test Oa1 ≈ Oe
      Oa2 = inner(psi', Ha2, psi)
      @test Oa2 ≈ Oe
    end

    @testset "Three-site ops" begin
      os = OpSum()
      # To test version of add! taking a coefficient
      os += 1.0, "Sz", 1, "Sz", 2, "Sz", 3
      @test length(os) == 1
      for j in 2:(N - 2)
        os += "Sz", j, "Sz", j + 1, "Sz", j + 2
      end
      h = ones(N)
      for j in 1:N
        os += h[j], "Sx", j
      end
      sites = siteinds("S=1/2", N)
      Ha = MPO(os, sites)
      He = threeSiteIsingMPO(sites, h)
      psi = makeRandomMPS(sites)
      Oa = inner(psi', Ha, psi)
      Oe = inner(psi', He, psi)
      @test Oa ≈ Oe
    end

    @testset "Four-site ops" begin
      os = OpSum()
      for j in 1:(N - 3)
        os += "Sz", j, "Sz", j + 1, "Sz", j + 2, "Sz", j + 3
      end
      sites = siteinds("S=1/2", N)
      Ha = MPO(os, sites)
      He = fourSiteIsingMPO(sites)
      psi = makeRandomMPS(sites)
      Oa = inner(psi', Ha, psi)
      Oe = inner(psi', He, psi)
      @test Oa ≈ Oe
    end

    @testset "Next-neighbor Heisenberg" begin
      os = OpSum()
      J1 = 1.0
      J2 = 0.5
      for j in 1:(N - 1)
        os += J1, "Sz", j, "Sz", j + 1
        os += J1 * 0.5, "S+", j, "S-", j + 1
        os += J1 * 0.5, "S-", j, "S+", j + 1
      end
      for j in 1:(N - 2)
        os += J2, "Sz", j, "Sz", j + 2
        os += J2 * 0.5, "S+", j, "S-", j + 2
        os += J2 * 0.5, "S-", j, "S+", j + 2
      end
      sites = siteinds("S=1/2", N)
      Ha = MPO(os, sites)

      He = NNheisenbergMPO(sites, J1, J2)
      psi = makeRandomMPS(sites)
      Oa = inner(psi', Ha, psi)
      Oe = inner(psi', He, psi)
      @test Oa ≈ Oe
      #@test maxlinkdim(Ha) == 8
    end

    #@testset "-= syntax" begin
    #  os = OpSum()
    #  os += (-1,"Sz",1,"Sz",2)
    #  os2 = OpSum()
    #  os2 -= ("Sz",1,"Sz",2)
    #  @test os == os2
    #end

    @testset "Onsite Regression Test" begin
      sites = siteinds("S=1", 4)
      os = OpSum()
      os += 0.5, "Sx", 1
      os += 0.5, "Sy", 1
      H = MPO(os, sites)
      l = commonind(H[1], H[2])
      T = setelt(l => 1) * H[1]
      O = op(sites[1], "Sx") + op(sites[1], "Sy")
      @test norm(T - 0.5 * O) < 1E-8

      sites = siteinds("S=1", 2)
      os = OpSum()
      os += 0.5im, "Sx", 1
      os += 0.5, "Sy", 1
      H = MPO(os, sites)
      T = H[1] * H[2]
      O =
        im * op(sites[1], "Sx") * op(sites[2], "Id") +
        op(sites[1], "Sy") * op(sites[2], "Id")
      @test norm(T - 0.5 * O) < 1E-8
    end
  end

  @testset ".+= and .-= syntax" begin

    #@testset ".-= syntax" begin
    #  os = OpSum()
    #  os .+= (-1,"Sz",1,"Sz",2)
    #  os2 = OpSum()
    #  os2 .-= ("Sz",1,"Sz",2)
    #  @test os == os2
    #end

    @testset "Single creation op" begin
      os = OpSum()
      os .+= "Adagup", 3
      sites = siteinds("Electron", N)
      W = MPO(os, sites)
      psi = makeRandomMPS(sites)
      cdu_psi = copy(psi)
      cdu_psi[3] = noprime(cdu_psi[3] * op(sites, "Adagup", 3))
      @test inner(psi', W, psi) ≈ inner(cdu_psi, psi)
    end

    @testset "Ising" begin
      os = OpSum()
      for j in 1:(N - 1)
        os .+= "Sz", j, "Sz", j + 1
      end
      sites = siteinds("S=1/2", N)
      Ha = MPO(os, sites)
      He = isingMPO(sites)
      psi = makeRandomMPS(sites)
      Oa = inner(psi', Ha, psi)
      Oe = inner(psi', He, psi)
      @test Oa ≈ Oe
    end

    @testset "Ising-Different Order" begin
      os = OpSum()
      for j in 1:(N - 1)
        os .+= "Sz", j + 1, "Sz", j
      end
      sites = siteinds("S=1/2", N)
      Ha = MPO(os, sites)
      He = isingMPO(sites)
      psi = makeRandomMPS(sites)
      Oa = inner(psi', Ha, psi)
      Oe = inner(psi', He, psi)
      @test Oa ≈ Oe
    end

    @testset "Heisenberg" begin
      os = OpSum()
      h = rand(N) #random magnetic fields
      for j in 1:(N - 1)
        os .+= "Sz", j, "Sz", j + 1
        os .+= 0.5, "S+", j, "S-", j + 1
        os .+= 0.5, "S-", j, "S+", j + 1
      end
      for j in 1:N
        os .+= h[j], "Sz", j
      end

      sites = siteinds("S=1/2", N)
      Ha = MPO(os, sites)
      He = heisenbergMPO(sites, h)
      psi = makeRandomMPS(sites)
      Oa = inner(psi', Ha, psi)
      Oe = inner(psi', He, psi)
      @test Oa ≈ Oe
    end

    @testset "Multiple Onsite Ops" begin
      sites = siteinds("S=1", N)
      os1 = OpSum()
      for j in 1:(N - 1)
        os1 .+= "Sz", j, "Sz", j + 1
        os1 .+= 0.5, "S+", j, "S-", j + 1
        os1 .+= 0.5, "S-", j, "S+", j + 1
      end
      for j in 1:N
        os1 .+= "Sz * Sz", j
      end
      Ha1 = MPO(os1, sites)

      os2 = OpSum()
      for j in 1:(N - 1)
        os2 .+= "Sz", j, "Sz", j + 1
        os2 .+= 0.5, "S+", j, "S-", j + 1
        os2 .+= 0.5, "S-", j, "S+", j + 1
      end
      for j in 1:N
        os2 .+= "Sz", j, "Sz", j
      end
      Ha2 = MPO(os2, sites)

      He = heisenbergMPO(sites, ones(N), "Sz * Sz")
      psi = makeRandomMPS(sites)
      Oe = inner(psi', He, psi)
      Oa1 = inner(psi', Ha1, psi)
      @test Oa1 ≈ Oe
      Oa2 = inner(psi', Ha2, psi)
      @test Oa2 ≈ Oe
    end

    @testset "Three-site ops" begin
      os = OpSum()
      # To test version of add! taking a coefficient
      os .+= 1.0, "Sz", 1, "Sz", 2, "Sz", 3
      @test length(os) == 1
      for j in 2:(N - 2)
        os .+= "Sz", j, "Sz", j + 1, "Sz", j + 2
      end
      h = ones(N)
      for j in 1:N
        os .+= h[j], "Sx", j
      end
      sites = siteinds("S=1/2", N)
      Ha = MPO(os, sites)
      He = threeSiteIsingMPO(sites, h)
      psi = makeRandomMPS(sites)
      Oa = inner(psi', Ha, psi)
      Oe = inner(psi', He, psi)
      @test Oa ≈ Oe
    end

    @testset "Four-site ops" begin
      os = OpSum()
      for j in 1:(N - 3)
        os .+= "Sz", j, "Sz", j + 1, "Sz", j + 2, "Sz", j + 3
      end
      sites = siteinds("S=1/2", N)
      Ha = MPO(os, sites)
      He = fourSiteIsingMPO(sites)
      psi = makeRandomMPS(sites)
      Oa = inner(psi', Ha, psi)
      Oe = inner(psi', He, psi)
      @test Oa ≈ Oe
    end

    @testset "Next-neighbor Heisenberg" begin
      os = OpSum()
      J1 = 1.0
      J2 = 0.5
      for j in 1:(N - 1)
        os .+= J1, "Sz", j, "Sz", j + 1
        os .+= J1 * 0.5, "S+", j, "S-", j + 1
        os .+= J1 * 0.5, "S-", j, "S+", j + 1
      end
      for j in 1:(N - 2)
        os .+= J2, "Sz", j, "Sz", j + 2
        os .+= J2 * 0.5, "S+", j, "S-", j + 2
        os .+= J2 * 0.5, "S-", j, "S+", j + 2
      end
      sites = siteinds("S=1/2", N; conserve_qns=true)
      Ha = MPO(os, sites)

      He = NNheisenbergMPO(sites, J1, J2)
      psi = randomMPS(sites, [isodd(n) ? "Up" : "Dn" for n in 1:N])
      Oa = inner(psi', Ha, psi)
      Oe = inner(psi', He, psi)
      @test Oa ≈ Oe
      #@test maxlinkdim(Ha) == 8
    end

    @testset "Onsite Regression Test" begin
      sites = siteinds("S=1", 4)
      os = OpSum()
      os .+= 0.5, "Sx", 1
      os .+= 0.5, "Sy", 1
      H = MPO(os, sites)
      l = commonind(H[1], H[2])
      T = setelt(l => 1) * H[1]
      O = op(sites[1], "Sx") + op(sites[1], "Sy")
      @test norm(T - 0.5 * O) < 1E-8

      sites = siteinds("S=1", 2)
      os = OpSum()
      os .+= 0.5im, "Sx", 1
      os .+= 0.5, "Sy", 1
      H = MPO(os, sites)
      T = H[1] * H[2]
      O =
        im * op(sites[1], "Sx") * op(sites[2], "Id") +
        op(sites[1], "Sy") * op(sites[2], "Id")
      @test norm(T - 0.5 * O) < 1E-8
    end
  end

  @testset "Fermionic Operators" begin
    N = 5
    s = siteinds("Fermion", N)

    a1 = OpSum()
    a1 += "Cdag", 1, "C", 3
    M1 = MPO(a1, s)

    a2 = OpSum()
    a2 += -1, "C", 3, "Cdag", 1
    M2 = MPO(a2, s)

    a3 = OpSum()
    a3 += "Cdag", 1, "N", 2, "C", 3
    M3 = MPO(a3, s)

    p011 = productMPS(s, [1, 2, 2, 1, 1])
    p110 = productMPS(s, [2, 2, 1, 1, 1])

    @test inner(p110', M1, p011) ≈ -1.0
    @test inner(p110', M2, p011) ≈ -1.0
    @test inner(p110', M3, p011) ≈ -1.0

    p001 = productMPS(s, [1, 1, 2, 1, 1])
    p100 = productMPS(s, [2, 1, 1, 1, 1])

    @test inner(p100', M1, p001) ≈ +1.0
    @test inner(p100', M2, p001) ≈ +1.0
    @test inner(p100', M3, p001) ≈ 0.0

    #
    # Repeat similar test but
    # with Electron sites
    # 

    s = siteinds("Electron", N; conserve_qns=true)

    a1 = OpSum()
    a1 += "Cdagup", 1, "Cup", 3
    M1 = MPO(a1, s)

    a2 = OpSum()
    a2 += -1, "Cdn", 3, "Cdagdn", 1
    M2 = MPO(a2, s)

    p0uu = productMPS(s, [1, 2, 2, 1, 1])
    puu0 = productMPS(s, [2, 2, 1, 1, 1])
    p0ud = productMPS(s, [1, 2, 3, 1, 1])
    pdu0 = productMPS(s, [3, 2, 1, 1, 1])
    p00u = productMPS(s, [1, 1, 2, 1, 1])
    pu00 = productMPS(s, [2, 1, 1, 1, 1])
    p00d = productMPS(s, [1, 1, 3, 1, 1])
    pd00 = productMPS(s, [3, 1, 1, 1, 1])

    @test inner(puu0', M1, p0uu) ≈ -1.0
    @test inner(pdu0', M2, p0ud) ≈ -1.0
    @test inner(pu00', M1, p00u) ≈ +1.0
    @test inner(pd00', M2, p00d) ≈ +1.0
  end

  @testset "Complex OpSum Coefs" begin
    N = 4

    for use_qn in [false, true]
      sites = siteinds("S=1/2", N; conserve_qns=use_qn)
      os = OpSum()
      for i in 1:(N - 1)
        os += +1im, "S+", i, "S-", i + 1
        os += -1im, "S-", i, "S+", i + 1
      end
      H = MPO(os, sites)
      psiud = productMPS(sites, [1, 2, 1, 2])
      psidu = productMPS(sites, [2, 1, 1, 2])
      @test inner(psiud', H, psidu) ≈ +1im
      @test inner(psidu', H, psiud) ≈ -1im
    end
  end

  @testset "Non-zero QN MPO" begin
    N = 4
    s = siteinds("Boson", N; conserve_qns=true)

    j = 3
    terms = OpSum()
    terms += "Adag", j
    W = MPO(terms, s)

    function op_mpo(sites, which_op, j)
      N = length(sites)
      ops = [n < j ? "Id" : (n > j ? "Id" : which_op) for n in 1:N]
      M = MPO([op(ops[n], sites[n]) for n in 1:length(sites)])
      q = flux(op(which_op, sites[j]))
      links = [Index([n < j ? q => 1 : QN() => 1], "Link,l=$n") for n in 1:N]
      for n in 1:(N - 1)
        M[n] *= onehot(links[n] => 1)
        M[n + 1] *= onehot(dag(links[n]) => 1)
      end
      return M
    end
    M = op_mpo(s, "Adag", j)

    @test norm(prod(W) - prod(M)) < 1E-10

    psi = randomMPS(s, [isodd(n) ? "1" : "0" for n in 1:length(s)]; linkdims=4)
    Mpsi = apply(M, psi; alg="naive")
    Wpsi = apply(M, psi; alg="naive")
    @test abs(inner(Mpsi, Wpsi) / inner(Mpsi, Mpsi) - 1.0) < 1E-10
  end

  @testset "Fermion OpSum Issue 514 Regression Test" begin
    N = 4
    s = siteinds("Electron", N; conserve_qns=true)
    os1 = OpSum()
    os2 = OpSum()

    os1 += "Nup", 1
    os2 += "Cdagup", 1, "Cup", 1

    M1 = MPO(os1, s)
    M2 = MPO(os2, s)

    H1 = M1[1] * M1[2] * M1[3] * M1[4]
    H2 = M2[1] * M2[2] * M2[3] * M2[4]

    @test norm(H1 - H2) ≈ 0.0
  end

  @testset "OpSum in-place modification regression test" begin
    N = 2
    t = 1.0
    os = OpSum()
    for n in 1:(N - 1)
      os .+= -t, "Cdag", n, "C", n + 1
      os .+= -t, "Cdag", n + 1, "C", n
    end
    s = siteinds("Fermion", N; conserve_qns=true)
    os_original = deepcopy(os)
    for i in 1:4
      MPO(os, s)
      @test os == os_original
    end
  end

  @testset "Accuracy Regression Test (Issue 725)" begin
    ITensors.space(::SiteType"HardCore") = 2

    ITensors.state(::StateName"0", ::SiteType"HardCore") = [1.0, 0.0]
    ITensors.state(::StateName"1", ::SiteType"HardCore") = [0.0, 1.0]

    function ITensors.op!(Op::ITensor, ::OpName"N", ::SiteType"HardCore", s::Index)
      return Op[s' => 2, s => 2] = 1
    end

    function ITensors.op!(Op::ITensor, ::OpName"Adag", ::SiteType"HardCore", s::Index)
      return Op[s' => 1, s => 2] = 1
    end

    function ITensors.op!(Op::ITensor, ::OpName"A", ::SiteType"HardCore", s::Index)
      return Op[s' => 2, s => 1] = 1
    end

    t = 1.0
    V1 = 1E-3
    V2 = 2E-5

    N = 20
    sites = siteinds("HardCore", N)

    os = OpSum()
    for j in 1:(N - 1)
      os += -t, "Adag", j, "A", j + 1
      os += -t, "A", j, "Adag", j + 1
      os += V1, "N", j, "N", j + 1
    end
    for j in 1:(N - 2)
      os += V2, "N", j, "N", j + 2
    end
    H = MPO(os, sites)
    psi0 = productMPS(sites, n -> isodd(n) ? "0" : "1")
    @test abs(inner(psi0', H, psi0) - 0.00018) < 1E-10
  end

  @testset "Matrix operator representation" begin
    dim = 4
    op = rand(dim, dim)
    opt = op'
    s = [Index(dim), Index(dim)]
    a = OpSum()
    a += 1.0, op + opt, 1
    a += 1.0, op + opt, 2
    mpoa = MPO(a, s)
    b = OpSum()
    b += 1.0, op, 1
    b += 1.0, opt, 1
    b += 1.0, op, 2
    b += 1.0, opt, 2
    mpob = MPO(b, s)
    @test mpoa ≈ mpob
  end

  @testset "Matrix operator representation - hashing bug" begin
    n = 4
    dim = 4
    s = siteinds(dim, n)
    o = rand(dim, dim)
    os = OpSum()
    for j in 1:(n - 1)
      os += copy(o), j, copy(o), j + 1
    end
    H1 = MPO(os, s)
    H2 = ITensor()
    H2 += op(o, s[1]) * op(o, s[2]) * op("I", s[3]) * op("I", s[4])
    H2 += op("I", s[1]) * op(o, s[2]) * op(o, s[3]) * op("I", s[4])
    H2 += op("I", s[1]) * op("I", s[2]) * op(o, s[3]) * op(o, s[4])
    @test contract(H1) ≈ H2
  end

  @testset "Matrix operator representation - hashing bug" begin
    file_path = joinpath(@__DIR__, "utils", "opsum_hash_bug.jld2")
    comps, n, dims = load(file_path, "comps", "n", "dims")
    s = [Index(d) for d in dims]
    for _ in 1:100
      os = components_to_opsum(comps, n)
      # Before defining `hash(::Op, h::UInt)`, this
      # would randomly throw an error due to
      # some hashing issue in `MPO(::OpSum, ...)`
      MPO(os, s)
    end
  end

  @testset "Operator with empty blocks - issue #963" begin
    sites = siteinds("Fermion", 2; conserve_qns=true)
    opsum1 = OpSum()
    for p in 1:2, q in 1:2, r in 1:2, s in 1:2
      opsum1 += "c†", p, "c†", q, "c", r, "c", s
    end
    H1 = MPO(opsum1, sites)
    opsum2 = OpSum()
    for p in 1:2, q in 1:2, r in 1:2, s in 1:2
      if !(p == q == r == s)
        opsum2 += "c†", p, "c†", q, "c", r, "c", s
      end
    end
    H2 = MPO(opsum2, sites)
    @test H1 ≈ H2
  end

  @testset "One-site ops bond dimension test" begin
    sites = siteinds("S=1/2", N)

    # one-site operator on every site
    os = OpSum()
    for j in 1:N
      os += "Z", j
    end
    H = MPO(os, sites)
    @test all(linkdims(H) .== 2)

    # one-site operator on a single site
    os = OpSum()
    os += "Z", rand(1:N)
    H = MPO(os, sites)
    @test all(linkdims(H) .<= 2)
    @test_broken all(linkdims(H) .== 1)
  end

  @testset "Regression test (Issue 1150): Zero blocks operator" begin
    N = 4
    sites = siteinds("Fermion", N; conserve_qns=true)
    os = OpSum()
    os += (1.111, "Cdag", 3, "Cdag", 4, "C", 2, "C", 1)
    os += (2.222, "Cdag", 4, "Cdag", 1, "C", 3, "C", 2)
    os += (3.333, "Cdag", 1, "Cdag", 4, "C", 4, "C", 1)
    os += (4.444, "Cdag", 2, "Cdag", 3, "C", 1, "C", 4)
    # The following operator has C on site 2 twice, resulting
    # in a local operator with no blocks (exactly zero),
    # causing a certain logical step in working out the column qn
    # to fail:
    os += (5.555, "Cdag", 4, "Cdag", 4, "C", 2, "C", 2)
    @test_nowarn H = MPO(os, sites)
  end
end

nothing
