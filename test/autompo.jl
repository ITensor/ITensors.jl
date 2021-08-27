using ITensors, Test, Random

include("util.jl")

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

#! format: off
# JuliaFormatter tries to change this line to:
# heisenbergMPO(sites, h::Vector{Float64}; onsite::String="Sz")
# so turn it off for this line.
function heisenbergMPO(sites, h::Vector{Float64}, onsite::String="Sz")::MPO
#! format: on
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
  for n in 1:(N + 1)
    link[n] = Index(8, "Link,H,l=$(n-1)")
  end
  for n in 1:N
    s = sites[n]
    ll = link[n]
    rl = link[n + 1]
    H[n] = ITensor(ll, s, s', rl)
    H[n] += setelt(ll => 1) * setelt(rl => 1) * op(sites, "Id", n)
    H[n] += setelt(ll => 8) * setelt(rl => 8) * op(sites, "Id", n)

    H[n] += setelt(ll => 2) * setelt(rl => 1) * op(sites, "S-", n)
    H[n] += setelt(ll => 5) * setelt(rl => 2) * op(sites, "Id", n)
    H[n] += setelt(ll => 8) * setelt(rl => 2) * op(sites, "S+", n) * J1 / 2
    H[n] += setelt(ll => 8) * setelt(rl => 5) * op(sites, "S+", n) * J2 / 2

    H[n] += setelt(ll => 3) * setelt(rl => 1) * op(sites, "S+", n)
    H[n] += setelt(ll => 6) * setelt(rl => 3) * op(sites, "Id", n)
    H[n] += setelt(ll => 8) * setelt(rl => 3) * op(sites, "S-", n) * J1 / 2
    H[n] += setelt(ll => 8) * setelt(rl => 6) * op(sites, "S-", n) * J2 / 2

    H[n] += setelt(ll => 4) * setelt(rl => 1) * op(sites, "Sz", n)
    H[n] += setelt(ll => 7) * setelt(rl => 4) * op(sites, "Id", n)
    H[n] += setelt(ll => 8) * setelt(rl => 4) * op(sites, "Sz", n) * J1
    H[n] += setelt(ll => 8) * setelt(rl => 7) * op(sites, "Sz", n) * J2
  end
  H[1] *= setelt(link[1] => 8)
  H[N] *= setelt(link[N + 1] => 1)
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
    ampo = OpSum()
    add!(ampo, "Sz", 1, "Sz", 2)
    @test length(sprint(show, ITensors.data(ampo)[1])) > 1
  end

  @testset "Show OpSum" begin
    ampo = OpSum()
    add!(ampo, "Sz", 1, "Sz", 2)
    add!(ampo, "Sz", 2, "Sz", 3)
    @test length(sprint(show, ampo)) > 1
  end

  @testset "Single creation op" begin
    ampo = OpSum()
    add!(ampo, "Adagup", 3)
    sites = siteinds("Electron", N)
    W = MPO(ampo, sites)
    psi = makeRandomMPS(sites)
    cdu_psi = copy(psi)
    cdu_psi[3] = noprime(cdu_psi[3] * op(sites, "Adagup", 3))
    @test inner(psi, W, psi) ≈ inner(cdu_psi, psi)
  end

  @testset "Ising" begin
    ampo = OpSum()
    for j in 1:(N - 1)
      ampo += "Sz", j, "Sz", j + 1
    end
    sites = siteinds("S=1/2", N)
    Ha = MPO(ampo, sites)
    He = isingMPO(sites)
    psi = makeRandomMPS(sites)
    Oa = inner(psi, Ha, psi)
    Oe = inner(psi, He, psi)
    @test Oa ≈ Oe
  end

  @testset "Ising-Different Order" begin
    ampo = OpSum()
    for j in 1:(N - 1)
      ampo += "Sz", j, "Sz", j + 1
    end
    sites = siteinds("S=1/2", N)
    Ha = MPO(ampo, sites)
    He = isingMPO(sites)
    psi = makeRandomMPS(sites)
    Oa = inner(psi, Ha, psi)
    Oe = inner(psi, He, psi)
    @test Oa ≈ Oe
  end

  @testset "Heisenberg" begin
    ampo = OpSum()
    h = rand(N) #random magnetic fields
    for j in 1:(N - 1)
      ampo += "Sz", j, "Sz", j + 1
      ampo += 0.5, "S+", j, "S-", j + 1
      ampo += 0.5, "S-", j, "S+", j + 1
    end
    for j in 1:N
      ampo += h[j], "Sz", j
    end

    sites = siteinds("S=1/2", N)
    Ha = MPO(ampo, sites)
    He = heisenbergMPO(sites, h)
    psi = makeRandomMPS(sites)
    Oa = inner(psi, Ha, psi)
    Oe = inner(psi, He, psi)
    @test Oa ≈ Oe
  end

  @testset "Multiple Onsite Ops" begin
    sites = siteinds("S=1", N)
    ampo1 = OpSum()
    for j in 1:(N - 1)
      ampo1 += "Sz", j, "Sz", j + 1
      ampo1 += 0.5, "S+", j, "S-", j + 1
      ampo1 += 0.5, "S-", j, "S+", j + 1
    end
    for j in 1:N
      ampo1 += "Sz*Sz", j
    end
    Ha1 = MPO(ampo1, sites)

    ampo2 = OpSum()
    for j in 1:(N - 1)
      ampo2 += "Sz", j, "Sz", j + 1
      ampo2 += 0.5, "S+", j, "S-", j + 1
      ampo2 += 0.5, "S-", j, "S+", j + 1
    end
    for j in 1:N
      ampo2 += "Sz", j, "Sz", j
    end
    Ha2 = MPO(ampo2, sites)

    He = heisenbergMPO(sites, ones(N), "Sz*Sz")
    psi = makeRandomMPS(sites)
    Oe = inner(psi, He, psi)
    Oa1 = inner(psi, Ha1, psi)
    @test Oa1 ≈ Oe
    Oa2 = inner(psi, Ha2, psi)
    @test Oa2 ≈ Oe
  end

  @testset "Three-site ops" begin
    ampo = OpSum()
    # To test version of add! taking a coefficient
    add!(ampo, 1.0, "Sz", 1, "Sz", 2, "Sz", 3)
    @test length(ITensors.data(ampo)) == 1
    for j in 2:(N - 2)
      add!(ampo, "Sz", j, "Sz", j + 1, "Sz", j + 2)
    end
    h = ones(N)
    for j in 1:N
      add!(ampo, h[j], "Sx", j)
    end
    sites = siteinds("S=1/2", N)
    Ha = MPO(ampo, sites)
    He = threeSiteIsingMPO(sites, h)
    psi = makeRandomMPS(sites)
    Oa = inner(psi, Ha, psi)
    Oe = inner(psi, He, psi)
    @test Oa ≈ Oe
  end

  @testset "Four-site ops" begin
    ampo = OpSum()
    for j in 1:(N - 3)
      add!(ampo, "Sz", j, "Sz", j + 1, "Sz", j + 2, "Sz", j + 3)
    end
    sites = siteinds("S=1/2", N)
    Ha = MPO(ampo, sites)
    He = fourSiteIsingMPO(sites)
    psi = makeRandomMPS(sites)
    Oa = inner(psi, Ha, psi)
    Oe = inner(psi, He, psi)
    @test Oa ≈ Oe
  end

  @testset "Next-neighbor Heisenberg" begin
    ampo = OpSum()
    J1 = 1.0
    J2 = 0.5
    for j in 1:(N - 1)
      add!(ampo, J1, "Sz", j, "Sz", j + 1)
      add!(ampo, J1 * 0.5, "S+", j, "S-", j + 1)
      add!(ampo, J1 * 0.5, "S-", j, "S+", j + 1)
    end
    for j in 1:(N - 2)
      add!(ampo, J2, "Sz", j, "Sz", j + 2)
      add!(ampo, J2 * 0.5, "S+", j, "S-", j + 2)
      add!(ampo, J2 * 0.5, "S-", j, "S+", j + 2)
    end
    sites = siteinds("S=1/2", N)
    Ha = MPO(ampo, sites)

    He = NNheisenbergMPO(sites, J1, J2)
    psi = makeRandomMPS(sites)
    Oa = inner(psi, Ha, psi)
    Oe = inner(psi, He, psi)
    @test Oa ≈ Oe
    #@test maxlinkdim(Ha) == 8
  end

  @testset "Onsite Regression Test" begin
    sites = siteinds("S=1", 4)
    ampo = OpSum()
    add!(ampo, 0.5, "Sx", 1)
    add!(ampo, 0.5, "Sy", 1)
    H = MPO(ampo, sites)
    l = commonind(H[1], H[2])
    T = setelt(l => 1) * H[1]
    O = op(sites[1], "Sx") + op(sites[1], "Sy")
    @test norm(T - 0.5 * O) < 1E-8

    sites = siteinds("S=1", 2)
    ampo = OpSum()
    add!(ampo, 0.5im, "Sx", 1)
    add!(ampo, 0.5, "Sy", 1)
    H = MPO(ampo, sites)
    T = H[1] * H[2]
    O =
      im * op(sites[1], "Sx") * op(sites[2], "Id") + op(sites[1], "Sy") * op(sites[2], "Id")
    @test norm(T - 0.5 * O) < 1E-8
  end

  @testset "+ syntax" begin
    @testset "Single creation op" begin
      ampo = OpSum()
      ampo += "Adagup", 3
      sites = siteinds("Electron", N)
      W = MPO(ampo, sites)
      psi = makeRandomMPS(sites)
      cdu_psi = copy(psi)
      cdu_psi[3] = noprime(cdu_psi[3] * op(sites, "Adagup", 3))
      @test inner(psi, W, psi) ≈ inner(cdu_psi, psi)
    end

    @testset "Ising" begin
      ampo = OpSum()
      for j in 1:(N - 1)
        ampo += "Sz", j, "Sz", j + 1
      end
      sites = siteinds("S=1/2", N)
      Ha = MPO(ampo, sites)
      He = isingMPO(sites)
      psi = makeRandomMPS(sites)
      Oa = inner(psi, Ha, psi)
      Oe = inner(psi, He, psi)
      @test Oa ≈ Oe
    end

    @testset "Ising-Different Order" begin
      ampo = OpSum()
      for j in 1:(N - 1)
        ampo += "Sz", j + 1, "Sz", j
      end
      sites = siteinds("S=1/2", N)
      Ha = MPO(ampo, sites)
      He = isingMPO(sites)
      psi = makeRandomMPS(sites)
      Oa = inner(psi, Ha, psi)
      Oe = inner(psi, He, psi)
      @test Oa ≈ Oe
    end

    @testset "Heisenberg" begin
      ampo = OpSum()
      h = rand(N) #random magnetic fields
      for j in 1:(N - 1)
        ampo += "Sz", j, "Sz", j + 1
        ampo += 0.5, "S+", j, "S-", j + 1
        ampo += 0.5, "S-", j, "S+", j + 1
      end
      for j in 1:N
        ampo += h[j], "Sz", j
      end

      sites = siteinds("S=1/2", N)
      Ha = MPO(ampo, sites)
      He = heisenbergMPO(sites, h)
      psi = makeRandomMPS(sites)
      Oa = inner(psi, Ha, psi)
      Oe = inner(psi, He, psi)
      @test Oa ≈ Oe
    end

    @testset "Multiple Onsite Ops" begin
      sites = siteinds("S=1", N)
      ampo1 = OpSum()
      for j in 1:(N - 1)
        ampo1 += "Sz", j, "Sz", j + 1
        ampo1 += 0.5, "S+", j, "S-", j + 1
        ampo1 += 0.5, "S-", j, "S+", j + 1
      end
      for j in 1:N
        ampo1 += "Sz*Sz", j
      end
      Ha1 = MPO(ampo1, sites)

      ampo2 = OpSum()
      for j in 1:(N - 1)
        ampo2 += "Sz", j, "Sz", j + 1
        ampo2 += 0.5, "S+", j, "S-", j + 1
        ampo2 += 0.5, "S-", j, "S+", j + 1
      end
      for j in 1:N
        ampo2 += "Sz", j, "Sz", j
      end
      Ha2 = MPO(ampo2, sites)

      He = heisenbergMPO(sites, ones(N), "Sz*Sz")
      psi = makeRandomMPS(sites)
      Oe = inner(psi, He, psi)
      Oa1 = inner(psi, Ha1, psi)
      @test Oa1 ≈ Oe
      Oa2 = inner(psi, Ha2, psi)
      @test Oa2 ≈ Oe
    end

    @testset "Three-site ops" begin
      ampo = OpSum()
      # To test version of add! taking a coefficient
      ampo += 1.0, "Sz", 1, "Sz", 2, "Sz", 3
      @test length(ITensors.data(ampo)) == 1
      for j in 2:(N - 2)
        ampo += "Sz", j, "Sz", j + 1, "Sz", j + 2
      end
      h = ones(N)
      for j in 1:N
        ampo += h[j], "Sx", j
      end
      sites = siteinds("S=1/2", N)
      Ha = MPO(ampo, sites)
      He = threeSiteIsingMPO(sites, h)
      psi = makeRandomMPS(sites)
      Oa = inner(psi, Ha, psi)
      Oe = inner(psi, He, psi)
      @test Oa ≈ Oe
    end

    @testset "Four-site ops" begin
      ampo = OpSum()
      for j in 1:(N - 3)
        ampo += "Sz", j, "Sz", j + 1, "Sz", j + 2, "Sz", j + 3
      end
      sites = siteinds("S=1/2", N)
      Ha = MPO(ampo, sites)
      He = fourSiteIsingMPO(sites)
      psi = makeRandomMPS(sites)
      Oa = inner(psi, Ha, psi)
      Oe = inner(psi, He, psi)
      @test Oa ≈ Oe
    end

    @testset "Next-neighbor Heisenberg" begin
      ampo = OpSum()
      J1 = 1.0
      J2 = 0.5
      for j in 1:(N - 1)
        ampo += J1, "Sz", j, "Sz", j + 1
        ampo += J1 * 0.5, "S+", j, "S-", j + 1
        ampo += J1 * 0.5, "S-", j, "S+", j + 1
      end
      for j in 1:(N - 2)
        ampo += J2, "Sz", j, "Sz", j + 2
        ampo += J2 * 0.5, "S+", j, "S-", j + 2
        ampo += J2 * 0.5, "S-", j, "S+", j + 2
      end
      sites = siteinds("S=1/2", N)
      Ha = MPO(ampo, sites)

      He = NNheisenbergMPO(sites, J1, J2)
      psi = makeRandomMPS(sites)
      Oa = inner(psi, Ha, psi)
      Oe = inner(psi, He, psi)
      @test Oa ≈ Oe
      #@test maxlinkdim(Ha) == 8
    end

    #@testset "-= syntax" begin
    #  ampo = OpSum()
    #  ampo += (-1,"Sz",1,"Sz",2)
    #  ampo2 = OpSum()
    #  ampo2 -= ("Sz",1,"Sz",2)
    #  @test ampo == ampo2
    #end

    @testset "Onsite Regression Test" begin
      sites = siteinds("S=1", 4)
      ampo = OpSum()
      ampo += 0.5, "Sx", 1
      ampo += 0.5, "Sy", 1
      H = MPO(ampo, sites)
      l = commonind(H[1], H[2])
      T = setelt(l => 1) * H[1]
      O = op(sites[1], "Sx") + op(sites[1], "Sy")
      @test norm(T - 0.5 * O) < 1E-8

      sites = siteinds("S=1", 2)
      ampo = OpSum()
      ampo += 0.5im, "Sx", 1
      ampo += 0.5, "Sy", 1
      H = MPO(ampo, sites)
      T = H[1] * H[2]
      O =
        im * op(sites[1], "Sx") * op(sites[2], "Id") +
        op(sites[1], "Sy") * op(sites[2], "Id")
      @test norm(T - 0.5 * O) < 1E-8
    end
  end

  @testset ".+= and .-= syntax" begin

    #@testset ".-= syntax" begin
    #  ampo = OpSum()
    #  ampo .+= (-1,"Sz",1,"Sz",2)
    #  ampo2 = OpSum()
    #  ampo2 .-= ("Sz",1,"Sz",2)
    #  @test ampo == ampo2
    #end

    @testset "Single creation op" begin
      ampo = OpSum()
      ampo .+= "Adagup", 3
      sites = siteinds("Electron", N)
      W = MPO(ampo, sites)
      psi = makeRandomMPS(sites)
      cdu_psi = copy(psi)
      cdu_psi[3] = noprime(cdu_psi[3] * op(sites, "Adagup", 3))
      @test inner(psi, W, psi) ≈ inner(cdu_psi, psi)
    end

    @testset "Ising" begin
      ampo = OpSum()
      for j in 1:(N - 1)
        ampo .+= "Sz", j, "Sz", j + 1
      end
      sites = siteinds("S=1/2", N)
      Ha = MPO(ampo, sites)
      He = isingMPO(sites)
      psi = makeRandomMPS(sites)
      Oa = inner(psi, Ha, psi)
      Oe = inner(psi, He, psi)
      @test Oa ≈ Oe
    end

    @testset "Ising-Different Order" begin
      ampo = OpSum()
      for j in 1:(N - 1)
        ampo .+= "Sz", j + 1, "Sz", j
      end
      sites = siteinds("S=1/2", N)
      Ha = MPO(ampo, sites)
      He = isingMPO(sites)
      psi = makeRandomMPS(sites)
      Oa = inner(psi, Ha, psi)
      Oe = inner(psi, He, psi)
      @test Oa ≈ Oe
    end

    @testset "Heisenberg" begin
      ampo = OpSum()
      h = rand(N) #random magnetic fields
      for j in 1:(N - 1)
        ampo .+= "Sz", j, "Sz", j + 1
        ampo .+= 0.5, "S+", j, "S-", j + 1
        ampo .+= 0.5, "S-", j, "S+", j + 1
      end
      for j in 1:N
        ampo .+= h[j], "Sz", j
      end

      sites = siteinds("S=1/2", N)
      Ha = MPO(ampo, sites)
      He = heisenbergMPO(sites, h)
      psi = makeRandomMPS(sites)
      Oa = inner(psi, Ha, psi)
      Oe = inner(psi, He, psi)
      @test Oa ≈ Oe
    end

    @testset "Multiple Onsite Ops" begin
      sites = siteinds("S=1", N)
      ampo1 = OpSum()
      for j in 1:(N - 1)
        ampo1 .+= "Sz", j, "Sz", j + 1
        ampo1 .+= 0.5, "S+", j, "S-", j + 1
        ampo1 .+= 0.5, "S-", j, "S+", j + 1
      end
      for j in 1:N
        ampo1 .+= "Sz*Sz", j
      end
      Ha1 = MPO(ampo1, sites)

      ampo2 = OpSum()
      for j in 1:(N - 1)
        ampo2 .+= "Sz", j, "Sz", j + 1
        ampo2 .+= 0.5, "S+", j, "S-", j + 1
        ampo2 .+= 0.5, "S-", j, "S+", j + 1
      end
      for j in 1:N
        ampo2 .+= "Sz", j, "Sz", j
      end
      Ha2 = MPO(ampo2, sites)

      He = heisenbergMPO(sites, ones(N), "Sz*Sz")
      psi = makeRandomMPS(sites)
      Oe = inner(psi, He, psi)
      Oa1 = inner(psi, Ha1, psi)
      @test Oa1 ≈ Oe
      Oa2 = inner(psi, Ha2, psi)
      @test Oa2 ≈ Oe
    end

    @testset "Three-site ops" begin
      ampo = OpSum()
      # To test version of add! taking a coefficient
      ampo .+= 1.0, "Sz", 1, "Sz", 2, "Sz", 3
      @test length(ITensors.data(ampo)) == 1
      for j in 2:(N - 2)
        ampo .+= "Sz", j, "Sz", j + 1, "Sz", j + 2
      end
      h = ones(N)
      for j in 1:N
        ampo .+= h[j], "Sx", j
      end
      sites = siteinds("S=1/2", N)
      Ha = MPO(ampo, sites)
      He = threeSiteIsingMPO(sites, h)
      psi = makeRandomMPS(sites)
      Oa = inner(psi, Ha, psi)
      Oe = inner(psi, He, psi)
      @test Oa ≈ Oe
    end

    @testset "Four-site ops" begin
      ampo = OpSum()
      for j in 1:(N - 3)
        ampo .+= "Sz", j, "Sz", j + 1, "Sz", j + 2, "Sz", j + 3
      end
      sites = siteinds("S=1/2", N)
      Ha = MPO(ampo, sites)
      He = fourSiteIsingMPO(sites)
      psi = makeRandomMPS(sites)
      Oa = inner(psi, Ha, psi)
      Oe = inner(psi, He, psi)
      @test Oa ≈ Oe
    end

    @testset "Next-neighbor Heisenberg" begin
      ampo = OpSum()
      J1 = 1.0
      J2 = 0.5
      for j in 1:(N - 1)
        ampo .+= J1, "Sz", j, "Sz", j + 1
        ampo .+= J1 * 0.5, "S+", j, "S-", j + 1
        ampo .+= J1 * 0.5, "S-", j, "S+", j + 1
      end
      for j in 1:(N - 2)
        ampo .+= J2, "Sz", j, "Sz", j + 2
        ampo .+= J2 * 0.5, "S+", j, "S-", j + 2
        ampo .+= J2 * 0.5, "S-", j, "S+", j + 2
      end
      sites = siteinds("S=1/2", N)
      Ha = MPO(ampo, sites)

      He = NNheisenbergMPO(sites, J1, J2)
      psi = makeRandomMPS(sites)
      Oa = inner(psi, Ha, psi)
      Oe = inner(psi, He, psi)
      @test Oa ≈ Oe
      #@test maxlinkdim(Ha) == 8
    end

    @testset "Onsite Regression Test" begin
      sites = siteinds("S=1", 4)
      ampo = OpSum()
      ampo .+= 0.5, "Sx", 1
      ampo .+= 0.5, "Sy", 1
      H = MPO(ampo, sites)
      l = commonind(H[1], H[2])
      T = setelt(l => 1) * H[1]
      O = op(sites[1], "Sx") + op(sites[1], "Sy")
      @test norm(T - 0.5 * O) < 1E-8

      sites = siteinds("S=1", 2)
      ampo = OpSum()
      ampo .+= 0.5im, "Sx", 1
      ampo .+= 0.5, "Sy", 1
      H = MPO(ampo, sites)
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

    @test inner(p110, M1, p011) ≈ -1.0
    @test inner(p110, M2, p011) ≈ -1.0
    @test inner(p110, M3, p011) ≈ -1.0

    p001 = productMPS(s, [1, 1, 2, 1, 1])
    p100 = productMPS(s, [2, 1, 1, 1, 1])

    @test inner(p100, M1, p001) ≈ +1.0
    @test inner(p100, M2, p001) ≈ +1.0
    @test inner(p100, M3, p001) ≈ 0.0

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

    @test inner(puu0, M1, p0uu) ≈ -1.0
    @test inner(pdu0, M2, p0ud) ≈ -1.0
    @test inner(pu00, M1, p00u) ≈ +1.0
    @test inner(pd00, M2, p00d) ≈ +1.0
  end

  @testset "Complex OpSum Coefs" begin
    N = 4

    for use_qn in [false, true]
      sites = siteinds("S=1/2", N; conserve_qns=use_qn)
      ampo = OpSum()
      for i in 1:(N - 1)
        ampo += +1im, "S+", i, "S-", i + 1
        ampo += -1im, "S-", i, "S+", i + 1
      end
      H = MPO(ampo, sites)
      psiud = productMPS(sites, [1, 2, 1, 2])
      psidu = productMPS(sites, [2, 1, 1, 2])
      @test inner(psiud, H, psidu) ≈ +1im
      @test inner(psidu, H, psiud) ≈ -1im
    end
  end

  @testset "Fermion OpSum Issue 514 Regression Test" begin
    N = 4
    s = siteinds("Electron", N; conserve_qns=true)
    ampo1 = OpSum()
    ampo2 = OpSum()

    ampo1 += "Nup", 1
    ampo2 += "Cdagup", 1, "Cup", 1

    M1 = MPO(ampo1, s)
    M2 = MPO(ampo2, s)

    H1 = M1[1] * M1[2] * M1[3] * M1[4]
    H2 = M2[1] * M2[2] * M2[3] * M2[4]

    @test norm(H1 - H2) ≈ 0.0
  end

  @testset "OpSum in-place modification regression test" begin
    N = 2
    t = 1.0
    ampo = OpSum()
    for n in 1:(N - 1)
      ampo .+= -t, "Cdag", n, "C", n + 1
      ampo .+= -t, "Cdag", n + 1, "C", n
    end
    s = siteinds("Fermion", N; conserve_qns=true)
    ampo_original = deepcopy(ampo)
    for i in 1:4
      MPO(ampo, s)
      @test ampo == ampo_original
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

    ampo = OpSum()
    for j in 1:(N - 1)
      ampo += -t, "Adag", j, "A", j + 1
      ampo += -t, "A", j, "Adag", j + 1
      ampo += V1, "N", j, "N", j + 1
    end
    for j in 1:(N - 2)
      ampo += V2, "N", j, "N", j + 2
    end
    H = MPO(ampo, sites)
    psi0 = productMPS(sites, n -> isodd(n) ? "0" : "1")
    @test abs(inner(psi0, H, psi0) - 0.00018) < 1E-10
  end
end

nothing
