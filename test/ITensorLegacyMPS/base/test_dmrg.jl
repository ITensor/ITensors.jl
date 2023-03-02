using ITensors, Test, Random

using ITensors: nsite, set_nsite!

@testset "Basic DMRG" begin
  @testset "Spin-one Heisenberg" begin
    N = 10
    sites = siteinds("S=1", N)

    os = OpSum()
    for j in 1:(N - 1)
      add!(os, "Sz", j, "Sz", j + 1)
      add!(os, 0.5, "S+", j, "S-", j + 1)
      add!(os, 0.5, "S-", j, "S+", j + 1)
    end
    H = MPO(os, sites)

    psi = randomMPS(sites)

    sweeps = Sweeps(3)
    @test length(sweeps) == 3
    maxdim!(sweeps, 10, 20, 40)
    mindim!(sweeps, 1, 10)
    cutoff!(sweeps, 1E-11)
    noise!(sweeps, 1E-10)
    str = split(sprint(show, sweeps), '\n')
    @test length(str) > 1
    energy, psi = dmrg(H, psi, sweeps; outputlevel=0)
    @test energy < -12.0
  end

  @testset "QN-conserving Spin-one Heisenberg" begin
    N = 10
    sites = siteinds("S=1", N; conserve_qns=true)

    os = OpSum()
    for j in 1:(N - 1)
      os += "Sz", j, "Sz", j + 1
      os += 0.5, "S+", j, "S-", j + 1
      os += 0.5, "S-", j, "S+", j + 1
    end
    H = MPO(os, sites)

    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    psi = randomMPS(sites, state; linkdims=4)

    sweeps = Sweeps(3)
    @test length(sweeps) == 3
    maxdim!(sweeps, 10, 20, 40)
    mindim!(sweeps, 1, 10)
    cutoff!(sweeps, 1E-11)
    noise!(sweeps, 1E-10)
    str = split(sprint(show, sweeps), '\n')
    @test length(str) > 1
    energy, psi = dmrg(H, psi, sweeps; outputlevel=0)
    @test energy < -12.0
  end

  @testset "QN-conserving Spin-one Heisenberg with disk caching" begin
    N = 10
    sites = siteinds("S=1", N; conserve_qns=true)

    os = OpSum()
    for j in 1:(N - 1)
      os += "Sz", j, "Sz", j + 1
      os += 0.5, "S+", j, "S-", j + 1
      os += 0.5, "S-", j, "S+", j + 1
    end
    H = MPO(os, sites)

    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    psi = randomMPS(sites, state; linkdims=4)

    sweeps = Sweeps(3)
    @test length(sweeps) == 3
    maxdim!(sweeps, 10, 20, 40)
    mindim!(sweeps, 1, 10)
    cutoff!(sweeps, 1E-11)
    noise!(sweeps, 1E-10)
    str = split(sprint(show, sweeps), '\n')
    @test length(str) > 1
    energy, psi = dmrg(H, psi, sweeps; outputlevel=0, write_when_maxdim_exceeds=15)
    @test energy < -12.0
  end

  @testset "ProjMPO with disk caching" begin
    N = 10
    sites = siteinds("S=1", N; conserve_qns=true)

    os = OpSum()
    for j in 1:(N - 1)
      os += "Sz", j, "Sz", j + 1
      os += 0.5, "S+", j, "S-", j + 1
      os += 0.5, "S-", j, "S+", j + 1
    end
    H = MPO(os, sites)

    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    psi = randomMPS(sites, state; linkdims=4)
    PH = ProjMPO(H)

    PHc = copy(PH)

    n = 4
    orthogonalize!(psi, n)
    position!(PH, psi, n)
    PHdisk = ITensors.disk(PH)

    @test length(PH) == N
    @test length(PHdisk) == N
    @test ITensors.site_range(PH) == n:(n + 1)
    @test eltype(PH) == Float64
    @test size(PH) == (3^2 * 4^2, 3^2 * 4^2)
    @test PH.lpos == n - 1
    @test PH.rpos == n + 2
    @test PHc.lpos == 0
    @test PHc.rpos == N + 1
    @test rproj(PH) ≈ rproj(PHdisk)
    @test PHdisk.LR isa ITensors.DiskVector{ITensor}
    @test PHdisk.LR[PHdisk.rpos] ≈ PHdisk.Rcache
    position!(PH, psi, N)
    @test PH.lpos == N - 1
  end

  @testset "ProjMPOSum DMRG with disk caching" begin
    N = 10
    sites = siteinds("S=1", N; conserve_qns=true)

    osA = OpSum()
    for j in 1:(N - 1)
      osA += "Sz", j, "Sz", j + 1
    end
    HA = MPO(osA, sites)

    osB = OpSum()
    for j in 1:(N - 1)
      osB += 0.5, "S+", j, "S-", j + 1
      osB += 0.5, "S-", j, "S+", j + 1
    end
    HB = MPO(osB, sites)

    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    psi = randomMPS(sites, state; linkdims=4)

    energy, psi = dmrg(
      [HA, HB], psi; nsweeps=3, maxdim=[10, 20, 30], write_when_maxdim_exceeds=10
    )
    @test energy < -12.0
  end

  @testset "ProjMPO: nsite" begin
    N = 10
    sites = siteinds("S=1", N)

    os1 = OpSum()
    for j in 1:(N - 1)
      os1 += 0.5, "S+", j, "S-", j + 1
      os1 += 0.5, "S-", j, "S+", j + 1
    end
    os2 = OpSum()
    for j in 1:(N - 1)
      os2 += "Sz", j, "Sz", j + 1
    end
    H1 = MPO(os1, sites)
    H2 = MPO(os2, sites)

    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    psi = randomMPS(sites, state; linkdims=4)
    PH1 = ProjMPO(H1)
    PH = ProjMPOSum([H1, H2])
    PH1c = copy(PH1)
    PHc = copy(PH)
    @test nsite(PH1) == 2
    @test nsite(PH) == 2
    @test nsite(PH1c) == 2
    @test nsite(PHc) == 2

    set_nsite!(PH1, 3)
    @test nsite(PH1) == 3
    @test nsite(PH1c) == 2
    @test nsite(PHc) == 2

    set_nsite!(PH, 4)
    @test nsite(PH) == 4
    @test nsite(PH1c) == 2
    @test nsite(PHc) == 2
  end

  @testset "Transverse field Ising" begin
    N = 32
    sites = siteinds("S=1/2", N)
    Random.seed!(432)
    psi0 = randomMPS(sites)

    os = OpSum()
    for j in 1:N
      j < N && add!(os, -1.0, "Z", j, "Z", j + 1)
      add!(os, -1.0, "X", j)
    end
    H = MPO(os, sites)

    sweeps = Sweeps(5)
    maxdim!(sweeps, 10, 20)
    cutoff!(sweeps, 1E-12)
    noise!(sweeps, 1E-10)
    energy, psi = dmrg(H, psi0, sweeps; outputlevel=0)

    # Exact energy for transverse field Ising model
    # with open boundary conditions at criticality
    energy_exact = 1.0 - 1.0 / sin(π / (4 * N + 2))
    @test abs((energy - energy_exact) / energy_exact) < 1e-4
  end

  @testset "Compact Sweeps syntax" begin
    N = 32
    sites = siteinds("S=1/2", N)
    Random.seed!(432)
    psi0 = randomMPS(sites)

    function ising(N; h=1.0)
      os = OpSum()
      for j in 1:N
        j < N && (os -= ("Z", j, "Z", j + 1))
        os -= h, "X", j
      end
      return os
    end

    h = 1.0
    H = MPO(ising(N; h=h), sites)
    energy, psi = dmrg(
      H, psi0; nsweeps=5, maxdim=[10, 20], cutoff=1e-12, noise=1e-10, outputlevel=0
    )

    energy_exact = 1.0 - 1.0 / sin(π / (4 * N + 2))
    @test abs((energy - energy_exact) / energy_exact) < 1e-4
  end

  @testset "Transverse field Ising, conserve Sz parity" begin
    N = 32
    sites = siteinds("S=1/2", N; conserve_szparity=true)
    Random.seed!(432)

    state = [isodd(j) ? "↑" : "↓" for j in 1:N]
    psi0 = randomMPS(sites, state)

    os = OpSum()
    for j in 1:N
      j < N && add!(os, -1.0, "X", j, "X", j + 1)
      add!(os, -1.0, "Z", j)
    end
    H = MPO(os, sites)

    sweeps = Sweeps(5)
    maxdim!(sweeps, 10, 20)
    cutoff!(sweeps, 1E-12)
    noise!(sweeps, 1E-10)
    energy, psi = dmrg(H, psi0, sweeps; outputlevel=0)

    # Exact energy for transverse field Ising model
    # with open boundary conditions at criticality
    energy_exact = 1.0 - 1.0 / sin(π / (4 * N + 2))
    @test abs((energy - energy_exact) / energy_exact) < 1e-4
  end

  @testset "DMRGObserver" begin

    # Test that basic constructors work
    observer = DMRGObserver()
    observer = DMRGObserver(; minsweeps=2, energy_tol=1E-4)

    # Test in a DMRG calculation
    N = 10
    sites = siteinds("S=1/2", N)
    Random.seed!(42)
    psi0 = randomMPS(sites)

    os = OpSum()
    for j in 1:(N - 1)
      os += -1, "Sz", j, "Sz", j + 1
    end
    for j in 1:N
      os += -0.2, "Sx", j
    end
    H = MPO(os, sites)

    sweeps = Sweeps(3)
    maxdim!(sweeps, 10)
    cutoff!(sweeps, 1E-12)

    observer = DMRGObserver(["Sz", "Sx"], sites)

    E, psi = dmrg(H, psi0, sweeps; observer=observer, outputlevel=0)
    @test length(measurements(observer)["Sz"]) == 3
    @test length(measurements(observer)["Sx"]) == 3
    @test all(length.(measurements(observer)["Sz"]) .== N)
    @test all(length.(measurements(observer)["Sx"]) .== N)
    @test length(energies(observer)) == 3
    @test length(truncerrors(observer)) == 3
    @test energies(observer)[end] == E
    @test all(truncerrors(observer) .< 1E-9)

    orthogonalize!(psi, 1)
    m = scalar(dag(psi[1]) * noprime(op(sites, "Sz", 1) * psi[1]))
    @test measurements(observer)["Sz"][end][1] ≈ m
  end

  @testset "Sum of MPOs (ProjMPOSum)" begin
    N = 10
    sites = siteinds("S=1", N)

    osZ = OpSum()
    for j in 1:(N - 1)
      osZ += "Sz", j, "Sz", j + 1
    end
    HZ = MPO(osZ, sites)

    osXY = OpSum()
    for j in 1:(N - 1)
      osXY += 0.5, "S+", j, "S-", j + 1
      osXY += 0.5, "S-", j, "S+", j + 1
    end
    HXY = MPO(osXY, sites)

    psi = randomMPS(sites)

    sweeps = Sweeps(3)
    maxdim!(sweeps, 10, 20, 40)
    mindim!(sweeps, 1, 10, 10)
    cutoff!(sweeps, 1E-11)
    noise!(sweeps, 1E-10)
    energy, psi = dmrg([HZ, HXY], psi, sweeps; outputlevel=0)
    @test energy < -12.0
  end

  @testset "Excited-state DMRG" begin
    N = 10
    weight = 15.0

    sites = siteinds("S=1", N)
    sites[1] = Index(2, "S=1/2,n=1,Site")
    sites[N] = Index(2, "S=1/2,n=$N,Site")

    os = OpSum()
    for j in 1:(N - 1)
      os += "Sz", j, "Sz", j + 1
      os += 0.5, "S+", j, "S-", j + 1
      os += 0.5, "S-", j, "S+", j + 1
    end
    H = MPO(os, sites)

    psi0i = randomMPS(sites; linkdims=10)

    sweeps = Sweeps(4)
    maxdim!(sweeps, 10, 20, 100, 100)
    cutoff!(sweeps, 1E-11)
    noise!(sweeps, 1E-10)

    energy0, psi0 = dmrg(H, psi0i, sweeps; outputlevel=0)
    @test energy0 < -11.5

    psi1i = randomMPS(sites; linkdims=10)
    energy1, psi1 = dmrg(H, [psi0], psi1i, sweeps; outputlevel=0, weight=weight)

    @test energy1 > energy0
    @test energy1 < -11.1

    @test inner(psi1, psi0) < 1E-5
  end

  @testset "Fermionic Hamiltonian" begin
    N = 10
    t1 = 1.0
    t2 = 0.5
    V = 0.2
    s = siteinds("Fermion", N; conserve_qns=true)

    state = fill(1, N)
    state[1] = 2
    state[3] = 2
    state[5] = 2
    state[7] = 2
    psi0 = productMPS(s, state)

    os = OpSum()
    for j in 1:(N - 1)
      os += -t1, "Cdag", j, "C", j + 1
      os += -t1, "Cdag", j + 1, "C", j
      os += V, "N", j, "N", j + 1
    end
    for j in 1:(N - 2)
      os += -t2, "Cdag", j, "C", j + 2
      os += -t2, "Cdag", j + 2, "C", j
    end
    H = MPO(os, s)

    sweeps = Sweeps(5)
    maxdim!(sweeps, 10, 20, 100, 100, 200)
    cutoff!(sweeps, 1E-8)
    noise!(sweeps, 1E-10)

    energy, psi = dmrg(H, psi0, sweeps; outputlevel=0)
    @test (-6.5 < energy < -6.4)
  end

  @testset "Hubbard model" begin
    N = 10
    Npart = 8
    t1 = 1.0
    U = 1.0
    V1 = 0.5
    sites = siteinds("Electron", N; conserve_qns=true)
    os = OpSum()
    for i in 1:N
      os += (U, "Nupdn", i)
    end
    for b in 1:(N - 1)
      os += -t1, "Cdagup", b, "Cup", b + 1
      os += -t1, "Cdagup", b + 1, "Cup", b
      os += -t1, "Cdagdn", b, "Cdn", b + 1
      os += -t1, "Cdagdn", b + 1, "Cdn", b
      os += V1, "Ntot", b, "Ntot", b + 1
    end
    H = MPO(os, sites)
    sweeps = Sweeps(6)
    maxdim!(sweeps, 50, 100, 200, 400, 800, 800)
    cutoff!(sweeps, 1E-10)
    state = ["Up", "Dn", "Dn", "Up", "Emp", "Up", "Up", "Emp", "Dn", "Dn"]
    psi0 = randomMPS(sites, state; linkdims=10)
    energy, psi = dmrg(H, psi0, sweeps; outputlevel=0)
    @test (-8.02 < energy < -8.01)
  end

  @testset "Input Without Ortho Center or Not at 1" begin
    N = 6
    sites = siteinds("S=1", N)

    os = OpSum()
    for j in 1:(N - 1)
      add!(os, "Sz", j, "Sz", j + 1)
      add!(os, 0.5, "S+", j, "S-", j + 1)
      add!(os, 0.5, "S-", j, "S+", j + 1)
    end
    H = MPO(os, sites)

    sweeps = Sweeps(1)
    maxdim!(sweeps, 10)
    cutoff!(sweeps, 1E-11)

    psi0 = randomMPS(sites; linkdims=4)

    # Test that input works with wrong ortho center:
    orthogonalize!(psi0, 5)
    energy, psi = dmrg(H, psi0, sweeps; outputlevel=0)

    # Test that input works with no ortho center:
    for j in 1:N
      psi0[j] = randomITensor(inds(psi0[j]))
    end
    energy, psi = dmrg(H, psi0, sweeps; outputlevel=0)
  end
end

nothing
