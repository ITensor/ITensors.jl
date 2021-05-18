using ITensors, Test, Random

@testset "Basic DMRG" begin
  @testset "Spin-one Heisenberg" begin
    N = 10
    sites = siteinds("S=1", N)

    ampo = OpSum()
    for j in 1:(N - 1)
      add!(ampo, "Sz", j, "Sz", j + 1)
      add!(ampo, 0.5, "S+", j, "S-", j + 1)
      add!(ampo, 0.5, "S-", j, "S+", j + 1)
    end
    H = MPO(ampo, sites)

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

    ampo = OpSum()
    for j in 1:(N - 1)
      ampo += "Sz", j, "Sz", j + 1
      ampo += 0.5, "S+", j, "S-", j + 1
      ampo += 0.5, "S-", j, "S+", j + 1
    end
    H = MPO(ampo, sites)

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

    ampo = OpSum()
    for j in 1:(N - 1)
      ampo += "Sz", j, "Sz", j + 1
      ampo += 0.5, "S+", j, "S-", j + 1
      ampo += 0.5, "S-", j, "S+", j + 1
    end
    H = MPO(ampo, sites)

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

    ampo = OpSum()
    for j in 1:(N - 1)
      ampo += "Sz", j, "Sz", j + 1
      ampo += 0.5, "S+", j, "S-", j + 1
      ampo += 0.5, "S-", j, "S+", j + 1
    end
    H = MPO(ampo, sites)

    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    psi = randomMPS(sites, state; linkdims=4)
    PH = ProjMPO(H)

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
    @test rproj(PH) ≈ rproj(PHdisk)
    @test PHdisk.LR isa ITensors.DiskVector{ITensor}
    @test PHdisk.LR[PHdisk.rpos] ≈ PHdisk.Rcache
    position!(PH, psi, N)
    @test PH.lpos == N - 1
  end

  @testset "Transverse field Ising" begin
    N = 32
    sites = siteinds("S=1/2", N)
    Random.seed!(432)
    psi0 = randomMPS(sites)

    ampo = OpSum()
    for j in 1:N
      j < N && add!(ampo, -1.0, "Z", j, "Z", j + 1)
      add!(ampo, -1.0, "X", j)
    end
    H = MPO(ampo, sites)

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

  @testset "Transverse field Ising, conserve Sz parity" begin
    N = 32
    sites = siteinds("S=1/2", N; conserve_szparity=true)
    Random.seed!(432)

    state = [isodd(j) ? "↑" : "↓" for j in 1:N]
    psi0 = randomMPS(sites, state)

    ampo = OpSum()
    for j in 1:N
      j < N && add!(ampo, -1.0, "X", j, "X", j + 1)
      add!(ampo, -1.0, "Z", j)
    end
    H = MPO(ampo, sites)

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

    ampo = OpSum()
    for j in 1:(N - 1)
      ampo += -1, "Sz", j, "Sz", j + 1
    end
    for j in 1:N
      ampo += -0.2, "Sx", j
    end
    H = MPO(ampo, sites)

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

    ampoZ = OpSum()
    for j in 1:(N - 1)
      ampoZ += "Sz", j, "Sz", j + 1
    end
    HZ = MPO(ampoZ, sites)

    ampoXY = OpSum()
    for j in 1:(N - 1)
      ampoXY += 0.5, "S+", j, "S-", j + 1
      ampoXY += 0.5, "S-", j, "S+", j + 1
    end
    HXY = MPO(ampoXY, sites)

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

    ampo = OpSum()
    for j in 1:(N - 1)
      ampo += "Sz", j, "Sz", j + 1
      ampo += 0.5, "S+", j, "S-", j + 1
      ampo += 0.5, "S-", j, "S+", j + 1
    end
    H = MPO(ampo, sites)

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

    ampo = OpSum()
    for j in 1:(N - 1)
      ampo += -t1, "Cdag", j, "C", j + 1
      ampo += -t1, "Cdag", j + 1, "C", j
      ampo += V, "N", j, "N", j + 1
    end
    for j in 1:(N - 2)
      ampo += -t2, "Cdag", j, "C", j + 2
      ampo += -t2, "Cdag", j + 2, "C", j
    end
    H = MPO(ampo, s)

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
    ampo = OpSum()
    for i in 1:N
      ampo += (U, "Nupdn", i)
    end
    for b in 1:(N - 1)
      ampo += -t1, "Cdagup", b, "Cup", b + 1
      ampo += -t1, "Cdagup", b + 1, "Cup", b
      ampo += -t1, "Cdagdn", b, "Cdn", b + 1
      ampo += -t1, "Cdagdn", b + 1, "Cdn", b
      ampo += V1, "Ntot", b, "Ntot", b + 1
    end
    H = MPO(ampo, sites)
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

    ampo = OpSum()
    for j in 1:(N - 1)
      add!(ampo, "Sz", j, "Sz", j + 1)
      add!(ampo, 0.5, "S+", j, "S-", j + 1)
      add!(ampo, 0.5, "S-", j, "S+", j + 1)
    end
    H = MPO(ampo, sites)

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

  @testset "Bug fixed in threaded block sparse" begin
    maxdim = 10
    nsweeps = 2
    outputlevel = 0
    cutoff = 0.0
    Nx = 4
    Ny = 2
    U = 4.0
    t = 1.0
    N = Nx * Ny
    sweeps = Sweeps(nsweeps)
    maxdims = min.(maxdim, [100, 200, 400, 800, 2000, 3000, maxdim])
    maxdim!(sweeps, maxdims...)
    cutoff!(sweeps, cutoff)
    noise!(sweeps, 1e-6, 1e-7, 1e-8, 0.0)
    sites = siteinds("Electron", N; conserve_qns=true)
    lattice = square_lattice(Nx, Ny; yperiodic=true)
    ampo = OpSum()
    for b in lattice
      ampo .+= -t, "Cdagup", b.s1, "Cup", b.s2
      ampo .+= -t, "Cdagup", b.s2, "Cup", b.s1
      ampo .+= -t, "Cdagdn", b.s1, "Cdn", b.s2
      ampo .+= -t, "Cdagdn", b.s2, "Cdn", b.s1
    end
    for n in 1:N
      ampo .+= U, "Nupdn", n
    end
    H = MPO(ampo, sites)
    Hsplit = splitblocks(linkinds, H)
    state = [isodd(n) ? "↑" : "↓" for n in 1:N]
    ψ0 = productMPS(sites, state)
    using_threaded_blocksparse = ITensors.enable_threaded_blocksparse()
    energy, _ = dmrg(H, ψ0, sweeps; outputlevel=outputlevel)
    energy_split, _ = dmrg(Hsplit, ψ0, sweeps; outputlevel=outputlevel)
    @test energy_split ≈ energy
    if !using_threaded_blocksparse
      ITensors.disable_threaded_blocksparse()
    end
  end
end

nothing
