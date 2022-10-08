using ITensorGPU, ITensors, Test, Random

function heisenberg(n)
  opsum = OpSum()
  for j in 1:(n - 1)
    opsum += 0.5, "S+", j, "S-", j + 1
    opsum += 0.5, "S-", j, "S+", j + 1
    opsum += "Sz", j, "Sz", j + 1
  end
  return opsum
end

@testset "Basic DMRG" begin
  @testset "Spin-one Heisenberg" begin
    N = 10
    sites = siteinds("S=1", N)
    H = cuMPO(MPO(heisenberg(N), sites))

    psi = randomCuMPS(sites)

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

  #=@testset "QN-conserving Spin-one Heisenberg" begin
    N = 10
    sites = siteinds("S=1",N; conserve_qns=true)

    ampo = AutoMPO()
    for j=1:N-1
      add!(ampo,"Sz",j,"Sz",j+1)
      add!(ampo,0.5,"S+",j,"S-",j+1)
      add!(ampo,0.5,"S-",j,"S+",j+1)
    end
    H = cuMPO(MPO(ampo,sites))

    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    psi = randomCuMPS(sites,state,4)

    sweeps = Sweeps(3)
    @test length(sweeps) == 3
    maxdim!(sweeps,10,20,40)
    mindim!(sweeps,1,10)
    cutoff!(sweeps,1E-11)
    noise!(sweeps,1E-10)
    str = split(sprint(show, sweeps), '\n')
    @test length(str) > 1
    energy,psi = dmrg(H, psi, sweeps; outputlevel=0)
    @test energy < -12.0
  end=#

  @testset "Transverse field Ising" begin
    N = 32
    sites = siteinds("S=1/2", N)
    Random.seed!(432)
    psi0 = randomCuMPS(sites)

    ampo = AutoMPO()
    for j in 1:N
      j < N && add!(ampo, -1.0, "Sz", j, "Sz", j + 1)
      add!(ampo, -0.5, "Sx", j)
    end
    H = cuMPO(MPO(ampo, sites))

    sweeps = Sweeps(5)
    maxdim!(sweeps, 10, 20)
    cutoff!(sweeps, 1E-12)
    noise!(sweeps, 1E-10)
    energy, psi = dmrg(H, psi0, sweeps; outputlevel=0)

    # Exact energy for transverse field Ising model
    # with open boundary conditions at criticality
    energy_exact = 0.25 - 0.25 / sin(π / (2 * (2 * N + 1)))
    @test abs((energy - energy_exact) / energy_exact) < 1e-2
  end

  @testset "DMRGObserver" begin
    device = cu
    n = 5
    s = siteinds("S=1/2", n)

    H = device(MPO(heisenberg(n), s))
    ψ0 = device(randomMPS(s))

    dmrg_params = (; nsweeps=4, maxdim=10, cutoff=1e-8, noise=1e-8, outputlevel=0)
    observer = DMRGObserver(["Z"], s; energy_tol=1e-4, minsweeps=10)
    E, ψ = dmrg(H, ψ0; observer=observer, dmrg_params...)
    @test expect(ψ, "Z") ≈ observer.measurements["Z"][end]
    @test correlation_matrix(ψ, "Z", "Z") ≈ correlation_matrix(cpu(ψ), "Z", "Z")
  end

  #=@testset "DMRGObserver" begin
    N = 10
    sites = siteinds("S=1/2",N)
    Random.seed!(42)
    psi0 = randomCuMPS(sites)

    ampo = AutoMPO()
    for j = 1:N-1
      add!(ampo,-1.0,"Sz",j,"Sz",j+1)
    end
    for j = 1:N
      add!(ampo,-0.2,"Sx",j)
    end
    H = cuMPO(MPO(ampo,sites))

    sweeps = Sweeps(3)
    maxdim!(sweeps,10)
    cutoff!(sweeps,1E-12)

    observer = DMRGObserver(["Sz","Sx"], sites)

    E,psi = dmrg(H,psi0,sweeps,observer=observer,outputlevel=0)
    @test length(measurements(observer)["Sz"])==3
    @test length(measurements(observer)["Sx"])==3
    @test all(length.(measurements(observer)["Sz"]) .== N)
    @test all(length.(measurements(observer)["Sx"]) .== N)
    @test length(energies(observer))==3
    @test length(truncerrors(observer))==3
    @test energies(observer)[end]==E
    @test all(truncerrors(observer) .< 1E-9)

    orthogonalize!(psi,1)
    m = scalar(dag(psi[1])*noprime(op(sites, "Sz", 1)*psi[1]))
    @test measurements(observer)["Sz"][end][1] ≈ m
  end

  @testset "Sum of MPOs (ProjMPOSum)" begin
    N = 10
    sites = siteinds("S=1",N)

    ampoZ = AutoMPO()
    for j=1:N-1
      add!(ampoZ,"Sz",j,"Sz",j+1)
    end
    HZ = MPO(ampoZ,sites)

    ampoXY = AutoMPO()
    for j=1:N-1
      add!(ampoXY,0.5,"S+",j,"S-",j+1)
      add!(ampoXY,0.5,"S-",j,"S+",j+1)
    end
    HXY = MPO(ampoXY,sites)

    psi = randomMPS(sites)

    sweeps = Sweeps(3)
    maxdim!(sweeps,10,20,40)
    mindim!(sweeps,1,10,10)
    cutoff!(sweeps,1E-11)
    noise!(sweeps,1E-10)
    energy,psi = dmrg([HZ,HXY], psi, sweeps; outputlevel=0)
    @test energy < -12.0
  end=#

  #=@testset "Excited-state DMRG" begin
    N = 10
    weight = 15.0

    sites = siteinds("S=1",N)
    sites[1] = Index(2,"S=1/2,n=1,Site")
    sites[N] = Index(2,"S=1/2,n=$N,Site")

    ampo = AutoMPO()
    for j=1:N-1
      add!(ampo,"Sz",j,"Sz",j+1)
      add!(ampo,0.5,"S+",j,"S-",j+1)
      add!(ampo,0.5,"S-",j,"S+",j+1)
    end
    H = cuMPO(MPO(ampo,sites))

    psi0i = randomCuMPS(sites,10)

    sweeps = Sweeps(4)
    maxdim!(sweeps, 10,20,100,100)
    cutoff!(sweeps, 1E-11)
    noise!(sweeps,1E-10)

    energy0, psi0 = dmrg(H,psi0i, sweeps; outputlevel=0)
    @test energy0 < -11.5

    psi1i = randomCuMPS(sites,10)
    energy1,psi1 = dmrg(H,[psi0],psi1i,sweeps;outputlevel=0,weight=weight)

    @test energy1 > energy0
    @test energy1 < -11.1

    @test inner(psi1,psi0) < 1E-6
  end=#

  #=@testset "Fermionic Hamiltonian" begin
    N = 10
    t1 = 1.0
    t2 = 0.5
    V = 0.2
    s = siteinds("Fermion", N; conserve_qns = true)

    state = fill(1,N)
    state[1] = 2
    state[3] = 2
    state[5] = 2
    state[7] = 2
    psi0 = productMPS(s,state)

    ampo = AutoMPO()
    for j=1:N-1
      ampo += (-t1, "Cdag", j,   "C", j+1)
      ampo += (-t1, "Cdag", j+1, "C", j)
      ampo += (  V, "N",    j,   "N", j+1)
    end
    for j=1:N-2
      ampo += (-t2, "Cdag", j,   "C", j+2)
      ampo += (-t2, "Cdag", j+2, "C", j)
    end
    H = MPO(ampo, s)

    sweeps = Sweeps(5)
    maxdim!(sweeps, 10, 20, 100, 100, 200)
    cutoff!(sweeps, 1E-8)
    noise!(sweeps, 1E-10)

    energy, psi = dmrg(H, psi0, sweeps; outputlevel = 0)
    @test (-6.5 < energy < -6.4)
  end=#
end

nothing
