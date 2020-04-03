using ITensors, Test, Random

@testset "Basic DMRG" begin
  @testset "Spin-one Heisenberg" begin
    N = 100
    sites = siteinds("S=1",N)

    ampo = AutoMPO()
    for j=1:N-1
      add!(ampo,"Sz",j,"Sz",j+1)
      add!(ampo,0.5,"S+",j,"S-",j+1)
      add!(ampo,0.5,"S-",j,"S+",j+1)
    end
    H = toMPO(ampo,sites)

    psi = randomMPS(sites)

    sweeps = Sweeps(3)
    @test length(sweeps) == 3
    maxdim!(sweeps,10,20,40)
    mindim!(sweeps,1,10,10)
    cutoff!(sweeps,1E-11)
    str = split(sprint(show, sweeps), '\n')
    @test length(str) > 1
    energy,psi = dmrg(H, psi, sweeps; quiet=true)
    @test energy < -120.0
    # test with SVD too! 
    psi = randomMPS(sites)
    energy,psi = dmrg(H, psi, sweeps; 
                      quiet=true)
    @test energy < -120.0
  end

  @testset "Transverse field Ising" begin
    N = 32
    sites = siteinds("S=1/2",N)
    Random.seed!(432)
    psi0 = randomMPS(sites)

    ampo = AutoMPO()
    for j = 1:N
      j < N && add!(ampo,-1.0,"Sz",j,"Sz",j+1)
      add!(ampo,-0.5,"Sx",j)
    end
    H = toMPO(ampo,sites)

    sweeps = Sweeps(5)
    maxdim!(sweeps,10,20)
    cutoff!(sweeps,1E-12)
    energy,psi = dmrg(H,psi0,sweeps,quiet=true)

    # Exact energy for transverse field Ising model
    # with open boundary conditions at criticality
    energy_exact = 0.25 - 0.25/sin(π/(2*(2*N+1)))
    @test abs((energy-energy_exact)/energy_exact) < 1e-6
  end

  @testset "DMRGObserver" begin
    N = 10
    sites = siteinds("S=1/2",N)
    Random.seed!(42)
    psi0 = randomMPS(sites)

    ampo = AutoMPO()
    for j = 1:N
      j < N && add!(ampo,-1.0,"Sz",j,"Sz",j+1)
      add!(ampo,-0.2,"Sx",j)
    end
    H = toMPO(ampo,sites)

    sweeps = Sweeps(3)
    maxdim!(sweeps,10)
    cutoff!(sweeps,1E-12)

    observer = DMRGObserver(["Sz","Sx"], sites)

    E,psi = dmrg(H,psi0,sweeps,observer=observer,quiet=true)
    @test length(measurements(observer)["Sz"])==3
    @test length(measurements(observer)["Sx"])==3
    @test all(length.(measurements(observer)["Sz"]) .== N)
    @test all(length.(measurements(observer)["Sx"]) .== N)
    @test length(energies(observer))==3
    @test length(truncerrors(observer))==3
    @test energies(observer)[end]==E
    @test all(truncerrors(observer) .< 1E-11)

    orthogonalize!(psi,1)
    m = scalar(dag(psi[1])*noprime(op(sites, "Sz", 1)*psi[1]))
    @test measurements(observer)["Sz"][end][1] ≈ m
  end
end
