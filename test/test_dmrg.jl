using ITensors, Test

@testset "Basic DMRG" begin
  @testset "Spin-one Heisenberg" begin
    N = 100
    sites = spinOneSites(N)

    ampo = AutoMPO(sites)
    for j=1:N-1
      add!(ampo,"Sz",j,"Sz",j+1)
      add!(ampo,0.5,"S+",j,"S-",j+1)
      add!(ampo,0.5,"S-",j,"S+",j+1)
    end
    H = toMPO(ampo)

    psi = randomMPS(sites)

    sweeps = Sweeps(5)
    @test length(sweeps) == 5
    maxdim!(sweeps,10,20,100,100)
    mindim!(sweeps,1,10,20,20)
    cutoff!(sweeps,1E-11)
    str = split(sprint(show, sweeps), '\n')
    @test str[1] == "Sweeps"
    @test str[2] == "1 cutoff=1.0E-11, maxdim=10, mindim=1"
    @test str[3] == "2 cutoff=1.0E-11, maxdim=20, mindim=10"
    @test str[4] == "3 cutoff=1.0E-11, maxdim=100, mindim=20"
    @test str[5] == "4 cutoff=1.0E-11, maxdim=100, mindim=20"
    @test str[6] == "5 cutoff=1.0E-11, maxdim=100, mindim=20"
    energy,psi = dmrg(H,psi,sweeps,maxiter=2,quiet=true)
    @test energy ≈ -138.94 rtol=1e-3
    # test with SVD too! 
    psi = randomMPS(sites)
    energy,psi = dmrg(H,psi,sweeps,maxiter=2,quiet=true,which_factorization="svd")
    @test energy ≈ -138.94 rtol=1e-3
  end

  @testset "Transverse field Ising" begin
    N = 32
    sites = spinHalfSites(N)
    ψ0 = randomMPS(sites)

    ampo = AutoMPO(sites)
    for j = 1:N
      j < N && add!(ampo,-1.0,"Sz",j,"Sz",j+1)
      add!(ampo,-0.5,"Sx",j)
    end
    H = toMPO(ampo)

    sweeps = Sweeps(3)
    maxdim!(sweeps,10,20)
    cutoff!(sweeps,1E-12)
    energy,ψ = dmrg(H,ψ0,sweeps,quiet=true)

    # Exact energy for transverse field Ising model
    # with open boundary conditions at criticality
    energy_exact = 0.25 - 0.25/sin(π/(2*(2*N+1)))
    energy ≈ energy_exact
  end

  @testset "DMRGObserver" begin
    N = 10
    sites = spinHalfSites(N)
    ψ0 = randomMPS(sites)

    ampo = AutoMPO(sites)
    for j = 1:N
      j < N && add!(ampo,-1.0,"Sz",j,"Sz",j+1)
      add!(ampo,-0.2,"Sx",j)
    end
    H = toMPO(ampo)

    sweeps = Sweeps(5)
    maxdim!(sweeps,10)
    cutoff!(sweeps,1E-12)

    observer = DMRGObserver(["Sz","Sx"], sites)

    E,psi = dmrg(H,ψ0,sweeps,observer=observer,quiet=true)
    @test length(measurements(observer)["Sz"])==5
    @test length(measurements(observer)["Sx"])==5
    @test all(length.(measurements(observer)["Sz"]) .== N)
    @test all(length.(measurements(observer)["Sx"]) .== N)
    @test length(energies(observer))==5
    @test energies(observer)[end]==E

    orthogonalize!(psi,1)
    m = scalar(dag(psi[1])*noprime(op(sites, "Sz", 1)*psi[1]))
    @test measurements(observer)["Sz"][end][1] ≈ m
  end
end
