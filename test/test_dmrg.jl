using ITensors, Test

@testset "basic DMRG" begin
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
