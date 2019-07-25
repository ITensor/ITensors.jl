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
  energy,psi = dmrg(H,psi,sweeps,maxiter=2,quiet=true)
  @test energy ≈ -138.94 rtol=1e-3
end
