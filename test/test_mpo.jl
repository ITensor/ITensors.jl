using ITensors,
      Test

@testset "MPO Basics" begin

  N = 4 
  sites = SiteSet(N,2)
  @test length(MPO()) == 0
  O = MPO(sites)
  @test length(O) == N

  str = split(sprint(show, O), '\n')
  @test str[1] == "MPO"
  @test length(str) == length(O) + 2

  O[1] = ITensor(sites[1], prime(sites[1]))
  @test hasindex(O[1],sites[1])
  @test hasindex(O[1],prime(sites[1]))

  P = copy(O)
  @test hasindex(P[1],sites[1])
  @test hasindex(P[1],prime(sites[1]))
  
  @testset "inner" begin
    phi = randomMPS(sites)
    K = randomMPO(sites)
    @test maxDim(K) == 1
    psi = randomMPS(sites)
    phidag = dag(phi)
    prime!(phidag)
    phiKpsi = phidag[1]*K[1]*psi[1]
    for j = 2:N
      phiKpsi *= phidag[j]*K[j]*psi[j]
    end
    @test phiKpsi[] ≈ inner(phi,K,psi)

    badsites = SiteSet(N+1,2)
    badpsi = randomMPS(badsites)
    @test_throws DimensionMismatch inner(phi,K,badpsi)
  end
  
  @testset "applyMPO" begin
    phi = randomMPS(sites)
    K = randomMPO(sites)
    @test maxDim(K) == 1
    psi = randomMPS(sites)
    psi_out = applyMPO(K, psi)
    @test inner(phi,psi_out) ≈ inner(phi,K,psi)
    @test_throws ArgumentError applyMPO(K, psi, method="fakemethod")

    badsites = SiteSet(N+1,2)
    badpsi = randomMPS(badsites)
    @test_throws DimensionMismatch applyMPO(K,badpsi)
  end
  @testset "add" begin
    shsites = spinHalfSites(N)
    K = randomMPO(shsites)
    L = randomMPO(shsites)
    M = sum(K, L)
    @test length(M) == N
    psi = randomMPS(shsites)
    k_psi = applyMPO(K, psi)
    l_psi = applyMPO(L, psi)
    @test inner(psi, sum(k_psi, l_psi)) ≈ inner(psi, M, psi)
  end

  @testset "nmultMPO" begin
    psi = randomMPS(sites)
    K = randomMPO(sites)
    L = randomMPO(sites)
    @test maxDim(K) == 1
    @test maxDim(L) == 1
    KL = nmultMPO(K, L)
    psi_kl_out = applyMPO(K, applyMPO(L, psi))
    @test inner(psi,KL,psi) ≈ inner(psi, psi_kl_out)

    badsites = SiteSet(N+1,2)
    badL = randomMPO(badsites)
    @test_throws DimensionMismatch nmultMPO(K,badL)
  end
  
  sites = spinHalfSites(N)
  O = MPO(sites,"Sz")
  @test length(O) == N # just make sure this works
 
  @test_throws ArgumentError randomMPO(sites, 2)
end
