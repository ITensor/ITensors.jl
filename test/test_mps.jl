using ITensors,
      Test

@testset "MPS Basics" begin

  N = 10
  sites = SiteSet(N,2)
  psi = MPS(sites)
  @test length(psi) == N
  @test length(MPS()) == 0

  str = split(sprint(show, psi), '\n')
  @test str[1] == "MPS"
  @test length(str) == length(psi) + 2

  @test siteindex(psi,2) == sites[2]
  @test hasindex(psi[3],linkindex(psi,2))
  @test hasindex(psi[3],linkindex(psi,3))

  psi[1] = ITensor(sites[1])
  @test hasindex(psi[1],sites[1])

  @testset "RandomMPS" begin
    phi = randomMPS(sites)
    @test hasindex(phi[1],sites[1])
    @test norm(phi[1])≈1.0
    @test hasindex(phi[4],sites[4])
    @test norm(phi[4])≈1.0
  end

  @testset "inner different MPS" begin
    phi = randomMPS(sites)
    psi = randomMPS(sites)
    phipsi = dag(phi[1])*psi[1]
    for j = 2:N
      phipsi *= dag(phi[j])*psi[j]
    end
    @test phipsi[] ≈ inner(phi,psi)
 
    badsites = SiteSet(N+1,2)
    badpsi = randomMPS(badsites)
    @test_throws DimensionMismatch inner(phi,badpsi)
  end

  @testset "inner same MPS" begin
    psi = randomMPS(sites)
    psidag = dag(psi)
    primelinks!(psidag)
    psipsi = psidag[1]*psi[1]
    for j = 2:N
      psipsi *= psidag[j]*psi[j]
    end
    @test psipsi[] ≈ inner(psi,psi)
  end

  @testset "Parametric constructor" begin
    ψ = MPS{Dense{Float64}}(sites)
    param(::MPS{T}) where {T} = T
    @test typeof(store(ψ)) == param(ψ)
  end

    
  sites = spinHalfSites(N)
  psi = MPS(sites)
  @test length(psi) == N # just make sure this works
  @test length(siteinds(psi)) == N

  psi = randomMPS(sites)
  position!(psi, N-1)
  @test ITensors.leftLim(psi) == N-2
  @test ITensors.rightLim(psi) == N
  position!(psi, 2)
  @test ITensors.leftLim(psi) == 1
  @test ITensors.rightLim(psi) == 3
end
