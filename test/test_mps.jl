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

  @testset "productMPS" begin
    @testset "vector of string input" begin
      sites = spinHalfSites(N)
      state = fill("",N)
      for j=1:N
        state[j] = isodd(j) ? "Up" : "Dn"
      end
      psi = productMPS(sites,state)
      for j=1:N
        sign = isodd(j) ? +1.0 : -1.0
        @test (psi[j]*op(sites,"Sz",j)*dag(prime(psi[j],"Site")))[] ≈ sign/2
      end
      @test_throws DimensionMismatch productMPS(sites, fill("", N - 1))
    end

    @testset "vector of int input" begin
      sites = spinHalfSites(N)
      state = fill(0,N)
      for j=1:N
        state[j] = isodd(j) ? 1 : 2
      end
      psi = productMPS(sites,state)
      for j=1:N
        sign = isodd(j) ? +1.0 : -1.0
        @test (psi[j]*op(sites,"Sz",j)*dag(prime(psi[j],"Site")))[] ≈ sign/2
      end
    end

  end

  @testset "randomMPS" begin
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

  @testset "add MPS" begin
    psi = randomMPS(sites)
    phi = similar(psi)
    phi.A_ = deepcopy(psi.A_)
    xi = sum(psi, phi)
    @test inner(xi, xi) ≈ 4.0 * inner(psi, psi) 
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
  psi = randomMPS(sites)
  psi.rlim_ = N+1 # do this to test qr from rightmost tensor
  position!(psi, div(N, 2))
  @test ITensors.leftLim(psi) == div(N, 2) - 1
  @test ITensors.rightLim(psi) == div(N, 2) + 1

  @test_throws ErrorException linkindex(MPS(N, fill(ITensor(), N), 0, N + 1), 1)

  # make sure factorization preserves the bond index tags
  phi = psi[1]*psi[2]
  bondindtags = tags(linkindex(psi,1))
  replaceBond!(psi,1,phi)
  @test tags(linkindex(psi,1)) == bondindtags
end
