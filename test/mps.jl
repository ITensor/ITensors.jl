using ITensors,
      Test

include("util.jl")

@testset "MPS Basics" begin

  N = 10
  sites = [Index(2,"Site") for n=1:N]
  psi = MPS(sites)
  @test length(psi) == N
  @test length(MPS()) == 0

  str = split(sprint(show, psi), '\n')
  @test str[1] == "MPS"
  @test length(str) == length(psi) + 2

  @test siteind(psi,2) == sites[2]
  @test hasind(psi[3],linkind(psi,2))
  @test hasind(psi[3],linkind(psi,3))

  psi[1] = ITensor(sites[1])
  @test hasind(psi[1],sites[1])

  @testset "productMPS" begin
    @testset "vector of string input" begin
      sites = siteinds("S=1/2",N)
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
      sites = siteinds("S=1/2",N)
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
    @testset "vector of ivals input" begin
      sites  = siteinds("S=1/2",N)
      states = fill(0,N)
      for j=1:N
        states[j] = isodd(j) ? 1 : 2
      end
      ivals  = [state(sites[n],states[n]) for n=1:length(sites)]
      psi = productMPS(ivals)
      for j=1:N
        sign = isodd(j) ? +1.0 : -1.0
        @test (psi[j]*op(sites,"Sz",j)*dag(prime(psi[j],"Site")))[] ≈ sign/2
      end
    end
  end

  @testset "randomMPS" begin
    phi = randomMPS(sites)
    @test hasind(phi[1],sites[1])
    @test norm(phi[1])≈1.0
    @test hasind(phi[4],sites[4])
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
 
    badsites = [Index(2) for n=1:N+1]
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
  
  @testset "scaling MPS" begin
    psi = randomMPS(sites)
    twopsidag = 2.0*dag(psi)
    primelinks!(twopsidag)
    @test inner(twopsidag, psi) ≈ 2.0*inner(psi,psi)
  end
  
  @testset "flip sign of MPS" begin
    psi = randomMPS(sites)
    minuspsidag = -dag(psi)
    primelinks!(minuspsidag)
    @test inner(minuspsidag, psi) ≈ -inner(psi,psi)
  end

  @testset "add MPS" begin
    psi = randomMPS(sites)
    phi = similar(psi)
    phi.A_ = deepcopy(psi.A_)
    xi = sum(psi, phi)
    @test inner(xi, xi) ≈ 4.0 * inner(psi, psi) 
    # sum of many MPSs
    Ks = [randomMPS(sites) for i in 1:3]
    K12  = sum(Ks[1], Ks[2])
    K123 = sum(K12, Ks[3])
    @test inner(sum(Ks), K123) ≈ inner(K123,K123)
  end

  sites = siteinds(2,N)
  psi = MPS(sites)
  @test length(psi) == N # just make sure this works
  @test length(siteinds(psi)) == N

  psi = randomMPS(sites)
  orthogonalize!(psi, N-1)
  @test ITensors.leftlim(psi) == N-2
  @test ITensors.rightlim(psi) == N
  orthogonalize!(psi, 2)
  @test ITensors.leftlim(psi) == 1
  @test ITensors.rightlim(psi) == 3
  psi = randomMPS(sites)
  psi.rlim_ = N+1 # do this to test qr from rightmost tensor
  orthogonalize!(psi, div(N, 2))
  @test ITensors.leftlim(psi) == div(N, 2) - 1
  @test ITensors.rightlim(psi) == div(N, 2) + 1

  @test_throws ErrorException linkind(MPS(N, fill(ITensor(), N), 0, N + 1), 1)

  @testset "replacebond!" begin
  # make sure factorization preserves the bond index tags
    psi = randomMPS(sites)
    phi = psi[1]*psi[2]
    bondindtags = tags(linkind(psi,1))
    replacebond!(psi,1,phi)
    @test tags(linkind(psi,1)) == bondindtags

    # check that replacebond! updates llim_ and rlim_ properly
    orthogonalize!(psi,5)
    phi = psi[5]*psi[6]
    replacebond!(psi,5,phi, dir="fromleft")
    @test ITensors.leftlim(psi)==5
    @test ITensors.rightlim(psi)==7

    phi = psi[5]*psi[6]
    replacebond!(psi,5,phi,dir="fromright")
    @test ITensors.leftlim(psi)==4
    @test ITensors.rightlim(psi)==6

    psi.llim_ = 3
    psi.rlim_ = 7
    phi = psi[5]*psi[6]
    replacebond!(psi,5,phi,dir="fromleft")
    @test ITensors.leftlim(psi)==3
    @test ITensors.rightlim(psi)==7
  end

end

# Helper function for making MPS
function basicRandomMPS(N::Int;dim=4)
  sites = [Index(2,"Site") for n=1:N]
  M = MPS(sites)
  links = [Index(dim,"n=$(n-1),Link") for n=1:N+1]
  for n=1:N
    M[n] = randomITensor(links[n],sites[n],links[n+1])
  end
  M[1] *= delta(links[1])
  M[N] *= delta(links[N+1])
  M[1] /= sqrt(inner(M,M))
  return M
end

@testset "MPS gauging and truncation" begin

  N = 30

  @testset "orthogonalize! method" begin
    c = 12
    M = basicRandomMPS(N)
    orthogonalize!(M,c)

    @test leftlim(M) == c-1
    @test rightlim(M) == c+1

    # Test for left-orthogonality
    L = M[1]*prime(M[1],"Link")
    l = linkind(M,1)
    @test norm(L-delta(l,l')) < 1E-12
    for j=2:c-1
      L = L*M[j]*prime(M[j],"Link")
      l = linkind(M,j)
      @test norm(L-delta(l,l')) < 1E-12
    end

    # Test for right-orthogonality
    R = M[N]*prime(M[N],"Link")
    r = linkind(M,N-1)
    @test norm(R-delta(r,r')) < 1E-12
    for j in reverse(c+1:N-1)
      R = R*M[j]*prime(M[j],"Link")
      r = linkind(M,j-1)
      @test norm(R-delta(r,r')) < 1E-12
    end

    @test norm(M[c]) ≈ 1.0
  end

  @testset "truncate! method" begin
    M = basicRandomMPS(N;dim=10)
    M0 = copy(M)
    truncate!(M;maxdim=5)

    @test rightlim(M) == 2

    # Test for right-orthogonality
    R = M[N]*prime(M[N],"Link")
    r = linkind(M,N-1)
    @test norm(R-delta(r,r')) < 1E-12
    for j in reverse(2:N-1)
      R = R*M[j]*prime(M[j],"Link")
      r = linkind(M,j-1)
      @test norm(R-delta(r,r')) < 1E-12
    end

    @test inner(M,M0) > 0.1
  end


end

@testset "Other MPS methods" begin

  @testset "sample! method" begin
    N = 10
    sites = [Index(3,"Site,n=$n") for n=1:N]
    psi = makeRandomMPS(sites,chi=3)
    nrm2 = inner(psi,psi)
    psi[1] *= (1.0/sqrt(nrm2))

    s = sample!(psi)

    @test length(s) == N
    for n=1:N
      @test 1 <= s[n] <= 3
    end

    # Throws becase not orthogonalized to site 1:
    orthogonalize!(psi,3)
    @test_throws ErrorException sample(psi)

    # Throws becase not normalized
    orthogonalize!(psi,1)
    psi[1] *= (5.0/norm(psi[1]))
    @test_throws ErrorException sample(psi)

    # Works when ortho & normalized:
    orthogonalize!(psi,1)
    psi[1] *= (1.0/norm(psi[1]))
    s = sample(psi)
    @test length(s) == N
  end

end
