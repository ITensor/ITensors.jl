using Combinatorics
using ITensors
using Test

include("util.jl")

@testset "MPS Basics" begin

  N = 10
  sites = [Index(2,"Site") for n=1:N]
  psi = MPS(sites)
  @test length(psi) == N
  @test length(MPS()) == 0
  @test isnothing(flux(psi))

  str = split(sprint(show, psi), '\n')
  @test str[1] == "MPS"
  @test length(str) == length(psi) + 2

  @test siteind(psi,2) == sites[2]
  @test findfirstsiteind(psi, sites[2]) == 2
  @test findfirstsiteind(psi, sites[4]) == 4
  @test findfirstsiteinds(psi, IndexSet(sites[5])) == 5
  @test hasind(psi[3],linkind(psi,2))
  @test hasind(psi[3],linkind(psi,3))

  @test isnothing(linkind(psi, N))
  @test isnothing(linkind(psi, N+1))
  @test isnothing(linkind(psi, 0))
  @test isnothing(linkind(psi, -1))
  @test linkind(psi, 3) == commonind(psi[3], psi[4])

  psi[1] = ITensor(sites[1])
  @test hasind(psi[1],sites[1])

  @testset "N=1 MPS" begin
    sites1 = [Index(2,"Site,n=1")]
    psi = MPS(sites1)
    @test length(psi) == 1
    @test siteind(psi,1) == sites1[1]
    @test siteinds(psi)[1] == sites1[1]
  end

  @testset "Missing links" begin
    psi = MPS([randomITensor(sites[i]) for i in 1:N])
    @test isnothing(linkind(psi, 1))
    @test isnothing(linkind(psi, 5))
    @test isnothing(linkind(psi, N))
    @test maxlinkdim(psi) == 0
    @test psi ⋅ psi ≈ *(dag(psi)..., psi...)[]
  end

  @testset "productMPS" begin

    @testset "vector of string input" begin
      sites = siteinds("S=1/2", N)
      state = fill("", N)
      for j=1:N
        state[j] = isodd(j) ? "Up" : "Dn"
      end
      psi = productMPS(sites,state)
      for j=1:N
        sign = isodd(j) ? +1.0 : -1.0
        @test (psi[j] * op(sites,"Sz",j) * dag(prime(psi[j],"Site")))[] ≈ sign/2
      end
      @test_throws DimensionMismatch productMPS(sites, fill("", N - 1))
    end

    @testset "String input" begin
      sites = siteinds("S=1/2", N)
      psi = productMPS(sites, "Dn")
      for j=1:N
        sign = -1.0
        @test (psi[j] * op(sites,"Sz",j) * dag(prime(psi[j],"Site")))[] ≈ sign/2
      end
    end

    @testset "Int input" begin
      sites = siteinds("S=1/2", N)
      psi = productMPS(sites, 2)
      for j=1:N
        sign = -1.0
        @test (psi[j] * op(sites,"Sz",j) * dag(prime(psi[j],"Site")))[] ≈ sign/2
      end
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

      @testset "ComplexF64 eltype" begin
        sites  = siteinds("S=1/2",N)
        psi = productMPS(ComplexF64,sites,fill(1,N))
        for j=1:N
          @test eltype(psi[j]) <: ComplexF64
        end
      end
    end

    @testset "N=1 case" begin
      site = Index(2,"Site,n=1")
      psi = productMPS([site],[1])
      @test psi[1][1] ≈ 1.0
      @test psi[1][2] ≈ 0.0
      psi = productMPS([site],[2])
      @test psi[1][1] ≈ 0.0
      @test psi[1][2] ≈ 1.0
    end
  end

  @testset "randomMPS with chi==1" begin
    phi = randomMPS(sites)

    @test maxlinkdim(phi) == 1

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
    #ITensors.prime_linkinds!(psidag)
    psipsi = psidag[1]*psi[1]
    for j = 2:N
      psipsi *= psidag[j]*psi[j]
    end
    @test psipsi[] ≈ inner(psi,psi)
  end
  
  @testset "norm MPS" begin
    psi = randomMPS(sites,10)
    psidag = ITensors.sim_linkinds(dag(psi))
    psi² = ITensor(1)
    for j = 1:N
      psi² *= psidag[j] * psi[j]
    end
    @test psi²[] ≈ psi ⋅ psi
    @test sqrt(psi²[]) ≈ norm(psi)
    for j in 1:N
      psi[j] .*= j
    end
    @test norm(psi) ≈ factorial(N)
  end

  @testset "lognorm MPS" begin
    psi = randomMPS(sites,10)
    for j in 1:N
      psi[j] .*= j
    end
    psidag = ITensors.sim_linkinds(dag(psi))
    psi² = ITensor(1)
    for j = 1:N
      psi² *= psidag[j] * psi[j]
    end
    @test psi²[] ≈ psi ⋅ psi
    @test 0.5 * log(psi²[]) ≈ lognorm(psi)
    @test lognorm(psi) ≈ log(factorial(N))
  end

  @testset "scaling MPS" begin
    psi = randomMPS(sites)
    twopsidag = 2.0*dag(psi)
    #ITensors.prime_linkinds!(twopsidag)
    @test inner(twopsidag, psi) ≈ 2.0*inner(psi,psi)
  end
  
  @testset "flip sign of MPS" begin
    psi = randomMPS(sites)
    minuspsidag = -dag(psi)
    #ITensors.primelinkinds!(minuspsidag)
    @test inner(minuspsidag, psi) ≈ -inner(psi,psi)
  end

  @testset "add MPS" begin
    psi = randomMPS(sites)
    phi = deepcopy(psi)
    xi = add(psi, phi)
    @test inner(xi, xi) ≈ 4.0 * inner(psi, psi) 
    # sum of many MPSs
    Ks = [randomMPS(sites) for i in 1:3]
    K12  = add(Ks[1], Ks[2])
    K123 = add(K12, Ks[3])
    @test inner(sum(Ks), K123) ≈ inner(K123,K123)
  end

  @testset "+ MPS" begin
    psi = randomMPS(sites)
    phi = deepcopy(psi)
    xi = psi + phi
    @test inner(xi, xi) ≈ 4.0 * inner(psi, psi) 
    # sum of many MPSs
    Ks = [randomMPS(sites) for i in 1:3]
    K12  = Ks[1] + Ks[2]
    K123 = K12 + Ks[3]
    @test inner(sum(Ks), K123) ≈ inner(K123,K123)
  end

  sites = siteinds(2,N)
  psi = MPS(sites)
  @test length(psi) == N # just make sure this works
  @test length(siteinds(psi)) == N

  psi = randomMPS(sites)
  l0s = linkinds(psi)
  orthogonalize!(psi, N-1)
  ls = linkinds(psi)
  for (l0,l) in zip(l0s,ls)
    @test tags(l0) == tags(l)
  end
  @test ITensors.leftlim(psi) == N-2
  @test ITensors.rightlim(psi) == N
  orthogonalize!(psi, 2)
  @test ITensors.leftlim(psi) == 1
  @test ITensors.rightlim(psi) == 3
  psi = randomMPS(sites)
  ITensors.setrightlim!(psi, N+1) # do this to test qr 
                                  # from rightmost tensor
  orthogonalize!(psi, div(N, 2))
  @test ITensors.leftlim(psi) == div(N, 2) - 1
  @test ITensors.rightlim(psi) == div(N, 2) + 1

  @test isnothing(linkind(MPS(fill(ITensor(), N), 0, N + 1), 1))

  @testset "replacebond!" begin
  # make sure factorization preserves the bond index tags
    psi = randomMPS(sites)
    phi = psi[1]*psi[2]
    bondindtags = tags(linkind(psi,1))
    replacebond!(psi,1,phi)
    @test tags(linkind(psi,1)) == bondindtags

    # check that replacebond! updates llim and rlim properly
    orthogonalize!(psi,5)
    phi = psi[5]*psi[6]
    replacebond!(psi, 5, phi; ortho = "left")
    @test ITensors.leftlim(psi) == 5
    @test ITensors.rightlim(psi) == 7

    phi = psi[5]*psi[6]
    replacebond!(psi, 5, phi; ortho = "right")
    @test ITensors.leftlim(psi) == 4
    @test ITensors.rightlim(psi)==6

    ITensors.setleftlim!(psi, 3)
    ITensors.setrightlim!(psi, 7)
    phi = psi[5]*psi[6]
    replacebond!(psi, 5, phi; ortho = "left")
    @test ITensors.leftlim(psi) == 3
    @test ITensors.rightlim(psi) == 7
  end

end

@testset "orthogonalize! with QNs" begin
  N = 8
  sites = siteinds("S=1/2",N, conserve_qns=true)
  init_state = [isodd(n) ? "Up" : "Dn" for n=1:N]
  psi0 = productMPS(sites,init_state)
  orthogonalize!(psi0,4)
  @test ITensors.leftlim(psi0) == 3
  @test ITensors.rightlim(psi0) == 5
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

    @test ITensors.leftlim(M) == c-1
    @test ITensors.rightlim(M) == c+1

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

    @test ITensors.rightlim(M) == 2

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

  @testset "randomMPS with chi > 1" begin
    N = 20
    chi = 8
    sites = siteinds(2,N)
    M = randomMPS(sites,chi)

    @test ITensors.leftlim(M) == 0
    @test ITensors.rightlim(M) == 2

    @test norm(M[1]) ≈ 1.0

    @test maxlinkdim(M) == chi

    # Test for right-orthogonality
    R = M[N]*prime(M[N],"Link")
    r = linkind(M,N-1)
    @test norm(R-delta(r,r')) < 1E-10
    for j in reverse(2:N-1)
      R = R*M[j]*prime(M[j],"Link")
      r = linkind(M,j-1)
      @test norm(R-delta(r,r')) < 1E-10
    end
  end

  @testset "randomMPS from initial state (QN case)" begin
    N = 20
    chi = 8
    sites = siteinds("S=1/2",N;conserve_qns=true)

    # Make flux-zero random MPS
    state = [isodd(n) ? 1 : 2 for n=1:N]
    M = randomMPS(sites,state,chi)
    @test flux(M) == QN("Sz",0)

    @test ITensors.leftlim(M) == 0
    @test ITensors.rightlim(M) == 2

    @test norm(M[1]) ≈ 1.0
    @test inner(M,M) ≈ 1.0

    @test maxlinkdim(M) == chi

    # Test making random MPS with different flux
    state[1] = 2
    M = randomMPS(sites,state,chi)
    @test flux(M) == QN("Sz",-2)
    state[3] = 2
    M = randomMPS(sites,state,chi)
    @test flux(M) == QN("Sz",-4)
  end

  @testset "swapbondsites" begin
    N = 5
    sites = siteinds("S=1/2", N)
    ψ0 = randomMPS(sites)
    ψ = replacebond(ψ0, 3, ψ0[3] * ψ0[4];
                    swapsites = true,
                    cutoff = 1e-15)
    @test siteind(ψ, 1) == siteind(ψ0, 1)
    @test siteind(ψ, 2) == siteind(ψ0, 2)
    @test siteind(ψ, 4) == siteind(ψ0, 3)
    @test siteind(ψ, 3) == siteind(ψ0, 4)
    @test siteind(ψ, 5) == siteind(ψ0, 5)
    @test prod(ψ) ≈ prod(ψ0)
    @test maxlinkdim(ψ) == 1

    ψ = swapbondsites(ψ0, 4;
                      cutoff = 1e-15)
    @test siteind(ψ, 1) == siteind(ψ0, 1)
    @test siteind(ψ, 2) == siteind(ψ0, 2)
    @test siteind(ψ, 3) == siteind(ψ0, 3)
    @test siteind(ψ, 5) == siteind(ψ0, 4)
    @test siteind(ψ, 4) == siteind(ψ0, 5)
    @test prod(ψ) ≈ prod(ψ0)
    @test maxlinkdim(ψ) == 1
  end

  @testset "map!" begin
    N = 5
    s = siteinds("S=½", N)
    M0 = productMPS(s, "↑")

    # Test map! with limits getting set
    M = orthogonalize(M0, 1)
    @test ITensors.leftlim(M) == 0
    @test ITensors.rightlim(M) == 2
    map!(prime, M)
    @test ITensors.leftlim(M) == 0
    @test ITensors.rightlim(M) == N+1

    # Test map! without limits getting set
    M = orthogonalize(M0, 1)
    map!(prime, M, set_limits = false)
    @test ITensors.leftlim(M) == 0
    @test ITensors.rightlim(M) == 2

    # Test prime! with limits getting set
    M = orthogonalize(M0, 1)
    @test ITensors.leftlim(M) == 0
    @test ITensors.rightlim(M) == 2
    prime!(M, set_limits = true)
    @test ITensors.leftlim(M) == 0
    @test ITensors.rightlim(M) == N+1

    # Test prime! without limits getting set
    M = orthogonalize(M0, 1)
    prime!(M)
    @test ITensors.leftlim(M) == 0
    @test ITensors.rightlim(M) == 2
  end

  @testset "setindex!(::MPS, _, ::Colon)" begin
    N = 4
    s = siteinds("S=½", N)
    ψ = randomMPS(s)
    ϕ = productMPS(s, "↑")
    orthogonalize!(ϕ, 1)
    ψ[:] = ϕ
    @test ITensors.orthocenter(ψ) == 1
    @test inner(ψ, ϕ) ≈ 1

    ψ = randomMPS(s)
    ϕ = productMPS(s, "↑")
    orthogonalize!(ϕ, 1)
    ψ[:] = ITensors.data(ϕ)
    @test ITensors.leftlim(ψ) == 0
    @test ITensors.rightlim(ψ) == N+1
    @test inner(ψ, ϕ) ≈ 1
  end

  @testset "findsite[s](::MPS/MPO, is)" begin
    s = siteinds("S=1/2", 5)
    ψ = randomMPS(s)
    l = linkinds(ψ)

    A = randomITensor(s[4]', s[2]', dag(s[4]), dag(s[2]))

    @test findsite(ψ, s[3]) == 3
    @test findsite(ψ, (s[3], s[5])) == 3
    @test findsite(ψ, l[2]) == 2
    @test findsite(ψ, A) == 2

    @test findsites(ψ, s[3]) == [3]
    @test findsites(ψ, (s[4], s[1])) == [1, 4]
    @test findsites(ψ, l[2]) == [2, 3]
    @test findsites(ψ, (l[2], l[3])) == [2, 3, 4]
    @test findsites(ψ, A) == [2, 4]

    M = randomMPO(s)
    lM = linkinds(M)

    @test findsite(M, s[4]) == 4
    @test findsite(M, s[4]') == 4
    @test findsite(M, (s[4]', s[4])) == 4
    @test findsite(M, (s[4]', s[3])) == 3
    @test findsite(M, lM[2]) == 2
    @test findsite(M, A) == 2

    @test findsites(M, s[4]) == [4]
    @test findsites(M, s[4]') == [4]
    @test findsites(M, (s[4]', s[4])) == [4]
    @test findsites(M, (s[4]', s[3])) == [3, 4]
    @test findsites(M, (lM[2], lM[3])) == [2, 3, 4]
    @test findsites(M, A) == [2, 4]
  end

  @testset "[first]siteind[s](::MPS/MPO, j::Int)" begin
    s = siteinds("S=1/2", 5)
    ψ = randomMPS(s)
    @test firstsiteind(ψ, 3) == s[3]
    @test siteind(ψ, 4) == s[4]
    @test isnothing(siteind(ψ, 4; plev = 1))
    @test siteinds(ψ, 3) == IndexSet(s[3])
    @test siteinds(ψ, 3; plev = 1) == IndexSet()

    M = randomMPO(s)
    @test noprime(firstsiteind(M, 4)) == s[4]
    @test firstsiteind(M, 4; plev = 0) == s[4]
    @test firstsiteind(M, 4; plev = 1) == s[4]'
    @test siteind(M, 4) == s[4]
    @test siteind(M, 4; plev = 0) == s[4]
    @test siteind(M, 4; plev = 1) == s[4]'
    @test isnothing(siteind(M, 4; plev = 2))
    @test siteinds(M, 3) == IndexSet(s[3], s[3]')
    @test siteinds(M, 3; plev = 1) == IndexSet(s[3]')
    @test siteinds(M, 3; plev = 0) == IndexSet(s[3])
    @test siteinds(M, 3; tags = "n=2") == IndexSet()
  end

  @testset "movesites $N sites" for N in 1:7
    s0 = siteinds("S=1/2", N)
    ψ0 = productMPS(s0, "↑")
    for perm in permutations(1:N)
      s = s0[perm]
      ψ = productMPS(s, rand(("↑", "↓"), N))
      ns′ = [findsite(ψ0, i) for i in s]
      @test ns′ == perm
      ψ′ = movesites(ψ, 1:N .=> ns′)
      for n in 1:N
        @test siteind(ψ0, n) == siteind(ψ′, n)
      end
      @test prod(ψ) ≈ prod(ψ′)
    end
  end

  @testset "Construct MPS from ITensor" begin
    N = 5
    s = siteinds("S=1/2", N)
    l = [Index(3, "left_$n") for n in 1:2]
    r = [Index(3, "right_$n") for n in 1:2]

    #
    # MPS
    #

    A = randomITensor(s...)
    ψ = MPS(A, s)
    @test prod(ψ) ≈ A
    @test ITensors.orthocenter(ψ) == N
    @test maxlinkdim(ψ) == 4

    ψ0 = productMPS(s, "↑")
    A = prod(ψ0)
    ψ = MPS(A, s; cutoff = 1e-15)
    @test prod(ψ) ≈ A
    @test ITensors.orthocenter(ψ) == N
    @test maxlinkdim(ψ) == 1

    ψ0 = randomMPS(s, 2)
    A = prod(ψ0)
    ψ = MPS(A, s; cutoff = 1e-15, orthocenter = 2)
    @test prod(ψ) ≈ A
    @test ITensors.orthocenter(ψ) == 2
    @test maxlinkdim(ψ) == 2

    A = randomITensor(s..., l[1], r[1])
    ψ = MPS(A, s, leftinds = l[1], orthocenter = 3)
    ls = linkinds(ψ)
    @test hassameinds(ψ[1], (l[1], s[1], ls[1]))
    @test hassameinds(ψ[N], (r[1], s[N], ls[N - 1]))
    @test prod(ψ) ≈ A
    @test ITensors.orthocenter(ψ) == 3
    @test maxlinkdim(ψ) == 12

    A = randomITensor(s..., l..., r...)
    ψ = MPS(A, s, leftinds = l)
    ls = linkinds(ψ)
    @test hassameinds(ψ[1], (l..., s[1], ls[1]))
    @test hassameinds(ψ[N], (r..., s[N], ls[N - 1]))
    @test prod(ψ) ≈ A
    @test ITensors.orthocenter(ψ) == N
    @test maxlinkdim(ψ) == 36
  end

  @testset "Set range of MPS tensors" begin
    N = 5
    s = siteinds("S=1/2", N)
    ψ0 = randomMPS(s, 3)

    ψ = orthogonalize(ψ0, 2)
    A = prod(ITensors.data(ψ)[2:N-1])
    randn!(A)
    ϕ = MPS(A, s[2:N-1], orthocenter = 1)
    ψ[2:N-1] = ϕ
    @test prod(ψ) ≈ ψ[1] * A * ψ[N]
    @test maxlinkdim(ψ) == 4
    @test ITensors.orthocenter(ψ) == 2

    ψ = orthogonalize(ψ0, 1)
    A = prod(ITensors.data(ψ)[2:N-1])
    randn!(A)
    @test_throws AssertionError ψ[2:N-1] = A

    ψ = orthogonalize(ψ0, 2)
    A = prod(ITensors.data(ψ)[2:N-1])
    randn!(A)
    ψ[2:N-1, orthocenter = 3] = A
    @test prod(ψ) ≈ ψ[1] * A * ψ[N]
    @test maxlinkdim(ψ) == 4
    @test ITensors.orthocenter(ψ) == 3
  end

  @testset "movesites reverse sites" begin
    N = 6
    s = siteinds("S=1/2", N)
    ψ0 = randomMPS(s)
    ψ = movesites(ψ0, 1:N .=> reverse(1:N))
    for n in 1:N
      @test siteind(ψ, n) == s[N-n+1]
    end
  end

end

nothing
