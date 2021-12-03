using ITensors, ITensorGPU, Test
@testset "cuMPS Basics" begin
  N = 10
  sites = [Index(2, "Site") for n in 1:N]
  psi = cuMPS(sites)
  @test length(psi) == N
  @test length(cuMPS()) == 0

  str = split(sprint(show, psi), '\n')
  @test str[1] == "MPS"
  @test length(str) == length(psi) + 2

  @test siteind(psi, 2) == sites[2]
  @test hasind(psi[3], linkind(psi, 2))
  @test hasind(psi[3], linkind(psi, 3))

  psi[1] = cuITensor(sites[1])
  @test hasind(psi[1], sites[1])

  L = randomMPS(sites)
  K = cuMPS(L)
  @test all(ITensors.data(cpu(K)) .== ITensors.data(cpu(L)))

  @testset "cuproductMPS" begin
    @testset "vector of string input" begin
      sites = siteinds("S=1/2", N)
      state = fill("", N)
      for j in 1:N
        state[j] = isodd(j) ? "Up" : "Dn"
      end
      psi = productCuMPS(sites, state)
      for j in 1:N
        sign = isodd(j) ? +1.0 : -1.0
        ops = cuITensor(op(sites, "Sz", j))
        psip = prime(psi[j], "Site")
        res = psi[j] * ops * dag(psip)
        @test res[] ≈ sign / 2
      end
      @test_throws DimensionMismatch cuMPS(sites, fill("", N - 1))
      @test_throws DimensionMismatch productCuMPS(sites, fill("", N - 1))
    end

    @testset "vector of int input" begin
      sites = siteinds("S=1/2", N)
      state = fill(0, N)
      for j in 1:N
        state[j] = isodd(j) ? 1 : 2
      end
      psi = productCuMPS(sites, state)
      for j in 1:N
        sign = isodd(j) ? +1.0 : -1.0
        ops = cuITensor(op(sites, "Sz", j))
        psip = prime(psi[j], "Site")
        @test (psi[j] * ops * dag(psip))[] ≈ sign / 2
      end
    end
  end

  @testset "randomMPS" begin
    phi = randomCuMPS(sites)
    @test hasind(phi[1], sites[1])
    @test norm(phi[1]) ≈ 1.0
    @test hasind(phi[4], sites[4])
    @test norm(phi[4]) ≈ 1.0
  end

  @testset "inner different MPS" begin
    phi = randomMPS(sites)
    psi = randomMPS(sites)
    phipsi = dag(phi[1]) * psi[1]
    for j in 2:N
      phipsi *= dag(phi[j]) * psi[j]
    end
    @test phipsi[] ≈ inner(phi, psi)
    phi = randomCuMPS(sites)
    psi = randomCuMPS(sites)
    cphi = MPS([cpu(phi[i]) for i in 1:length(phi)])
    cpsi = MPS([cpu(psi[i]) for i in 1:length(psi)])
    phipsi = dag(phi[1]) * psi[1]
    cphipsi = dag(cphi[1]) * cpsi[1]
    for j in 2:N
      phipsi *= dag(phi[j]) * psi[j]
      cphipsi *= dag(cphi[j]) * cpsi[j]
    end
    @test cpu(phipsi)[] ≈ cphipsi[]
    @test cpu(phipsi)[] ≈ inner(cphi, cpsi)
    @test cpu(phipsi)[] ≈ inner(phi, psi)
    phipsi = dag(phi[1]) * psi[1]
    for j in 2:N
      phipsi = phipsi * dag(phi[j]) * psi[j]
    end
    @test cpu(phipsi)[] ≈ inner(phi, psi)

    badsites = [Index(2) for n in 1:(N + 1)]
    badpsi = randomCuMPS(badsites)
    @test_throws DimensionMismatch inner(phi, badpsi)
  end

  @testset "inner same MPS" begin
    psi = randomMPS(sites)
    psidag = dag(deepcopy(psi))
    ITensors.prime_linkinds!(psidag)
    psipsi = psidag[1] * psi[1]
    for j in 2:N
      psipsi = psipsi * psidag[j] * psi[j]
    end
    @test psipsi[] ≈ inner(psi, psi)
    psi = randomCuMPS(sites)
    psidag = dag(deepcopy(psi))
    ITensors.prime_linkinds!(psidag)
    psipsi = psidag[1] * psi[1]
    for j in 2:N
      psipsi = psipsi * psidag[j] * psi[j]
    end
    @test psipsi[] ≈ inner(psi, psi)
  end

  @testset "add MPS" begin
    psi = randomMPS(sites)
    phi = deepcopy(psi)
    xi = add(psi, phi)
    @test inner(xi, xi) ≈ 4.0 * inner(psi, psi)
    psi = randomCuMPS(sites)
    phi = deepcopy(psi)
    xi = add(psi, phi)
    @test inner(xi, xi) ≈ 4.0 * inner(psi, psi)
  end

  sites = siteinds("S=1/2", N)
  psi = cuMPS(sites)
  @test length(psi) == N # just make sure this works
  @test length(siteinds(psi)) == N

  psi = randomCuMPS(sites)
  orthogonalize!(psi, N - 1)
  @test ITensors.leftlim(psi) == N - 2
  @test ITensors.rightlim(psi) == N
  orthogonalize!(psi, 2)
  @test ITensors.leftlim(psi) == 1
  @test ITensors.rightlim(psi) == 3
  psi = randomCuMPS(sites)
  psi.rlim = N + 1 # do this to test qr from rightmost tensor
  orthogonalize!(psi, div(N, 2))
  @test ITensors.leftlim(psi) == div(N, 2) - 1
  @test ITensors.rightlim(psi) == div(N, 2) + 1

  #@test_throws ErrorException linkind(MPS(N, fill(cuITensor(), N), 0, N + 1), 1)

  @testset "replacebond!" begin
    # make sure factorization preserves the bond index tags
    psi = randomCuMPS(sites)
    phi = psi[1] * psi[2]
    bondindtags = tags(linkind(psi, 1))
    replacebond!(psi, 1, phi)
    @test tags(linkind(psi, 1)) == bondindtags

    # check that replaceBond! updates llim_ and rlim_ properly
    orthogonalize!(psi, 5)
    phi = psi[5] * psi[6]
    replacebond!(psi, 5, phi; ortho="left")
    @test ITensors.leftlim(psi) == 5
    @test ITensors.rightlim(psi) == 7

    phi = psi[5] * psi[6]
    replacebond!(psi, 5, phi; ortho="right")
    @test ITensors.leftlim(psi) == 4
    @test ITensors.rightlim(psi) == 6

    psi.llim = 3
    psi.rlim = 7
    phi = psi[5] * psi[6]
    replacebond!(psi, 5, phi; ortho="left")
    @test ITensors.leftlim(psi) == 3
    @test ITensors.rightlim(psi) == 7
  end
end

# Helper function for making MPS
function basicRandomCuMPS(N::Int; dim=4)
  sites = [Index(2, "Site") for n in 1:N]
  M = MPS(sites)
  links = [Index(dim, "n=$(n-1),Link") for n in 1:(N + 1)]
  M[1] = randomCuITensor(sites[1], links[2])
  for n in 2:(N - 1)
    M[n] = randomCuITensor(links[n], sites[n], links[n + 1])
  end
  M[N] = randomCuITensor(links[N], sites[N])
  M[1] /= sqrt(inner(M, M))
  return M
end
@testset "MPS gauging and truncation" begin
  N = 30

  @testset "orthogonalize! method" begin
    c = 12
    M = basicRandomCuMPS(N)
    orthogonalize!(M, c)

    @test ITensors.leftlim(M) == c - 1
    @test ITensors.rightlim(M) == c + 1

    # Test for left-orthogonality
    L = M[1] * prime(M[1], "Link")
    l = linkind(M, 1)
    @test cpu(L) ≈ delta(l, l') rtol = 1E-12
    for j in 2:(c - 1)
      L = L * M[j] * prime(M[j], "Link")
      l = linkind(M, j)
      @test cpu(L) ≈ delta(l, l') rtol = 1E-12
    end

    # Test for right-orthogonality
    R = M[N] * prime(M[N], "Link")
    r = linkind(M, N - 1)
    @test cpu(R) ≈ delta(r, r') rtol = 1E-12
    for j in reverse((c + 1):(N - 1))
      R = R * M[j] * prime(M[j], "Link")
      r = linkind(M, j - 1)
      @test cpu(R) ≈ delta(r, r') rtol = 1E-12
    end

    @test norm(M[c]) ≈ 1.0
  end

  @testset "truncate! method" begin
    M = basicRandomCuMPS(N; dim=10)
    M0 = copy(M)
    truncate!(M; maxdim=5)
    @test ITensors.rightlim(M) == 2
    # Test for right-orthogonality
    R = M[N] * prime(M[N], "Link")
    r = linkind(M, N - 1)
    @test cpu(R) ≈ delta(r, r') rtol = 1E-12
    for j in reverse(2:(N - 1))
      R = R * M[j] * prime(M[j], "Link")
      r = linkind(M, j - 1)
      @test cpu(R) ≈ delta(r, r') rtol = 1E-12
    end
    @test inner(M0, M) > 0.1
  end
end

#=@testset "Other MPS methods" begin

  @testset "sample! method" begin
    N = 10
    sites = [Index(3,"Site,n=$n") for n=1:N]
    psi = makeRandomCuMPS(sites,chi=3)
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

end=#
