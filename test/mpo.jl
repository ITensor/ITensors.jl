using Combinatorics
using ITensors
using Test

include("util.jl")

function basicRandomMPO(sites; dim=4)
  M = MPO(sites)
  N = length(M)
  links = [Index(dim,"n=$(n-1),Link") for n=1:N+1]
  for n=1:N
    M[n] = randomITensor(links[n],sites[n],sites[n]',links[n+1])
  end
  M[1] *= delta(links[1])
  M[N] *= delta(links[N+1])
  return M
end

@testset "[first]siteinds(::MPO)" begin
  N = 5
  s = siteinds("S=1/2", N)
  M = randomMPO(s)
  v = siteinds(M)
  for n in 1:N
    @test hassameinds(v[n], (s[n], s[n]'))
  end
  @test firstsiteinds(M) == s
end

@testset "MPO Basics" begin
  N = 6
  sites = [Index(2,"Site,n=$n") for n=1:N]
  @test length(MPO()) == 0
  O = MPO(sites)
  @test length(O) == N

  str = split(sprint(show, O), '\n')
  @test str[1] == "MPO"
  @test length(str) == length(O) + 2

  O[1] = ITensor(sites[1], prime(sites[1]))
  @test hasind(O[1],sites[1])
  @test hasind(O[1],prime(sites[1]))
  P = copy(O)
  @test hasind(P[1],sites[1])
  @test hasind(P[1],prime(sites[1]))
  # test constructor from Vector{ITensor}
  K = randomMPO(sites)
  @test ITensors.data(MPO(copy(ITensors.data(K)))) == ITensors.data(K)

  @testset "orthogonalize!" begin
    phi = randomMPS(sites)
    K = randomMPO(sites)
    orthogonalize!(phi, 1)
    orthogonalize!(K, 1)
    orig_inner = ⋅(phi, K, phi)
    orthogonalize!(phi, div(N, 2))
    orthogonalize!(K, div(N, 2))
    @test ⋅(phi, K, phi) ≈ orig_inner
  end

  @testset "norm MPO" begin
    A = randomMPO(sites)
    Adag = sim(linkinds, dag(A))
    A² = ITensor(1)
    for j = 1:N
      A² *= Adag[j] * A[j]
    end
    @test A²[] ≈ inner(A, A)
    @test sqrt(A²[]) ≈ norm(A)
    for j in 1:N
      A[j] ./= j
    end
    @test norm(A) ≈ 1 / factorial(N)
  end

  @testset "lognorm MPO" begin
    A = randomMPO(sites)
    for j in 1:N
      A[j] .*= j
    end
    Adag = sim(linkinds, dag(A))
    A² = ITensor(1)
    for j = 1:N
      A² *= Adag[j] * A[j]
    end
    @test A²[] ≈ A ⋅ A
    @test 0.5 * log(A²[]) ≈ lognorm(A)
    @test lognorm(A) ≈ log(factorial(N))
  end

  @testset "inner <y|A|x>" begin
    phi = randomMPS(sites)
    K = randomMPO(sites)
    @test maxlinkdim(K) == 1
    psi = randomMPS(sites)
    phidag = dag(phi)
    prime!(phidag)
    phiKpsi = phidag[1]*K[1]*psi[1]
    for j = 2:N
      phiKpsi *= phidag[j]*K[j]*psi[j]
    end
    @test phiKpsi[] ≈ inner(phi,K,psi)

    badsites = [Index(2,"Site") for n=1:N+1]
    badpsi = randomMPS(badsites)
    @test_throws DimensionMismatch inner(phi,K,badpsi)
    
    # make bigger random MPO...
    for link_dim in 2:5
        mpo_tensors  = ITensor[ITensor() for ii in 1:N]
        mps_tensors  = ITensor[ITensor() for ii in 1:N]
        mps_tensors2 = ITensor[ITensor() for ii in 1:N]
        mpo_link_inds = [Index(link_dim, "r$ii,Link") for ii in 1:N-1]
        mps_link_inds = [Index(link_dim, "r$ii,Link") for ii in 1:N-1]
        mpo_tensors[1] = randomITensor(mpo_link_inds[1], sites[1], sites[1]') 
        mps_tensors[1] = randomITensor(mps_link_inds[1], sites[1]) 
        mps_tensors2[1] = randomITensor(mps_link_inds[1], sites[1]) 
        for ii in 2:N-1
            mpo_tensors[ii] = randomITensor(mpo_link_inds[ii], mpo_link_inds[ii-1], sites[ii], sites[ii]') 
            mps_tensors[ii] = randomITensor(mps_link_inds[ii], mps_link_inds[ii-1], sites[ii]) 
            mps_tensors2[ii] = randomITensor(mps_link_inds[ii], mps_link_inds[ii-1], sites[ii]) 
        end
        mpo_tensors[N] = randomITensor(mpo_link_inds[N-1], sites[N], sites[N]')
        mps_tensors[N] = randomITensor(mps_link_inds[N-1], sites[N])
        mps_tensors2[N] = randomITensor(mps_link_inds[N-1], sites[N])
        K   = MPO(mpo_tensors, 0, N+1)
        psi = MPS(mps_tensors, 0, N+1)
        phi = MPS(mps_tensors2, 0, N+1)
        orthogonalize!(psi, 1; maxdim=link_dim)
        orthogonalize!(K, 1; maxdim=link_dim)
        orthogonalize!(phi, 1; normalize=true, maxdim=link_dim)
        phidag = dag(phi)
        prime!(phidag)
        phiKpsi = phidag[1]*K[1]*psi[1]
        for j = 2:N
          phiKpsi *= phidag[j]*K[j]*psi[j]
        end
        @test scalar(phiKpsi) ≈ inner(phi,K,psi)
    end
  end

  @testset "inner <By|A|x>" begin
    phi = makeRandomMPS(sites)

    K = makeRandomMPO(sites,chi=2)
    J = makeRandomMPO(sites,chi=2)

    psi = makeRandomMPS(sites)
    phidag = dag(phi)
    prime!(phidag, 2)
    Jdag = dag(J)
    prime!(Jdag)
    for j ∈ eachindex(Jdag)
      swapprime!(Jdag[j],2,3)
      swapprime!(Jdag[j],1,2)
      swapprime!(Jdag[j],3,1)
    end

    phiJdagKpsi = phidag[1]*Jdag[1]*K[1]*psi[1]
    for j ∈ eachindex(psi)[2:end]
      phiJdagKpsi = phiJdagKpsi*phidag[j]*Jdag[j]*K[j]*psi[j]
    end

    @test phiJdagKpsi[] ≈ inner(J,phi,K,psi)

    badsites = [Index(2,"Site") for n=1:N+1]
    badpsi = randomMPS(badsites)
    @test_throws DimensionMismatch inner(J,phi,K,badpsi)
  end

  @testset "error_contract" begin
    phi = makeRandomMPS(sites)
    K = makeRandomMPO(sites,chi=2)

    psi = makeRandomMPS(sites)

    dist = sqrt(abs(1 + (inner(phi,phi) - 2*real(inner(phi,K,psi)))
                        /inner(K,psi,K,psi)))
    @test dist ≈ error_contract(phi,K,psi)

    badsites = [Index(2,"Site") for n=1:N+1]
    badpsi = randomMPS(badsites)
    # Apply K to phi and check that error_contract is close to 0.
    Kphi = contract(K, phi; method="naive", cutoff=1E-8)
    @test error_contract(Kphi, K, phi) ≈ 0. atol=1e-4

    @test_throws DimensionMismatch contract(K,badpsi;method="naive", cutoff=1E-8)
    @test_throws DimensionMismatch error_contract(phi,K,badpsi)
  end

  @testset "contract" begin
    phi = randomMPS(sites)
    K   = randomMPO(sites)
    @test maxlinkdim(K) == 1
    psi = randomMPS(sites)
    psi_out = contract(K, psi; maxdim=1)
    @test inner(phi,psi_out) ≈ inner(phi,K,psi)
    @test_throws ArgumentError contract(K, psi, method="fakemethod")

    badsites = [Index(2,"Site") for n=1:N+1]
    badpsi = randomMPS(badsites)
    @test_throws DimensionMismatch contract(K,badpsi)

    # make bigger random MPO...
    for link_dim in 2:5
      mpo_tensors  = ITensor[ITensor() for ii in 1:N]
      mps_tensors  = ITensor[ITensor() for ii in 1:N]
      mps_tensors2 = ITensor[ITensor() for ii in 1:N]
      mpo_link_inds = [Index(link_dim, "r$ii,Link") for ii in 1:N-1]
      mps_link_inds = [Index(link_dim, "r$ii,Link") for ii in 1:N-1]
      mpo_tensors[1] = randomITensor(mpo_link_inds[1], sites[1], sites[1]') 
      mps_tensors[1] = randomITensor(mps_link_inds[1], sites[1]) 
      mps_tensors2[1] = randomITensor(mps_link_inds[1], sites[1]) 
      for ii in 2:N-1
        mpo_tensors[ii] = randomITensor(mpo_link_inds[ii], mpo_link_inds[ii-1], sites[ii], sites[ii]') 
        mps_tensors[ii] = randomITensor(mps_link_inds[ii], mps_link_inds[ii-1], sites[ii]) 
        mps_tensors2[ii] = randomITensor(mps_link_inds[ii], mps_link_inds[ii-1], sites[ii]) 
      end
      mpo_tensors[N] = randomITensor(mpo_link_inds[N-1], sites[N], sites[N]')
      mps_tensors[N] = randomITensor(mps_link_inds[N-1], sites[N])
      mps_tensors2[N] = randomITensor(mps_link_inds[N-1], sites[N])
      K   = MPO(mpo_tensors, 0, N+1)
      psi = MPS(mps_tensors, 0, N+1)
      phi = MPS(mps_tensors2, 0, N+1)
      orthogonalize!(psi, 1; maxdim=link_dim)
      orthogonalize!(K, 1; maxdim=link_dim)
      orthogonalize!(phi, 1; normalize=true, maxdim=link_dim)
      psi_out = contract(deepcopy(K), deepcopy(psi); maxdim=10*link_dim, cutoff=0.0)
      @test inner(phi, psi_out) ≈ inner(phi, K, psi)
    end
  end

  @testset "add(::MPO, ::MPO)" begin
    shsites = siteinds("S=1/2", N)
    K = randomMPO(shsites)
    L = randomMPO(shsites)
    M = add(K, L)
    @test length(M) == N
    psi = randomMPS(shsites)
    k_psi = contract(K, psi, maxdim=1)
    l_psi = contract(L, psi, maxdim=1)
    @test inner(psi, k_psi + l_psi) ≈ ⋅(psi, M, psi) atol=5e-3
    @test inner(psi, sum([k_psi, l_psi])) ≈ dot(psi, M, psi) atol=5e-3
    for dim in 2:4
        shsites = siteinds("S=1/2",N)
        K = basicRandomMPO(shsites; dim=dim)
        L = basicRandomMPO(shsites; dim=dim)
        M = K + L
        @test length(M) == N
        psi = randomMPS(shsites)
        k_psi = contract(K, psi)
        l_psi = contract(L, psi)
        @test inner(psi, k_psi + l_psi) ≈ dot(psi, M, psi) atol=5e-3
        @test inner(psi, sum([k_psi, l_psi])) ≈ inner(psi, M, psi) atol=5e-3
        psi = randomMPS(shsites)
        M = add(K, L; cutoff=1E-9)
        k_psi = contract(K, psi)
        l_psi = contract(L, psi)
        @test inner(psi, k_psi + l_psi) ≈ inner(psi, M, psi) atol=5e-3
    end
  end

  @testset "+(::MPO, ::MPO)" begin
    conserve_qns = true
    s = siteinds("S=1/2", N; conserve_qns = conserve_qns)

    ops = n -> isodd(n) ? "Sz" : "Id"
    H₁ = MPO(s, ops)
    H₂ = MPO(s, ops)

    H = H₁ + H₂

    @test inner(H, H) ≈ inner_add(H₁, H₂)
    @test maxlinkdim(H) ≤ maxlinkdim(H₁) + maxlinkdim(H₂)

    α₁ = 2.2
    α₂ = 3.4 + 1.2im

    H = α₁ * H₁ + H₂

    @test inner(H, H) ≈ inner_add((α₁, H₁), H₂)
    @test maxlinkdim(H) ≤ maxlinkdim(H₁) + maxlinkdim(H₂)

    H = H₁ - H₂

    @test inner(H, H) ≈ inner_add(H₁, (-1, H₂))
    @test maxlinkdim(H) ≤ maxlinkdim(H₁) + maxlinkdim(H₂)

    H = α₁ * H₁ - α₂ * H₂

    @test inner(H, H) ≈ inner_add((α₁, H₁), (-α₂, H₂))
    @test maxlinkdim(H) ≤ maxlinkdim(H₁) + maxlinkdim(H₂)
  end

  @testset "contract(::MPO, ::MPO)" begin
    psi = randomMPS(sites)
    K = randomMPO(sites)
    L = randomMPO(sites)
    @test maxlinkdim(K) == 1
    @test maxlinkdim(L) == 1
    KL = contract(prime(K), L, maxdim=1)
    psi_kl_out = contract(prime(K), contract(L, psi, maxdim=1), maxdim=1)
    @test inner(psi,KL,psi) ≈ inner(psi, psi_kl_out) atol=5e-3

    # where both K and L have differently labelled sites
    othersitesk = [Index(2,"Site,aaa") for n=1:N]
    othersitesl = [Index(2,"Site,bbb") for n=1:N]
    K = randomMPO(sites)
    L = randomMPO(sites)
    for ii in 1:N
      replaceind!(K[ii], sites[ii]', othersitesk[ii])
      replaceind!(L[ii], sites[ii]', othersitesl[ii])
    end
    KL = contract(K, L, maxdim=1)
    psik = randomMPS(othersitesk)
    psil = randomMPS(othersitesl)
    psi_kl_out = contract(K, contract(L, psil, maxdim=1), maxdim=1)
    @test inner(psik,KL,psil) ≈ inner(psik, psi_kl_out) atol=5e-3
    
    badsites = [Index(2,"Site") for n=1:N+1]
    badL = randomMPO(badsites)
    @test_throws DimensionMismatch contract(K,badL)
  end

  @testset "*(::MPO, ::MPO)" begin
    psi = randomMPS(sites)
    K = randomMPO(sites)
    L = randomMPO(sites)
    @test maxlinkdim(K) == 1
    @test maxlinkdim(L) == 1
    KL = *(prime(K), L, maxdim=1)
    psi_kl_out = *(prime(K), *(L, psi, maxdim=1), maxdim=1)
    @test ⋅(psi, KL, psi) ≈ dot(psi, psi_kl_out) atol=5e-3

    # where both K and L have differently labelled sites
    othersitesk = [Index(2,"Site,aaa") for n=1:N]
    othersitesl = [Index(2,"Site,bbb") for n=1:N]
    K = randomMPO(sites)
    L = randomMPO(sites)
    for ii in 1:N
      replaceind!(K[ii], sites[ii]', othersitesk[ii])
      replaceind!(L[ii], sites[ii]', othersitesl[ii])
    end
    KL = *(K, L, maxdim=1)
    psik = randomMPS(othersitesk)
    psil = randomMPS(othersitesl)
    psi_kl_out = *(K, *(L, psil, maxdim=1), maxdim=1)
    @test dot(psik, KL, psil) ≈ psik ⋅ psi_kl_out atol=5e-3
    
    badsites = [Index(2,"Site") for n=1:N+1]
    badL = randomMPO(badsites)
    @test_throws DimensionMismatch K * badL
  end

  sites = siteinds("S=1/2",N)
  O = MPO(sites,"Sz")
  @test length(O) == N # just make sure this works

  @test_throws ArgumentError randomMPO(sites, 2)
  @test isnothing(linkind(MPO(fill(ITensor(), N), 0, N + 1), 1))

  @testset "movesites $N sites" for N in 1:7
    s0 = siteinds("S=1/2", N)
    ψ0 = MPO(s0, "Id")
    for perm in permutations(1:N)
      s = s0[perm]
      ψ = randomMPO(s)
      ns′ = [findsite(ψ0, i) for i in s]
      @test ns′ == perm
      ψ′ = movesites(ψ, 1:N .=> ns′)
      for n in 1:N
        @test hassameinds(siteinds(ψ0, n), siteinds(ψ′, n))
      end
      @test @set_warn_order 15 prod(ψ) ≈ prod(ψ′)
    end
  end

  @testset "Construct MPO from ITensor" begin

    N = 5
    s = siteinds("S=1/2", N)
    l = [Index(3, "left_$n") for n in 1:2]
    r = [Index(3, "right_$n") for n in 1:2]

    sis = IndexSet.(prime.(s), s)

    A = randomITensor(s..., prime.(s)...)
    ψ = MPO(A, sis; orthocenter = 4)
    ls = linkinds(ψ)
    @test hassameinds(ψ[1], (s[1], s[1]', ls[1]))
    @test hassameinds(ψ[N], (s[N], s[N]', ls[N - 1]))
    @test prod(ψ) ≈ A
    @test ITensors.orthocenter(ψ) == 4
    @test maxlinkdim(ψ) == 16

    A = randomITensor(s..., prime.(s)...)
    ψ = MPO(A, s; orthocenter = 4)
    ls = linkinds(ψ)
    @test hassameinds(ψ[1], (s[1], s[1]', ls[1]))
    @test hassameinds(ψ[N], (s[N], s[N]', ls[N - 1]))
    @test prod(ψ) ≈ A
    @test ITensors.orthocenter(ψ) == 4
    @test maxlinkdim(ψ) == 16

    ψ0 = MPO(s, "Id")
    A = prod(ψ0)
    ψ = MPO(A, sis; cutoff = 1e-15, orthocenter = 3)
    ls = linkinds(ψ)
    @test hassameinds(ψ[1], (s[1], s[1]', ls[1]))
    @test hassameinds(ψ[N], (s[N], s[N]', ls[N - 1]))
    @test prod(ψ) ≈ A
    @test ITensors.orthocenter(ψ) == 3
    @test maxlinkdim(ψ) == 1

    ψ0 = MPO(s, "Id")
    A = prod(ψ0)
    ψ = MPO(A, s; cutoff = 1e-15, orthocenter = 3)
    ls = linkinds(ψ)
    @test hassameinds(ψ[1], (s[1], s[1]', ls[1]))
    @test hassameinds(ψ[N], (s[N], s[N]', ls[N - 1]))
    @test prod(ψ) ≈ A
    @test ITensors.orthocenter(ψ) == 3
    @test maxlinkdim(ψ) == 1

    A = randomITensor(s..., prime.(s)..., l[1], r[1])
    ψ = MPO(A, sis, leftinds = l[1])
    ls = linkinds(ψ)
    @test hassameinds(ψ[1], (l[1], s[1], s[1]', ls[1]))
    @test hassameinds(ψ[N], (r[1], s[N], s[N]', ls[N - 1]))
    @test prod(ψ) ≈ A
    @test ITensors.orthocenter(ψ) == N
    @test maxlinkdim(ψ) == 48

    A = randomITensor(s..., prime.(s)..., l[1], r[1])
    ψ = MPO(A, s, leftinds = l[1])
    ls = linkinds(ψ)
    @test hassameinds(ψ[1], (l[1], s[1], s[1]', ls[1]))
    @test hassameinds(ψ[N], (r[1], s[N], s[N]', ls[N - 1]))
    @test prod(ψ) ≈ A
    @test ITensors.orthocenter(ψ) == N
    @test maxlinkdim(ψ) == 48

    A = randomITensor(s..., prime.(s)..., l..., r...)
    ψ = MPO(A, sis, leftinds = l, orthocenter = 2)
    ls = linkinds(ψ)
    @test hassameinds(ψ[1], (l..., s[1], s[1]', ls[1]))
    @test hassameinds(ψ[N], (r..., s[N], s[N]', ls[N - 1]))
    @test @set_warn_order 15 prod(ψ) ≈ A
    @test ITensors.orthocenter(ψ) == 2
    @test maxlinkdim(ψ) == 144

    A = randomITensor(s..., prime.(s)..., l..., r...)
    ψ = MPO(A, s, leftinds = l, orthocenter = 2)
    ls = linkinds(ψ)
    @test hassameinds(ψ[1], (l..., s[1], s[1]', ls[1]))
    @test hassameinds(ψ[N], (r..., s[N], s[N]', ls[N - 1]))
    @test @set_warn_order 15 prod(ψ) ≈ A
    @test ITensors.orthocenter(ψ) == 2
    @test maxlinkdim(ψ) == 144
  end

  @testset "Set range of MPO tensors" begin
    N = 5
    s = siteinds("S=1/2", N)
    ψ0 = randomMPO(s)

    ψ = orthogonalize(ψ0, 2)
    A = prod(ITensors.data(ψ)[2:N-1])
    randn!(A)
    ϕ = MPO(A, s[2:N-1], orthocenter = 1)
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

  @testset "swapbondsites MPO" begin
    N = 5
    sites = siteinds("S=1/2", N)
    ψ0 = randomMPO(sites)

    # TODO: implement this?
    #ψ = replacebond(ψ0, 3, ψ0[3] * ψ0[4];
    #                swapsites = true,
    #                cutoff = 1e-15)
    #@test siteind(ψ, 1) == siteind(ψ0, 1)
    #@test siteind(ψ, 2) == siteind(ψ0, 2)
    #@test siteind(ψ, 4) == siteind(ψ0, 3)
    #@test siteind(ψ, 3) == siteind(ψ0, 4)
    #@test siteind(ψ, 5) == siteind(ψ0, 5)
    #@test prod(ψ) ≈ prod(ψ0)
    #@test maxlinkdim(ψ) == 1

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

  @testset "MPO(::MPS)" begin
    i = Index(QN(0, 2) => 1, QN(1, 2) => 1; tags = "i")
    j = settags(i, "j")
    A = randomITensor(ComplexF64, i, j)
    M = A' * dag(A)
    ψ = MPS(A, [i, j])
    @test prod(ψ) ≈ A
    ρ = MPO(ψ)
    @test prod(ρ) ≈ M
  end

  @testset "tr(::MPO)" begin
    N = 5
    s = siteinds("S=1/2", N)
    H = MPO(s, "Id")
    d = dim(s[1])
    @test tr(H) ≈ d^N
  end

  @testset "tr(::MPO) multiple site indices" begin
    N = 6
    s = siteinds("S=1/2", N)
    H = MPO(s, "Id")
    H2 = MPO([H[j] * H[j+1] for j in 1:2:N-1])
    d = dim(s[1])
    @test tr(H) ≈ d^N
    @test tr(H2) ≈ d^N
  end

  @testset "check_hascommonsiteinds checks in DMRG, inner, dot" begin
    N = 4
    s1 = siteinds("S=1/2", N)
    s2 = siteinds("S=1/2", N)
    psi1 = randomMPS(s1)
    psi2 = randomMPS(s2)
    H1 = MPO(AutoMPO() + ("Id", 1), s1)
    H2 = MPO(AutoMPO() + ("Id", 1), s2)

    @test_throws ErrorException inner(psi1, H2, psi1)
    @test_throws ErrorException inner(psi1, H2, psi2; make_inds_match = false)

    sweeps = Sweeps(1)
    maxdim!(sweeps, 10)

    @test_throws ErrorException dmrg(H2, psi1, sweeps)
    @test_throws ErrorException dmrg(H1, [psi2], psi1, sweeps)
    @test_throws ErrorException dmrg([H1, H2], psi1, sweeps)
  end

  @testset "MPO*MPO contraction with multiple site indices" begin
    N = 8
    s = siteinds("S=1/2", N)
    a = AutoMPO()
    for j in 1:N-1
      a .+= 0.5, "S+", j, "S-", j+1
      a .+= 0.5, "S-", j, "S+", j+1
      a .+=      "Sz", j, "Sz", j+1
    end
    H = MPO(a, s)
    # Create MPO/MPS with pairs of sites merged
    H2 = MPO([H[b] * H[b+1] for b in 1:2:N])
    @test @disable_warn_order prod(H) ≈ prod(H2)
    HH = H' * H
    H2H2 = H2' * H2
    @test @disable_warn_order prod(HH) ≈ prod(H2H2)
  end

  @testset "MPO*MPO contraction with multiple and combined site indices" begin
    N = 8
    s = siteinds("S=1/2", N)
    a = AutoMPO()
    for j in 1:N-1
      a .+= 0.5, "S+", j, "S-", j+1
      a .+= 0.5, "S-", j, "S+", j+1
      a .+=      "Sz", j, "Sz", j+1
    end
    H = MPO(a, s)
    HH = setprime(H' * H, 1; plev = 2)

    # Create MPO/MPS with pairs of sites merged
    H2 = MPO([H[b] * H[b+1] for b in 1:2:N])
    @test @disable_warn_order prod(H) ≈ prod(H2)
    s = siteinds(H2; plev = 1)
    C = combiner.(s; tags = "X")
    H2 .*= C
    H2H2 = prime(H2; tags = !ts"X") * dag(H2)
    @test @disable_warn_order prod(HH) ≈ prod(H2H2)
  end


end

nothing
