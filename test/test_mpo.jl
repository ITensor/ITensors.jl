using ITensors,
      Test

@testset "MPO Basics" begin
  N = 6
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

  @testset "orthogonalize" begin
    phi = randomMPS(sites)
    K = randomMPO(sites)
    orthogonalize!(phi, 1)
    orthogonalize!(K, 1)
    orig_inner = inner(phi, K, phi)
    orthogonalize!(phi, div(N, 2))
    orthogonalize!(K, div(N, 2))
    @test inner(phi, K, phi) ≈ orig_inner
  end

  @testset "inner <y|A|x>" begin
    phi = randomMPS(sites)
    K = randomMPO(sites)
    @test maxLinkDim(K) == 1
    psi = randomMPS(sites)
    phidag = dag(phi)
    prime!(phidag)
    phiKpsi = phidag[1]*K[1]*psi[1]
    for j = 2:N
      phiKpsi *= phidag[j]*K[j]*psi[j]
    end
    @test phiKpsi[] ≈ inner(phi,K,psi)

    # Do contraction manually.
    O = 1.
    for j ∈ eachindex(phi)
        psij = reshape(Array(psi[j]),2)
        phij = reshape(Array(phi[j]),2)
        Kj = reshape(Array(K[j]),2,2)
        O *= (phij'*transpose(Kj)*psij)[]
    end
    @test O ≈ inner(phi,K,psi)

    badsites = SiteSet(N+1,2)
    badpsi = randomMPS(badsites)
    @test_throws DimensionMismatch inner(phi,K,badpsi)
    
    # make bigger random MPO...
    for link_dim in 2:5
        mpo_tensors  = [ITensor() for ii in 1:N]
        mps_tensors  = [ITensor() for ii in 1:N]
        mps_tensors2 = [ITensor() for ii in 1:N]
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
        K   = MPO(N, mpo_tensors, 0, N+1)
        psi = MPS(N, mps_tensors, 0, N+1)
        phi = MPS(N, mps_tensors2, 0, N+1)
        orthogonalize!(psi, 1; maxdim=link_dim)
        orthogonalize!(K, 1; maxdim=link_dim)
        orthogonalize!(phi, 1; normalize=true, maxdim=link_dim)
        phidag = dag(phi)
        prime!(phidag)
        phiKpsi = phidag[1]*K[1]*psi[1]
        for j = 2:N
          phiKpsi *= phidag[j]*K[j]*psi[j]
        end
        @test phiKpsi[] ≈ inner(phi,K,psi)
    end
  end

  @testset "inner <By|A|x>" begin
    phi = randomMPS(sites)
    K = randomMPO(sites)
    J = randomMPO(sites)
    @test maxLinkDim(K) == 1
    @test maxLinkDim(J) == 1
    psi = randomMPS(sites)
    phidag = dag(phi)
    prime!(phidag, 2)
    Jdag = dag(J)
    prime!(Jdag)
    @inbounds for j ∈ eachindex(Jdag)
      swapprime!(inds(Jdag[j]),2,3)
      swapprime!(inds(Jdag[j]),1,2)
      swapprime!(inds(Jdag[j]),3,1)
    end
    phiJdagKpsi = phidag[1]*Jdag[1]*K[1]*psi[1]
    @inbounds for j ∈ eachindex(psi)[2:end]
      phiJdagKpsi = phiJdagKpsi*phidag[j]*Jdag[j]*K[j]*psi[j]
    end
    @test phiJdagKpsi[] ≈ inner(J,phi,K,psi)

    # Do contraction manually.
    O = 1.
    for j ∈ eachindex(phi)
        psij = reshape(Array(psi[j]),2)
        phij = reshape(Array(phi[j]),2)
        Kj = reshape(Array(K[j]),2,2)
        Jj = reshape(Array(J[j]),2,2)
        O *= ((transpose(Jj)*phij)'*transpose(Kj)*psij)[]
    end
    @test O ≈ inner(J,phi,K,psi)

    badsites = SiteSet(N+1,2)
    badpsi = randomMPS(badsites)
    @test_throws DimensionMismatch inner(J,phi,K,badpsi)
  end

  @testset "errorMPOProd" begin
    phi = randomMPS(sites)
    K = randomMPO(sites)
    @test maxLinkDim(K) == 1
    psi = randomMPS(sites)
    dist = sqrt(abs(1 + (inner(phi,phi) - 2*real(inner(phi,K,psi)))
                        /inner(K,psi,K,psi)))
    @test dist ≈ errorMPOProd(phi,K,psi)

    # Apply K to phi and check that errorMPOProd is close to 0.
    Kphi = applyMPO(K, phi, maxdim=1)
    @test errorMPOProd(Kphi, K, phi) ≈ 0. atol=1e-6

    badsites = SiteSet(N+1,2)
    badpsi = randomMPS(badsites)
    @test_throws DimensionMismatch errorMPOProd(phi,K,badpsi)
  end

  @testset "applyMPO" begin
    phi = randomMPS(sites)
    K   = randomMPO(sites)
    @test maxLinkDim(K) == 1
    psi = randomMPS(sites)
    psi_out = applyMPO(K, psi,maxdim=1)
    @test inner(phi,psi_out) ≈ inner(phi,K,psi)
    @test_throws ArgumentError applyMPO(K, psi, method="fakemethod")

    badsites = SiteSet(N+1,2)
    badpsi = randomMPS(badsites)
    @test_throws DimensionMismatch applyMPO(K,badpsi)

    # make bigger random MPO...
    for link_dim in 2:5
        mpo_tensors  = [ITensor() for ii in 1:N]
        mps_tensors  = [ITensor() for ii in 1:N]
        mps_tensors2 = [ITensor() for ii in 1:N]
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
        K   = MPO(N, mpo_tensors, 0, N+1)
        psi = MPS(N, mps_tensors, 0, N+1)
        phi = MPS(N, mps_tensors2, 0, N+1)
        orthogonalize!(psi, 1; maxdim=link_dim)
        orthogonalize!(K, 1; maxdim=link_dim)
        orthogonalize!(phi, 1; normalize=true, maxdim=link_dim)
        psi_out = applyMPO(deepcopy(K), deepcopy(psi); maxdim=10*link_dim, cutoff=0.0)
        @test inner(phi, psi_out) ≈ inner(phi, K, psi)
    end
  end
  @testset "add" begin
    shsites = spinHalfSites(N)
    K = randomMPO(shsites)
    L = randomMPO(shsites)
    M = sum(K, L)
    @test length(M) == N
    psi = randomMPS(shsites)
    k_psi = applyMPO(K, psi, maxdim=1)
    l_psi = applyMPO(L, psi, maxdim=1)
    @test inner(psi, sum(k_psi, l_psi)) ≈ inner(psi, M, psi) atol=5e-3
  end

  @testset "nmultMPO" begin
    psi = randomMPS(sites)
    K = randomMPO(sites)
    L = randomMPO(sites)
    @test maxLinkDim(K) == 1
    @test maxLinkDim(L) == 1
    KL = nmultMPO(K, L, maxdim=1)
    psi_kl_out = applyMPO(K, applyMPO(L, psi, maxdim=1), maxdim=1)
    @test inner(psi,KL,psi) ≈ inner(psi, psi_kl_out) atol=5e-3

    badsites = SiteSet(N+1,2)
    badL = randomMPO(badsites)
    @test_throws DimensionMismatch nmultMPO(K,badL)
  end

  sites = spinHalfSites(N)
  O = MPO(sites,"Sz")
  @test length(O) == N # just make sure this works

  @test_throws ArgumentError randomMPO(sites, 2)
  @test_throws ErrorException linkindex(MPO(N, fill(ITensor(), N), 0, N + 1), 1)
end
