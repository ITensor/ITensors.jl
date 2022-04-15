using ITensors, ITensorGPU, Test

@testset "CuMPO Basics" begin
  N = 6
  sites = [Index(2, "Site") for n in 1:N]
  @test length(cuMPO()) == 0
  O = cuMPO(sites)
  @test length(O) == N

  str = split(sprint(show, O), '\n')
  @test str[1] == "MPO"
  @test length(str) == length(O) + 2

  O[1] = cuITensor(sites[1], prime(sites[1]))
  @test hasind(O[1], sites[1])
  @test hasind(O[1], prime(sites[1]))
  P = copy(O)
  @test hasind(P[1], sites[1])
  @test hasind(P[1], prime(sites[1]))

  K = randomCuMPO(sites)
  K_ = cuMPO(ITensors.data(K))
  @test all(ITensors.data(K) .== ITensors.data(K_))

  s = siteinds("S=1/2", N)
  L = randomMPO(s)
  K = cuMPO(L)
  @test all(ITensors.data(cpu(K)) .== ITensors.data(cpu(L)))
  @testset "orthogonalize" begin
    phi = randomCuMPS(sites)
    K = randomCuMPO(sites)
    orthogonalize!(phi, 1)
    orthogonalize!(K, 1)
    orig_inner = inner(phi', K, phi)
    orthogonalize!(phi, div(N, 2))
    orthogonalize!(K, div(N, 2))
    @test inner(phi', K, phi) ≈ orig_inner
  end

  @testset "inner <y|A|x>" begin
    phi = randomCuMPS(sites)
    K = randomCuMPO(sites)
    @test maxlinkdim(K) == 1
    psi = randomCuMPS(sites)
    phidag = dag(phi)
    prime!(phidag)
    phiKpsi = phidag[1] * K[1] * psi[1]
    for j in 2:N
      phiKpsi *= phidag[j] * K[j] * psi[j]
    end
    @test phiKpsi[] ≈ inner(phi', K, psi)

    badsites = [Index(2, "Site") for n in 1:(N + 1)]
    badpsi = randomCuMPS(badsites)
    @test_throws DimensionMismatch inner(phi', K, badpsi)

    # make bigger random MPO...
    for link_dim in 2:5
      mpo_tensors = ITensor[cuITensor() for ii in 1:N]
      mps_tensors = ITensor[cuITensor() for ii in 1:N]
      mps_tensors2 = ITensor[cuITensor() for ii in 1:N]
      mpo_link_inds = [Index(link_dim, "r$ii,Link") for ii in 1:(N - 1)]
      mps_link_inds = [Index(link_dim, "r$ii,Link") for ii in 1:(N - 1)]
      mpo_tensors[1] = randomCuITensor(mpo_link_inds[1], sites[1], sites[1]')
      mps_tensors[1] = randomCuITensor(mps_link_inds[1], sites[1])
      mps_tensors2[1] = randomCuITensor(mps_link_inds[1], sites[1])
      for ii in 2:(N - 1)
        mpo_tensors[ii] = randomCuITensor(
          mpo_link_inds[ii], mpo_link_inds[ii - 1], sites[ii], sites[ii]'
        )
        mps_tensors[ii] = randomCuITensor(
          mps_link_inds[ii], mps_link_inds[ii - 1], sites[ii]
        )
        mps_tensors2[ii] = randomCuITensor(
          mps_link_inds[ii], mps_link_inds[ii - 1], sites[ii]
        )
      end
      mpo_tensors[N] = randomCuITensor(mpo_link_inds[N - 1], sites[N], sites[N]')
      mps_tensors[N] = randomCuITensor(mps_link_inds[N - 1], sites[N])
      mps_tensors2[N] = randomCuITensor(mps_link_inds[N - 1], sites[N])
      K = MPO(mpo_tensors, 0, N + 1)
      psi = MPS(mps_tensors, 0, N + 1)
      phi = MPS(mps_tensors2, 0, N + 1)
      orthogonalize!(psi, 1; maxdim=link_dim)
      orthogonalize!(K, 1; maxdim=link_dim)
      orthogonalize!(phi, 1; normalize=true, maxdim=link_dim)
      phidag = dag(phi)
      prime!(phidag)
      phiKpsi = phidag[1] * K[1] * psi[1]
      for j in 2:N
        phiKpsi *= phidag[j] * K[j] * psi[j]
      end
      @test scalar(phiKpsi) ≈ inner(phi', K, psi)
    end
  end

  @testset "contract" begin
    phi = randomCuMPS(sites)
    K = randomCuMPO(sites)
    @test maxlinkdim(K) == 1
    psi = randomCuMPS(sites)
    psi_out = contract(K, psi; maxdim=1)
    @test inner(phi', psi_out) ≈ inner(phi', K, psi)
    @test_throws MethodError contract(K', psi, method="fakemethod")

    badsites = [Index(2, "Site") for n in 1:(N + 1)]
    badpsi = randomCuMPS(badsites)
    @test_throws DimensionMismatch contract(K, badpsi)

    # make bigger random MPO...
    for link_dim in 2:5
      mpo_tensors = ITensor[ITensor() for ii in 1:N]
      mps_tensors = ITensor[ITensor() for ii in 1:N]
      mps_tensors2 = ITensor[ITensor() for ii in 1:N]
      mpo_link_inds = [Index(link_dim, "r$ii,Link") for ii in 1:(N - 1)]
      mps_link_inds = [Index(link_dim, "r$ii,Link") for ii in 1:(N - 1)]
      mpo_tensors[1] = randomCuITensor(mpo_link_inds[1], sites[1], sites[1]')
      mps_tensors[1] = randomCuITensor(mps_link_inds[1], sites[1])
      mps_tensors2[1] = randomCuITensor(mps_link_inds[1], sites[1])
      for ii in 2:(N - 1)
        mpo_tensors[ii] = randomCuITensor(
          mpo_link_inds[ii], mpo_link_inds[ii - 1], sites[ii], sites[ii]'
        )
        mps_tensors[ii] = randomCuITensor(
          mps_link_inds[ii], mps_link_inds[ii - 1], sites[ii]
        )
        mps_tensors2[ii] = randomCuITensor(
          mps_link_inds[ii], mps_link_inds[ii - 1], sites[ii]
        )
      end
      mpo_tensors[N] = randomCuITensor(mpo_link_inds[N - 1], sites[N], sites[N]')
      mps_tensors[N] = randomCuITensor(mps_link_inds[N - 1], sites[N])
      mps_tensors2[N] = randomCuITensor(mps_link_inds[N - 1], sites[N])
      K = MPO(mpo_tensors, 0, N + 1)
      psi = MPS(mps_tensors, 0, N + 1)
      phi = MPS(mps_tensors2, 0, N + 1)
      orthogonalize!(psi, 1; maxdim=link_dim)
      orthogonalize!(K, 1; maxdim=link_dim)
      orthogonalize!(phi, 1; normalize=true, maxdim=link_dim)
      psi_out = contract(deepcopy(K), deepcopy(psi); maxdim=10 * link_dim, cutoff=0.0)
      @test inner(phi', psi_out) ≈ inner(phi', K, psi)
    end
  end
  @testset "add" begin
    shsites = siteinds("S=1/2", N)
    K = randomCuMPO(shsites)
    L = randomCuMPO(shsites)
    M = add(K, L)
    @test length(M) == N
    psi = randomCuMPS(shsites)
    k_psi = contract(K, psi; maxdim=1)
    l_psi = contract(L, psi; maxdim=1)
    @test inner(psi', add(k_psi, l_psi)) ≈ inner(psi', M, psi) atol = 5e-3
  end
  @testset "contract(::CuMPO, ::CuMPO)" begin
    psi = randomCuMPS(sites)
    K = randomCuMPO(sites)
    L = randomCuMPO(sites)
    @test maxlinkdim(K) == 1
    @test maxlinkdim(L) == 1
    KL = contract(prime(K), L; maxdim=1)
    psi_kl_out = contract(prime(K), contract(L, psi; maxdim=1); maxdim=1)
    @test inner(psi'', KL, psi) ≈ inner(psi'', psi_kl_out) atol = 5e-3

    # where both K and L have differently labelled sites
    othersitesk = [Index(2, "Site,aaa") for n in 1:N]
    othersitesl = [Index(2, "Site,bbb") for n in 1:N]
    K = randomCuMPO(sites)
    L = randomCuMPO(sites)
    for ii in 1:N
      replaceind!(K[ii], sites[ii]', othersitesk[ii])
      replaceind!(L[ii], sites[ii]', othersitesl[ii])
    end
    KL = contract(K, L; maxdim=1)
    psik = randomCuMPS(othersitesk)
    psil = randomCuMPS(othersitesl)
    psi_kl_out = contract(K, contract(L, psil; maxdim=1); maxdim=1)
    @test inner(psik, KL, psil) ≈ inner(psik, psi_kl_out) atol = 5e-3

    badsites = [Index(2, "Site") for n in 1:(N + 1)]
    badL = randomCuMPO(badsites)
    @test_throws DimensionMismatch contract(K, badL)
  end
end
