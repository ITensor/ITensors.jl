using ITensors, Test

function op_mpo(sites, which_op, j)
  left_ops = "Id"
  right_ops = "Id"
  if has_fermion_string(which_op, sites[j])
    left_ops = "F"
  end
  ops = [n < j ? left_ops : (n > j ? right_ops : which_op) for n in 1:length(sites)]
  return MPO([op(ops[n], sites[n]) for n in 1:length(sites)])
end

@testset "MPO Basics" begin
  N = 6
  sites = [Index(QN(-1) => 1, QN(1) => 1; tags="Site,n=$n") for n in 1:N]
  links = [Index(QN() => 1; tags="Links,l=$n") for n in 1:(N - 1)]
  @test length(MPO()) == 0
  #O = MPO(sites)
  O = MPO(N)
  for i in 1:length(O)
    O[i] = randomITensor(QN(), sites[i], sites[i]')
  end
  @test length(O) == N

  O[1] = emptyITensor(sites[1], prime(sites[1]))
  @test hasind(O[1], sites[1])
  @test hasind(O[1], prime(sites[1]))
  P = copy(O)
  @test hasind(P[1], sites[1])
  @test hasind(P[1], prime(sites[1]))
  # test constructor from Vector{ITensor}

  K = MPO(N)
  K[1] = randomITensor(QN(), dag(sites[1]), sites[1]', links[1])
  for i in 2:(N - 1)
    K[i] = randomITensor(QN(), dag(sites[i]), sites[i]', dag(links[i - 1]), links[i])
  end
  K[N] = randomITensor(QN(), dag(sites[N]), sites[N]', dag(links[N - 1]))

  J = MPO(N)
  J[1] = randomITensor(QN(), dag(sites[1]), sites[1]', links[1])
  for i in 2:(N - 1)
    J[i] = randomITensor(QN(), dag(sites[i]), sites[i]', dag(links[i - 1]), links[i])
  end
  J[N] = randomITensor(QN(), dag(sites[N]), sites[N]', dag(links[N - 1]))

  L = MPO(N)
  L[1] = randomITensor(QN(), dag(sites[1]), sites[1]', links[1])
  for i in 2:(N - 1)
    L[i] = randomITensor(QN(), dag(sites[i]), sites[i]', dag(links[i - 1]), links[i])
  end
  L[N] = randomITensor(QN(), dag(sites[N]), sites[N]', dag(links[N - 1]))

  @test length(K) == N
  @test ITensors.data(MPO(copy(ITensors.data(K)))) == ITensors.data(K)

  phi = MPS(N)
  phi[1] = randomITensor(QN(-1), sites[1], links[1])
  for i in 2:(N - 1)
    phi[i] = randomITensor(QN(-1), sites[i], dag(links[i - 1]), links[i])
  end
  phi[N] = randomITensor(QN(-1), sites[N], dag(links[N - 1]))

  psi = MPS(N)
  psi[1] = randomITensor(QN(-1), sites[1], links[1])
  for i in 2:(N - 1)
    psi[i] = randomITensor(QN(-1), sites[i], dag(links[i - 1]), links[i])
  end
  psi[N] = randomITensor(QN(-1), sites[N], dag(links[N - 1]))

  @testset "orthogonalize!" begin
    orthogonalize!(phi, 1)
    orthogonalize!(K, 1)
    orig_inner = ⋅(phi', K, phi)
    orthogonalize!(phi, div(N, 2))
    orthogonalize!(K, div(N, 2))
    @test ⋅(phi', K, phi) ≈ orig_inner
  end

  @testset "inner <y|A|x>" begin
    @test maxlinkdim(K) == 1
    phidag = dag(phi)
    prime!(phidag)
    phiKpsi = phidag[1] * K[1] * psi[1]
    for j in 2:N
      phiKpsi *= phidag[j] * K[j] * psi[j]
    end
    @test phiKpsi[] ≈ inner(phi', K, psi)
  end

  @testset "inner <By|A|x>" begin
    phidag = dag(phi)
    prime!(phidag, 2)
    Jdag = dag(J)
    prime!(Jdag)
    for j in eachindex(Jdag)
      swapprime!(Jdag[j], 2, 3)
      swapprime!(Jdag[j], 1, 2)
      swapprime!(Jdag[j], 3, 1)
    end

    phiJdagKpsi = phidag[1] * Jdag[1] * K[1] * psi[1]
    for j in eachindex(psi)[2:end]
      phiJdagKpsi = phiJdagKpsi * phidag[j] * Jdag[j] * K[j] * psi[j]
    end

    @test phiJdagKpsi[] ≈ inner(J, phi, K, psi)

    badsites = [Index(2, "Site") for n in 1:(N + 1)]
    badpsi = randomMPS(badsites)
    @test_throws DimensionMismatch inner(J, phi, K, badpsi)
  end

  @testset "error_contract" begin
    dist = sqrt(
      abs(1 + (inner(phi, phi) - 2 * real(inner(phi', K, psi))) / inner(K, psi, K, psi))
    )
    @test dist ≈ error_contract(phi, K, psi)
  end

  @testset "contract" begin
    @test maxlinkdim(K) == 1
    psi_out = contract(K, psi; maxdim=1)
    @test inner(phi', psi_out) ≈ inner(phi', K, psi)
    @test_throws MethodError contract(K, psi; method="fakemethod")
  end

  # TODO: implement add for QN MPOs and add this test back
  #@testset "add(::MPO, ::MPO)" begin
  #  shsites = siteinds("S=1/2", N)
  #  M = add(K, L)
  #  @test length(M) == N
  #  k_psi = contract(K, psi, maxdim=1)
  #  l_psi = contract(L, psi, maxdim=1)
  #  @test inner(psi', k_psi + l_psi) ≈ ⋅(psi', M, psi) atol=5e-3
  #  @test inner(psi', sum([k_psi, l_psi])) ≈ dot(psi', M, psi) atol=5e-3
  #  for dim in 2:4
  #    shsites = siteinds("S=1/2",N)
  #    K = basicRandomMPO(N, shsites; dim=dim)
  #    L = basicRandomMPO(N, shsites; dim=dim)
  #    M = K + L
  #    @test length(M) == N
  #    psi = randomMPS(shsites)
  #    k_psi = contract(K, psi)
  #    l_psi = contract(L, psi)
  #    @test inner(psi, k_psi + l_psi) ≈ dot(psi, M, psi) atol=5e-3
  #    @test inner(psi, sum([k_psi, l_psi])) ≈ inner(psi, M, psi) atol=5e-3
  #    psi = randomMPS(shsites)
  #    M = add(K, L; cutoff=1E-9)
  #    k_psi = contract(K, psi)
  #    l_psi = contract(L, psi)
  #    @test inner(psi, k_psi + l_psi) ≈ inner(psi, M, psi) atol=5e-3
  #  end
  #end

  @testset "contract(::MPO, ::MPO)" begin
    @test maxlinkdim(K) == 1
    @test maxlinkdim(L) == 1
    KL = contract(prime(K), L; maxdim=1)
    Lpsi = contract(L, psi; maxdim=1)
    psi_kl_out = contract(prime(K), Lpsi; maxdim=1)
    @test inner(psi'', KL, psi) ≈ inner(psi'', psi_kl_out) atol = 5e-3
  end

  @testset "contract(::MPO, ::MPO) without truncation" begin
    s = siteinds("Electron", 10; conserve_qns=true)
    j1, j2 = 2, 4
    Cdagup = op_mpo(s, "Cdagup", j1)
    Cdagdn = op_mpo(s, "Cdagdn", j2)
    Cdagmpo = apply(Cdagup, Cdagdn; alg="naive", truncate=false)
    @test norm(Cdagmpo) ≈ 2^length(s) / 2
    for j in 1:length(s)
      if (j == j1) || (j == j2)
        @test norm(Cdagmpo[j]) ≈ √2
      else
        @test norm(Cdagmpo[j]) ≈ 2
      end
    end
  end

  @testset "*(::MPO, ::MPO)" begin
    @test maxlinkdim(K) == 1
    @test maxlinkdim(L) == 1
    KL = *(prime(K), L; maxdim=1)
    psi_kl_out = *(prime(K), *(L, psi; maxdim=1); maxdim=1)
    @test ⋅(psi'', KL, psi) ≈ dot(psi'', psi_kl_out) atol = 5e-3
  end

  sites = siteinds("S=1/2", N)
  O = MPO(sites, "Sz")
  @test length(O) == N # just make sure this works

  @test_throws ArgumentError randomMPO(sites, 2)
  @test isnothing(linkind(MPO(fill(ITensor(), N), 0, N + 1), 1))
end

@testset "splitblocks" begin
  N = 4
  sites = siteinds("S=1", N; conserve_qns=true)
  ampo = OpSum()
  for j in 1:(N - 1)
    ampo .+= 0.5, "S+", j, "S-", j + 1
    ampo .+= 0.5, "S-", j, "S+", j + 1
    ampo .+= "Sz", j, "Sz", j + 1
  end
  H = MPO(ampo, sites; splitblocks=false)

  # Split the tensors to make them more sparse
  # Drops zero blocks by default
  H̃ = splitblocks(linkinds, H)

  H̃2 = MPO(ampo, sites; splitblocks=true)

  # Defaults to true
  H̃3 = MPO(ampo, sites)

  @test prod(H) ≈ prod(H̃)
  @test prod(H) ≈ prod(H̃2)
  @test prod(H) ≈ prod(H̃3)

  @test nnz(H[1]) == 9
  @test nnz(H[2]) == 18
  @test nnz(H[3]) == 18
  @test nnz(H[4]) == 9

  @test nnzblocks(H[1]) == 9
  @test nnzblocks(H[2]) == 18
  @test nnzblocks(H[3]) == 18
  @test nnzblocks(H[4]) == 9

  @test nnz(H̃[1]) == nnzblocks(H̃[1]) == count(≠(0), H[1]) == count(≠(0), H̃[1]) == 9
  @test nnz(H̃[2]) == nnzblocks(H̃[2]) == count(≠(0), H[2]) == count(≠(0), H̃[2]) == 18
  @test nnz(H̃[3]) == nnzblocks(H̃[3]) == count(≠(0), H[3]) == count(≠(0), H̃[3]) == 18
  @test nnz(H̃[4]) == nnzblocks(H̃[4]) == count(≠(0), H[4]) == count(≠(0), H̃[4]) == 9

  @test nnz(H̃2[1]) == nnzblocks(H̃2[1]) == count(≠(0), H[1]) == count(≠(0), H̃2[1]) == 9
  @test nnz(H̃2[2]) == nnzblocks(H̃2[2]) == count(≠(0), H[2]) == count(≠(0), H̃2[2]) == 18
  @test nnz(H̃2[3]) == nnzblocks(H̃2[3]) == count(≠(0), H[3]) == count(≠(0), H̃2[3]) == 18
  @test nnz(H̃2[4]) == nnzblocks(H̃2[4]) == count(≠(0), H[4]) == count(≠(0), H̃2[4]) == 9

  @test nnz(H̃3[1]) == nnzblocks(H̃3[1]) == count(≠(0), H[1]) == count(≠(0), H̃3[1]) == 9
  @test nnz(H̃3[2]) == nnzblocks(H̃3[2]) == count(≠(0), H[2]) == count(≠(0), H̃3[2]) == 18
  @test nnz(H̃3[3]) == nnzblocks(H̃3[3]) == count(≠(0), H[3]) == count(≠(0), H̃3[3]) == 18
  @test nnz(H̃3[4]) == nnzblocks(H̃3[4]) == count(≠(0), H[4]) == count(≠(0), H̃3[4]) == 9
end

@testset "MPO operations with one or two sites" begin
  for N in 1:4, conserve_szparity in (true, false)
    s = siteinds("S=1/2", N; conserve_szparity=conserve_szparity)
    a = OpSum()
    h = 0.5
    for j in 1:(N - 1)
      a .+= -1, "Sx", j, "Sx", j + 1
    end
    for j in 1:N
      a .+= h, "Sz", j
    end
    H = MPO(a, s)
    if conserve_szparity
      ψ = randomMPS(s, n -> isodd(n) ? "↑" : "↓")
    else
      ψ = randomMPS(s)
    end

    # MPO * MPS
    Hψ = H * ψ
    @test prod(Hψ) ≈ prod(H) * prod(ψ)

    # MPO * MPO
    H² = H' * H
    @test prod(H²) ≈ prod(H') * prod(H)

    # DMRG
    sweeps = Sweeps(3)
    maxdim!(sweeps, 10)
    if N == 1
      @test_throws ErrorException dmrg(
        H, ψ, sweeps; eigsolve_maxiter=10, eigsolve_krylovdim=10, outputlevel=0
      )
    else
      e, ψgs = dmrg(H, ψ, sweeps; eigsolve_maxiter=10, eigsolve_krylovdim=10, outputlevel=0)
      @test prod(H) * prod(ψgs) ≈ e * prod(ψgs)'
      D, V = eigen(prod(H); ishermitian=true)
      if hasqns(ψ)
        fluxψ = flux(ψ)
        d = commonind(D, V)
        b = ITensors.findfirstblock(indblock -> ITensors.qn(indblock) == fluxψ, d)
        @test e ≈ minimum(storage(D[Block(b, b)]))
      else
        @test e ≈ minimum(storage(D))
      end
    end
  end
end

#
#  Build up Hamiltonians with non trival QN spaces in the link indices and further neighbour interactions.
#
function make_Heisenberg_AutoMPO(sites, NNN::Int64; J::Float64=1.0, kwargs...)::MPO
  N = length(sites)
  @assert N >= NNN
  ampo = OpSum()
  for dj in 1:NNN
    f = J / dj
    for j in 1:(N - dj)
      add!(ampo, f, "Sz", j, "Sz", j + dj)
      add!(ampo, f * 0.5, "S+", j, "S-", j + dj)
      add!(ampo, f * 0.5, "S-", j, "S+", j + dj)
    end
  end
  return MPO(ampo, sites; kwargs...)
end

function make_Hubbard_AutoMPO(
  sites, NNN::Int64; U::Float64=1.0, t::Float64=1.0, V::Float64=0.5, kwargs...
)::MPO
  N = length(sites)
  @assert(N >= NNN)
  os = OpSum()
  for i in 1:N
    os += (U, "Nupdn", i)
  end
  for dn in 1:NNN
    tj, Vj = t / dn, V / dn
    for n in 1:(N - dn)
      os += -tj, "Cdagup", n, "Cup", n + dn
      os += -tj, "Cdagup", n + dn, "Cup", n
      os += -tj, "Cdagdn", n, "Cdn", n + dn
      os += -tj, "Cdagdn", n + dn, "Cdn", n
      os += Vj, "Ntot", n, "Ntot", n + dn
    end
  end
  return MPO(os, sites; kwargs...)
end

test_combos = [(make_Heisenberg_AutoMPO, "S=1/2"), (make_Hubbard_AutoMPO, "Electron")]

@testset "QR/QL MPO tensors with complex block structures, H=$(test_combo[1])" for test_combo in
                                                                                   test_combos
  N, NNN = 10, 7 #10 lattice site, up 7th neight interactions
  sites = siteinds(test_combo[2], N; conserve_qns=true)
  H = test_combo[1](sites, NNN)
  for n in 1:(N - 1)
    W = H[n]
    @test flux(W) == QN("Sz", 0)
    ilr = filterinds(W; tags="l=$n")[1]
    ilq = noncommoninds(W, ilr)
    Q, R, q = qr(W, ilq)
    @test flux(Q) == QN("Sz", 0) #qr should move all flux on W (0 in this case) onto R
    @test flux(R) == QN("Sz", 0) #this effectively removes all flux between Q and R in thie case.
    @test W ≈ Q * R atol = 1e-13
    # blocksparse - diag is not supported so we must convert Q*Q_dagger to dense.
    # Also fails with error in permutedims so below we use norm(a-b)≈ 0.0 instead.
    # @test dense(Q*dag(prime(Q, q))) ≈ δ(Float64, q, q') atol = 1e-13
    @test norm(dense(Q * dag(prime(Q, q))) - δ(Float64, q, q')) ≈ 0.0 atol = 1e-13

    R, Q, q = ITensors.rq(W, ilr)
    @test flux(Q) == QN("Sz", 0)
    @test flux(R) == QN("Sz", 0)
    @test W ≈ Q * R atol = 1e-13
    @test norm(dense(Q * dag(prime(Q, q))) - δ(Float64, q, q')) ≈ 0.0 atol = 1e-13

    Q, L, q = ITensors.ql(W, ilq)
    @test flux(Q) == QN("Sz", 0)
    @test flux(L) == QN("Sz", 0)
    @test W ≈ Q * L atol = 1e-13
    @test norm(dense(Q * dag(prime(Q, q))) - δ(Float64, q, q')) ≈ 0.0 atol = 1e-13

    L, Q, q = ITensors.lq(W, ilr)
    @test flux(Q) == QN("Sz", 0)
    @test flux(L) == QN("Sz", 0)
    @test W ≈ Q * L atol = 1e-13
    @test norm(dense(Q * dag(prime(Q, q))) - δ(Float64, q, q')) ≈ 0.0 atol = 1e-13
  end
end
nothing
