using ITensorGaussianMPS
using ITensors
using LinearAlgebra
using Test

@testset "Basic" begin
  # Test Givens rotations
  v = randn(6)
  g, r = ITensorGaussianMPS.givens_rotations(v)
  @test g * v ≈ r * [n == 1 ? 1 : 0 for n in 1:length(v)]
end

@testset "Hamiltonians" begin
  N = 8
  t = -0.8 ###nearest neighbor hopping
  mu = 0.0 ###on-site chemical potential
  pairing = 1.2
  os = OpSum()
  for i in 1:N
    if 1 < i < N
      js = [i - 1, i + 1]
    elseif i == 1
      js = [i + 1]
    else
      js = [i - 1]
    end
    for j in js
      os .+= t, "Cdag", i, "C", j
    end
  end
  h_hop = ITensorGaussianMPS.hopping_hamiltonian(os)
  for i in 1:N
    if 1 < i < N
      js = [i - 1, i + 1]
    elseif i == 1
      js = [i + 1]
    else
      js = [i - 1]
    end
    for j in js
      os .+= pairing / 2.0, "Cdag", i, "Cdag", j
      os .+= -conj(pairing / 2.0), "C", i, "C", j
    end
  end

  h_hopandpair = ITensorGaussianMPS.quadratic_hamiltonian(os)
  h_hopandpair_spinful = ITensorGaussianMPS.quadratic_hamiltonian(os, os)

  @test all(
    abs.(
      (
        2 .* ITensorGaussianMPS.reverse_interleave(Matrix(h_hopandpair))[
          (N + 1):end, (N + 1):end
        ]
      ) - h_hop
    ) .< eps(Float32),
  )
end

@testset "Fermion (real and complex)" begin
  N = 10
  Nf = N ÷ 2

  # Hopping
  θs = [0.0, π / 8]
  for θ in θs
    t = exp(im * θ)

    # Hopping Hamiltonian
    h = Hermitian(diagm(1 => fill(-t, N - 1), -1 => fill(-conj(t), N - 1)))
    if θ == 0.0
      h = real(h)
    end
    e, u = eigen(h)

    @test h * u ≈ u * Diagonal(e)

    E = sum(e[1:Nf])

    # Get the Slater determinant
    Φ = u[:, 1:Nf]
    @test h * Φ ≈ Φ * Diagonal(e[1:Nf])

    # Diagonalize the correlation matrix as a
    # Gaussian MPS (GMPS)
    n, gmps = slater_determinant_to_gmps(Φ, N; maxblocksize=4)
    ns = round.(Int, n)
    @test sum(ns) == Nf

    Λ = conj(Φ) * transpose(Φ)
    @test gmps * Λ * gmps' ≈ Diagonal(ns) rtol = 1e-2
    @test gmps' * Diagonal(ns) * gmps ≈ Λ rtol = 1e-2

    # Form the MPS
    s = siteinds("Fermion", N; conserve_qns=true)
    ψ = slater_determinant_to_mps(s, Φ; maxblocksize=4)
    os = OpSum()
    for i in 1:N, j in 1:N
      if h[i, j] ≠ 0
        os .+= h[i, j], "Cdag", i, "C", j
      end
    end
    H = MPO(os, s)

    @test inner(ψ', H, ψ) ≈ E rtol = 1e-5
    @test inner(ψ', H, ψ) / norm(ψ) ≈ E rtol = 1e-5

    # Compare to DMRG
    sweeps = Sweeps(10)
    setmaxdim!(sweeps, 10, 20, 40, 60)
    setcutoff!(sweeps, 1E-12)
    energy, ψ̃ = dmrg(H, productMPS(s, n -> n ≤ Nf ? "1" : "0"), sweeps; outputlevel=0)

    # Create an mps
    @test abs(inner(ψ, ψ̃)) ≈ 1 rtol = 1e-5
    @test inner(ψ̃', H, ψ̃) ≈ inner(ψ', H, ψ) rtol = 1e-5
    @test E ≈ energy
  end
end

@testset "Fermion BCS (real,real - no pairing, complex)" begin
  N = 12
  Nf = N ÷ 2
  ts = [1.0, exp(im * pi / 3.0), 1.0]
  Deltas = [1.0, 1.0, 0.0]
  for (Delta, t) in zip(Deltas, ts)
    t = isreal(t) ? real(t) : t
    os_h = OpSum()
    for n in 1:(N - 1)
      os_h .+= -t, "Cdag", n, "C", n + 1
      os_h .+= -t', "Cdag", n + 1, "C", n
    end
    os_p = OpSum()
    for n in 1:(N - 1)
      os_p .+= Delta / 2.0, "Cdag", n, "Cdag", n + 1
      os_p .+= -Delta / 2.0, "Cdag", n + 1, "Cdag", n
      os_p .+= -Delta / 2.0, "C", n, "C", n + 1
      os_p .+= Delta / 2.0, "C", n + 1, "C", n
    end

    h = ITensorGaussianMPS.quadratic_hamiltonian(os_h + os_p)
    @assert ishermitian(h)
    ElT = eltype(h)
    e, u = ITensorGaussianMPS.eigen_gaussian(h)
    E = sum(e[1:(N)])
    Φ = (u[:, 1:N])
    @test h * Φ ≈ Φ * Diagonal(e[1:N])
    c = conj(Φ) * transpose(Φ)
    c2 = ITensorGaussianMPS.get_gaussian_GS_corr(h)
    @test norm(c - c2) <= sqrt(eps(real(eltype(h))))
    if ElT <: Real
      @assert norm(imag.(c)) <= sqrt(eps())
      c = real.(c)
    end
    n, gmps = correlation_matrix_to_gmps(ElT.(c), N; eigval_cutoff=1e-10, maxblocksize=14)
    ns = round.(Int, n)
    @test sum(ns) == N

    Λ = ITensorGaussianMPS.ConservesNfParity(c)
    @test gmps * Λ.data * gmps' ≈ Diagonal(ns) rtol = 1e-2
    @test gmps' * Diagonal(ns) * gmps ≈ Λ.data rtol = 1e-2

    # Form the MPS
    s = siteinds("Fermion", N; conserve_qns=false)
    h_mpo = MPO(os_h + os_p, s)
    psi = correlation_matrix_to_mps(
      s, ElT.(c); eigval_cutoff=1e-10, maxblocksize=14, cutoff=1e-11
    )
    @test eltype(psi[1]) <: ElT
    sweeps = Sweeps(5)
    _maxlinkdim = 60
    _cutoff = 1e-10
    setmaxdim!(sweeps, 10, 20, 40, _maxlinkdim)
    setcutoff!(sweeps, _cutoff)
    E_dmrg, psidmrg = dmrg(h_mpo, psi, sweeps; outputlevel=0)
    E_ni_mpo = inner(psi', h_mpo, psi)
    @test E_dmrg ≈ E_ni_mpo rtol = 1e-4
    @test inner(psidmrg, psi) ≈ 1 rtol = 1e-4

    # compare entries of the correlation matrix
    cdagc = correlation_matrix(psi, "Cdag", "C")
    cdagcdag = correlation_matrix(psi, "Cdag", "Cdag")
    ccdag = correlation_matrix(psi, "C", "Cdag")
    cc = correlation_matrix(psi, "C", "C")
    cblocked = ITensorGaussianMPS.reverse_interleave(c)
    tol = 1e-5
    @test all(abs.(cblocked[(N + 1):end, (N + 1):end] - cdagc[:, :]) .< tol)
    @test all(abs.(cblocked[1:N, 1:N] - ccdag[:, :]) .< tol)
    @test all(abs.(cblocked[1:N, (N + 1):end] - cc[:, :]) .< tol)
    @test all(abs.(cblocked[(N + 1):end, 1:N] - cdagcdag[:, :]) .< tol)
    @show "Completed test for: ", Delta, t
  end
end

@testset "Bad Terms" begin
  @testset "Bad single" begin
    os = OpSum()
    os += -1.0, "Nupdn", 1
    @test_throws Any h_hop = ITensorGaussianMPS.hopping_hamilontian(os)
  end
  @testset "Bad quadratic" begin
    os = OpSum()
    os += -1.0, "Ntot", 1, "Ntot", 2
    @test_throws Any h_hop = ITensorGaussianMPS.hopping_hamilontian(os)
  end
end

@testset "Rewrite Hamiltonians" begin
  @testset "Spinless" begin
    os = OpSum()
    os += -1.0, "Cdag", 1, "C", 2
    os += -1.0, "Cdag", 2, "C", 1
    os += 2, "N", 1
    os += 3, "N", 2
    h_hop = ITensorGaussianMPS.hopping_hamiltonian(os)
    @test h_hop[1, 1] == 2
    @test h_hop[2, 2] == 3
  end
  @testset "Spin $o" for o in ("up", "dn")
    os = OpSum()
    os += -1.0, "Cdag$o", 1, "C$o", 2
    os += -1.0, "Cdag$o", 2, "C$o", 1
    os += 2, "N$o", 1
    os += 3, "N$o", 2
    h_hop = ITensorGaussianMPS.hopping_hamiltonian(os)
    @test h_hop[1, 1] == 2
    @test h_hop[2, 2] == 3
  end
  @testset "Spin $o" for o in ("↑", "↓")
    os = OpSum()
    os += -1.0, "c†$o", 1, "c$o", 2
    os += -1.0, "c†$o", 2, "c$o", 1
    os += 2, "n$o", 1
    os += 3, "n$o", 2
    h_hop = ITensorGaussianMPS.hopping_hamiltonian(os)
    @test h_hop[1, 1] == 2
    @test h_hop[2, 2] == 3
  end
end
