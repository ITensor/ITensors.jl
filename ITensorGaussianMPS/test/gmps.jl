using ITensorGaussianMPS
using ITensors
using LinearAlgebra
using Test

@testset "Basic" begin
  # Test Givens rotations
  v = randn(6)
  g, r = ITensorGaussianMPS.givens_rotations(v)
  @test g * v ≈ r * [n == 1 ? 1 : 0 for n in 1:length(v)]
  v = randn(6)
  gr, r = ITensorGaussianMPS.givens_rotations_real(v)
  @test gr * v ≈ r * [n == 1 ? 1 : 0 for n in 1:length(v)]
end

@testset "Fermion" begin
  N = 10
  Nf = N ÷ 2

  # Hopping
  t = 1.0

  # Hopping Hamiltonian
  h = Hermitian(diagm(1 => fill(-t, N - 1), -1 => fill(-t, N - 1)))
  e, u = eigen(h)

  @test h * u ≈ u * Diagonal(e)

  E = sum(e[1:Nf])

  # Get the Slater determinant
  Φ = u[:, 1:Nf]
  @test h * Φ ≈ Φ * Diagonal(e[1:Nf])

  # Diagonalize the correlation matrix as a
  # Gaussian MPS (GMPS)
  n, gmps = slater_determinant_to_gmps(Φ; maxblocksize=4)

  ns = round.(Int, n)
  @test sum(ns) == Nf

  Λ = conj(Φ) * transpose(Φ)
  @test gmps * Λ * gmps' ≈ Diagonal(ns) rtol = 1e-2
  @test gmps' * Diagonal(ns) * gmps ≈ Λ rtol = 1e-2

  # Form the MPS
  s = siteinds("Fermion", N; conserve_qns=true)
  ψ = slater_determinant_to_mps(s, Φ; blocksize=4)

  os = OpSum()
  for i in 1:N, j in 1:N
    if h[i, j] ≠ 0
      os .+= h[i, j], "Cdag", i, "C", j
    end
  end
  H = MPO(os, s)

  @test inner(ψ', H, ψ) ≈ E rtol = 1e-5

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

@testset "Fermion (complex)" begin
  N = 10
  Nf = N ÷ 2

  # Hopping
  θ = π / 8
  t = exp(im * θ)

  # Hopping Hamiltonian
  h = Hermitian(diagm(1 => fill(-t, N - 1), -1 => fill(-conj(t), N - 1)))
  e, u = eigen(h)

  @test h * u ≈ u * Diagonal(e)

  E = sum(e[1:Nf])

  # Get the Slater determinant
  Φ = u[:, 1:Nf]
  @test h * Φ ≈ Φ * Diagonal(e[1:Nf])

  # Diagonalize the correlation matrix as a
  # Gaussian MPS (GMPS)
  n, gmps = slater_determinant_to_gmps(Φ; maxblocksize=4)

  ns = round.(Int, n)
  @test sum(ns) == Nf

  Λ = conj(Φ) * transpose(Φ)
  @test gmps * Λ * gmps' ≈ Diagonal(ns) rtol = 1e-2
  @test gmps' * Diagonal(ns) * gmps ≈ Λ rtol = 1e-2

  # Form the MPS
  s = siteinds("Fermion", N; conserve_qns=true)
  ψ = slater_determinant_to_mps(s, Φ; blocksize=4)

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

@testset "Fermion (BCS real)" begin
  N = 10
  Nf = N ÷ 2
  t = 1.2
  Delta = 0.5
  os_h = OpSum()
  for n in 1:(N - 1)
    os_h .+= -t, "Cdag", n, "C", n + 1
    os_h .+= -t, "Cdag", n + 1, "C", n
  end

  os_p = OpSum()
  for n in 1:(N - 1)
    os_p .+= Delta / 2.0, "Cdag", n, "Cdag", n + 1
    os_p .+= -Delta / 2.0, "Cdag", n + 1, "Cdag", n
    os_p .+= -Delta / 2.0, "C", n, "C", n + 1
    os_p .+= Delta / 2.0, "C", n + 1, "C", n
  end

  h, hb = ITensorGaussianMPS.pairing_hamiltonian(os_h, os_p)

  e, u = eigen(Hermitian(h))
  #H_D, U_D = Fu.Diag_h(Hermitian(hb))
  #get correlation matrix
  E = sum(e[1:N])
  #@show E
  # Get the Slater determinant
  Φ = u[:, 1:N]
  @test h * Φ ≈ Φ * Diagonal(e[1:N])
  c = conj(Φ) * transpose(Φ)
  #c2=Fu.GS_gamma(LinearAlgebra.Diagonal(H_D),U_D)
  #println("The energy of the ground state is: ", Fu.Energy(c2,(H_D,U_D)))
  tau = 1.0
  n, gmps = correlation_matrix_to_gmps(Real.(c); maxblocksize=8, is_bcs=true)
  ns = round.(Int, n)
  @test sum(ns) == N

  Λ = c
  @test gmps * Λ * gmps' ≈ Diagonal(ns) rtol = 1e-2
  @test gmps' * Diagonal(ns) * gmps ≈ Λ rtol = 1e-2

  # Form the MPS
  s = siteinds("Fermion", N; conserve_qns=false)
  psi = correlation_matrix_to_mps(s, c; eigval_cutoff=1e-10, maxblocksize=10)

  # compare entries of the correlation matrix
  cdagc = correlation_matrix(psi, "Cdag", "C")
  cdagcdag = correlation_matrix(psi, "Cdag", "Cdag")
  ccdag = correlation_matrix(psi, "C", "Cdag")
  cc = correlation_matrix(psi, "C", "C")
  cblocked = ITensorGaussianMPS.reverse_interleave(c)

  @test all(abs.(cblocked[(N + 1):end, (N + 1):end] - cdagc[:, :]) .< 1e-6)
  @test all(abs.(cblocked[1:N, 1:N] - ccdag[:, :]) .< 1e-6)
  @test all(abs.(cblocked[1:N, (N + 1):end] - cc[:, :]) .< 1e-6)
  @test all(abs.(cblocked[(N + 1):end, 1:N] - cdagcdag[:, :]) .< 1e-6)
end

@testset "Fermion (BCS real - no pairing)" begin
  N = 10
  Nf = N ÷ 2
  t = 1.2
  Delta = 0.0
  os_h = OpSum()
  for n in 1:(N - 1)
    os_h .+= -t, "Cdag", n, "C", n + 1
    os_h .+= -t, "Cdag", n + 1, "C", n
  end

  os_p = OpSum()
  for n in 1:(N - 1)
    os_p .+= Delta / 2.0, "Cdag", n, "Cdag", n + 1
    os_p .+= -Delta / 2.0, "Cdag", n + 1, "Cdag", n
    os_p .+= -Delta / 2.0, "C", n, "C", n + 1
    os_p .+= Delta / 2.0, "C", n + 1, "C", n
  end

  h, hb = ITensorGaussianMPS.pairing_hamiltonian(os_h, os_p)

  e, u = eigen(Hermitian(h))
  E = sum(e[1:N])
  # Get the Slater determinant
  Φ = u[:, 1:N]
  @test h * Φ ≈ Φ * Diagonal(e[1:N])
  c = conj(Φ) * transpose(Φ)
  tau = 1.0
  n, gmps = correlation_matrix_to_gmps(Real.(c); maxblocksize=8, is_bcs=true)
  ns = round.(Int, n)
  @test sum(ns) == Nf
  ##These tests don't work because the returned gmps is not designed to work on the full BCS-like correlation matrix
  ## only on the number conserving part here
  #Λ = c
  #@test gmps * Λ * gmps' ≈ Diagonal(ns) rtol = 1e-2
  #@test gmps' * Diagonal(ns) * gmps ≈ Λ rtol = 1e-2

  # Form the MPS
  s = siteinds("Fermion", N; conserve_qns=false)
  psi = correlation_matrix_to_mps(s, Real.(c); eigval_cutoff=1e-10, maxblocksize=10)
  @test eltype(psi[1]) == eltype(Real.(c))
  # compare entries of the correlation matrix
  cdagc = correlation_matrix(psi, "Cdag", "C")
  cdagcdag = correlation_matrix(psi, "Cdag", "Cdag")  #zero
  ccdag = correlation_matrix(psi, "C", "Cdag")
  cc = correlation_matrix(psi, "C", "C")  #zero
  cblocked = ITensorGaussianMPS.reverse_interleave(c)
  @test all(abs.(cblocked[(N + 1):end, (N + 1):end] - cdagc[:, :]) .< 1e-6)
  @test all(abs.(cblocked[1:N, 1:N] - ccdag[:, :]) .< 1e-6)
  @test all(abs.(cblocked[1:N, (N + 1):end] - cc[:, :]) .< 1e-6)
  @test all(abs.(cblocked[(N + 1):end, 1:N] - cdagcdag[:, :]) .< 1e-6)
end

@testset "Fermion (BCS complex)" begin
  N = 10
  Nf = N ÷ 2

  # Hopping
  # θ = π / 8
  t = 1.2
  Delta = 0.5
  Delta2 = 0.7
  os_h = OpSum()
  for n in 1:(N - 1)
    os_h .+= -t, "Cdag", n, "C", n + 1
    os_h .+= -t, "Cdag", n + 1, "C", n
  end

  os_p = OpSum()
  for n in 1:(N - 1)
    os_p .+= Delta / 2.0, "Cdag", n, "Cdag", n + 1
    os_p .+= -Delta / 2.0, "Cdag", n + 1, "Cdag", n
    os_p .+= -Delta / 2.0, "C", n, "C", n + 1
    os_p .+= Delta / 2.0, "C", n + 1, "C", n
  end

  os_p2 = OpSum()
  for n in 1:(N - 1)
    os_p2 .+= Delta2 / 2.0, "Cdag", n, "Cdag", n + 1
    os_p2 .+= -Delta2 / 2.0, "Cdag", n + 1, "Cdag", n
    os_p2 .+= -Delta2 / 2.0, "C", n, "C", n + 1
    os_p2 .+= Delta2 / 2.0, "C", n + 1, "C", n
  end

  h, hb = ITensorGaussianMPS.pairing_hamiltonian(os_h, os_p)
  h2, hb2 = ITensorGaussianMPS.pairing_hamiltonian(os_h, os_p2)

  e, u = eigen(Hermitian(h))
  E = sum(e[1:N])
  Φ = u[:, 1:N]
  @test h * Φ ≈ Φ * Diagonal(e[1:N])
  c = conj(Φ) * transpose(Φ)
  tau = 1.0
  Ud = exp(-tau * 1im * h2) ##generate complex state by time-evolving with perturbed Hamiltonian

  c = Ud' * c * Ud
  n, gmps = correlation_matrix_to_gmps(c; maxblocksize=8, is_bcs=true)
  ns = round.(Int, n)
  @test sum(ns) == N

  Λ = c
  @test gmps * Λ * gmps' ≈ Diagonal(ns) rtol = 1e-2
  @test gmps' * Diagonal(ns) * gmps ≈ Λ rtol = 1e-2

  # Form the MPS
  s = siteinds("Fermion", N; conserve_qns=false)
  psi = correlation_matrix_to_mps(s, c; eigval_cutoff=1e-10, maxblocksize=10)

  # compare entries of the correlation matrix
  cdagc = correlation_matrix(psi, "Cdag", "C")
  cdagcdag = correlation_matrix(psi, "Cdag", "Cdag")
  ccdag = correlation_matrix(psi, "C", "Cdag")
  cc = correlation_matrix(psi, "C", "C")
  cblocked = ITensorGaussianMPS.reverse_interleave(c)
  @test all(abs.(cblocked[(N + 1):end, (N + 1):end] - cdagc[:, :]) .< 1e-6)
  @test all(abs.(cblocked[1:N, 1:N] - ccdag[:, :]) .< 1e-6)
  @test all(abs.(cblocked[1:N, (N + 1):end] - cc[:, :]) .< 1e-6)
  @test all(abs.(cblocked[(N + 1):end, 1:N] - cdagcdag[:, :]) .< 1e-6)
end
