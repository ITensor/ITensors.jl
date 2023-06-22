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
  # Gaussian MPS (GMPS) gates
  n, gmps = slater_determinant_to_gmera(Φ; maxblocksize=10)

  ns = round.(Int, n)
  @test sum(ns) == Nf

  Λ = conj(Φ) * transpose(Φ)
  @test gmps * Λ * gmps' ≈ Diagonal(ns) rtol = 1e-2
  @test gmps' * Diagonal(ns) * gmps ≈ Λ rtol = 1e-2

  # Form the MPS
  s = siteinds("Fermion", N; conserve_qns=true)
  ψ = ITensorGaussianMPS.slater_determinant_to_mera(s, Φ; maxblocksize=4)

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
  n, gmps = slater_determinant_to_gmera(Φ; maxblocksize=4)

  ns = round.(Int, n)
  @test sum(ns) == Nf

  Λ = conj(Φ) * transpose(Φ)
  @test gmps * Λ * gmps' ≈ Diagonal(ns) rtol = 1e-2
  @test gmps' * Diagonal(ns) * gmps ≈ Λ rtol = 1e-2

  # Form the MPS
  s = siteinds("Fermion", N; conserve_qns=true)
  ψ = ITensorGaussianMPS.slater_determinant_to_mera(s, Φ; maxblocksize=4)

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

# Build 1-d SSH model
function SSH1dModel(N::Int, t::Float64, vardelta::Float64)
  # N should be even
  s = siteinds("Fermion", N; conserve_qns=true)
  limit = div(N - 1, 2)
  t1 = -t * (1 + vardelta / 2)
  t2 = -t * (1 - vardelta / 2)
  os = OpSum()
  for n in 1:limit
    os .+= t1, "Cdag", 2 * n - 1, "C", 2 * n
    os .+= t1, "Cdag", 2 * n, "C", 2 * n - 1
    os .+= t2, "Cdag", 2 * n, "C", 2 * n + 1
    os .+= t2, "Cdag", 2 * n + 1, "C", 2 * n
  end
  if N % 2 == 0
    os .+= t1, "Cdag", N - 1, "C", N
    os .+= t1, "Cdag", N, "C", N - 1
  end
  h = hopping_hamiltonian(os)
  H = MPO(os, s)
  #display(t1)
  return (h, H, s)
end

@testset "Energy" begin
  N = 2^4
  Nf = div(N, 2)
  t = 1.0
  gapsize = 0
  vardelta = gapsize / 2
  h, H, s = SSH1dModel(N, t, vardelta)

  Φ = slater_determinant_matrix(h, Nf)
  E, V = eigen(h)
  sort(E)
  Eana = sum(E[1:Nf])

  Λ0 = Φ * Φ'
  @test Eana ≈ tr(h * Λ0) rtol = 1e-5
  # Diagonalize the correlation matrix as a
  # Gaussian MPS (GMPS) and GMERA
  ngmps, V1 = ITensorGaussianMPS.correlation_matrix_to_gmps(Λ0; eigval_cutoff=1e-8)
  nmera, V1 = ITensorGaussianMPS.correlation_matrix_to_gmera(Λ0; eigval_cutoff=1e-8)#,maxblocksize=6)
  @test sum(round.(Int, nmera)) == sum(round.(Int, ngmps))

  U = ITensorGaussianMPS.UmatFromGates(V1, N)
  Etest = ITensorGaussianMPS.EfromGates(h, U)

  @test Eana ≈ Etest rtol = 1e-5
end
