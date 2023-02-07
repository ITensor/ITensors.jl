module BenchDMRG

using BenchmarkTools
using ITensors
using LinearAlgebra

BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)
ITensors.enable_threaded_blocksparse(false)

suite = BenchmarkGroup()

suite["1d_S=1_heisenberg"] = BenchmarkGroup()
for conserve_qns in (false, true)
  n = 100
  sites = siteinds("S=1", n)
  opsum = OpSum()
  for j in 1:(n - 1)
    opsum += "Sz", j, "Sz", j + 1
    opsum += 0.5, "S+", j, "S-", j + 1
    opsum += 0.5, "S-", j, "S+", j + 1
  end
  H = MPO(opsum, sites)
  state = [isodd(j) ? "↑" : "↓" for j in 1:n]
  psi = MPS(sites, state)

  # Sweeping parameters
  nsweeps = 5
  maxdim = [10, 20, 100, 100, 200]
  cutoff = 1e-10
  kwargs = (; nsweeps, cutoff, maxdim, outputlevel=0)

  # Precompile
  dmrg(H, psi; kwargs...)

  suite["1d_S=1_heisenberg"]["conserve_qns_$conserve_qns"] = @benchmarkable begin
    dmrg($H, $psi; $kwargs...)
  end
end

end

BenchDMRG.suite
