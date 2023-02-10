module BenchOpSum

using BenchmarkTools
using ITensors
using LinearAlgebra

BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)
ITensors.enable_threaded_blocksparse(false)

suite = BenchmarkGroup()

let N = 30
  s = siteinds("S=1/2", N; conserve_qns=false)
  a = OpSum()
  for k in 1:N, l in 1:N, m in 1:N, n in 1:N
    a .+= "projDn", k, "projDn", l, "projDn", m, "projDn", n
  end

  # Precompile
  MPO(a, s)

  suite["Quartic Hamiltonian"] = @benchmarkable MPO($a, $s)
end

let N = 30
  s = siteinds("S=1/2", N; conserve_qns=true)
  a = OpSum()
  for k in 1:N, l in 1:N, m in 1:N, n in 1:N
    a .+= "projDn", k, "projDn", l, "projDn", m, "projDn", n
  end

  # Precompile
  MPO(a, s)

  suite["Quartic QN Hamiltonian"] = @benchmarkable MPO($a, $s)
end

end

BenchOpSum.suite
