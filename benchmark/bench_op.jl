module BenchOp

using BenchmarkTools
using ITensors
using LinearAlgebra

BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)
ITensors.enable_threaded_blocksparse(false)

suite = BenchmarkGroup()

let N = 4
  s = siteinds("S=1/2", N; conserve_qns=false)
  suite["op"] = @benchmarkable op("Sz", $s, 1)
end

let N = 4
  s = siteinds("S=1/2", N; conserve_qns=true)
  suite["op QN"] = @benchmarkable op("Sz", $s, 1)
end

end

BenchOp.suite
