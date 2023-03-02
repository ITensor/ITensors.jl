module BenchInplace

using BenchmarkTools
using ITensors
using LinearAlgebra

BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)
ITensors.enable_threaded_blocksparse(false)

suite = BenchmarkGroup()

i = Index(100)
A = randomITensor(i', i)
B = randomITensor(i, i')

suite["axpy!"] = @benchmarkable axpy!(2, $A, $B)
end

BenchInplace.suite
