module BenchInplace

using BenchmarkTools
using ITensors

suite = BenchmarkGroup()

i = Index(100)
A = randomITensor(i', i)
B = randomITensor(i, i')

suite["axpy!"] = @benchmarkable axpy!(2, $A, $B)
end

BenchInplace.suite
