module BenchGetSetIndex

using BenchmarkTools
using ITensors
using LinearAlgebra

BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)
ITensors.enable_threaded_blocksparse(false)

suite = BenchmarkGroup()

i = Index(100)
A = randomITensor(i', i)

suite["setindex!"] = @benchmarkable $A[1, 1] = 1.0
suite["setindex! end"] = @benchmarkable $A[1, end] = 1.0
suite["setindex! indval"] = @benchmarkable $A[$(i') => 1, $i => 1] = 1.0
suite["setindex! indval end"] = @benchmarkable $A[$(i') => 1, $i => end] = 1.0
suite["getindex"] = @benchmarkable $A[1, 1]
suite["getindex end"] = @benchmarkable $A[1, end]
suite["getindex indval"] = @benchmarkable $A[$(i') => 1, $i => 1]
suite["getindex indval end"] = @benchmarkable $A[$(i') => 1, $i => end]
end

BenchGetSetIndex.suite
