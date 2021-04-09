module BenchGetSetIndex

using BenchmarkTools
using ITensors

suite = BenchmarkGroup()

i = Index(100);
A = randomITensor(i', i)

suite["setindex!"] = @benchmarkable $A[1, 1] = 1.0
suite["setindex! indval"] = @benchmarkable $A[$(i') => 1, $i => 1] = 1.0
suite["getindex"] = @benchmarkable $A[1, 1]
suite["getindex indval"] = @benchmarkable $A[$(i') => 1, $i => 1]
end

BenchGetSetIndex.suite
