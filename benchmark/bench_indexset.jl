module BenchIndexSet

using BenchmarkTools
using ITensors
using LinearAlgebra

BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)
ITensors.enable_threaded_blocksparse(false)

suite = BenchmarkGroup()

i = Index(2)
is = IndexSet(n -> i^(n - 1), Order(10))
is_tuple = ntuple(n -> i^(n - 1), Val(10))

suite["constructor"] = BenchmarkGroup()

suite["constructor"]["function"] = @benchmarkable IndexSet($(n -> i^(n - 1)), $(Order(6)))
suite["constructor"]["function, tuple"] = @benchmarkable ntuple(
  $(n -> i^(n - 1)), $(Val(6))
)

suite["filter"] = BenchmarkGroup()

suite["filter"]["kwargs"] = @benchmarkable filter($is; plev=3)
suite["filter"]["function"] = @benchmarkable filter($(i -> plev(i) < 2), $is)
suite["filter"]["function, tuple"] = @benchmarkable filter($(i -> plev(i) < 2), $is_tuple)

i, j, k, l = Index.(2, ("i", "j", "k", "l"))

Iij = (i, j)
Ijl = (j, l)
Ikl = (k, l)
Iijk = (i, j, k)

suite["set_functions"] = BenchmarkGroup()

suite["set_functions"]["uniqueinds"] = BenchmarkGroup()
suite["set_functions"]["uniqueinds"]["nofilter2"] = @benchmarkable uniqueinds($Iijk, $Ikl)
suite["set_functions"]["uniqueinds"]["nofilter0"] = @benchmarkable uniqueinds($Iij, $Iijk)
suite["set_functions"]["uniqueinds"]["filter_tags"] = @benchmarkable uniqueinds(
  $Iijk, $Ikl; tags=$(ts"i")
)
suite["set_functions"]["uniqueinds"]["filter_not_tags"] = @benchmarkable uniqueinds(
  $Iijk, $Ikl; tags=$(not("i"))
)
suite["set_functions"]["uniqueinds"]["3_inputs"] = @benchmarkable uniqueinds(
  $Iijk, $Ijl, $Ikl
)

A = randomITensor(i'', i', i)

suite["set_functions"]["prime, ITensor"] = @benchmarkable prime($A)
suite["set_functions"]["uniqueinds, ITensor"] = @benchmarkable uniqueinds($(A'), $A)
suite["set_functions"]["commoninds, ITensor"] = @benchmarkable commoninds($(A'), $A)
suite["set_functions"]["unioninds, ITensor"] = @benchmarkable unioninds($(A'), $A)
suite["set_functions"]["noncommoninds, ITensor"] = @benchmarkable noncommoninds($(A'), $A)

end

BenchIndexSet.suite
