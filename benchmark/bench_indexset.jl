module BenchIndexSet

using BenchmarkTools
using ITensors

suite = BenchmarkGroup()

i = Index(2)
is = IndexSet(n -> i^(n-1), Order(10))

suite["constructor"] = BenchmarkGroup()

suite["constructor"]["function"] = @benchmarkable IndexSet($(n -> i^(n-1)), $(Order(6)))

suite["filter"] = BenchmarkGroup()

suite["filter"]["kwargs"] = @benchmarkable filter($(i -> plev(i) < 2), $is; plev = 3)
suite["filter"]["function"] = @benchmarkable filter($(i -> plev(i) < 2), $is)
suite["filter"]["order"] = @benchmarkable filter($(i -> plev(i)), $(Order(2)), $is)

i,j,k,l = Index.(2, ("i", "j", "k", "l"))

Iij = IndexSet(i, j)
Ijl = IndexSet(j, l)
Ikl = IndexSet(k, l)
Iijk = IndexSet(i, j, k)

suite["uniqueinds"] = BenchmarkGroup()

suite["uniqueinds"]["nofilter2"] = @benchmarkable uniqueinds($Iijk, $Ikl)
suite["uniqueinds"]["nofilter0"] = @benchmarkable uniqueinds($Iij, $Iijk)
suite["uniqueinds"]["filter_tags"] = @benchmarkable uniqueinds($Iijk, $Ikl; tags = $(ts"i"))
suite["uniqueinds"]["filter_not_tags"] = @benchmarkable uniqueinds($Iijk, $Ikl; tags = $(not("i")))
suite["uniqueinds"]["3_inputs"] = @benchmarkable uniqueinds($Iijk, $Ijl, $Ikl)
suite["uniqueinds"]["order2"] = @benchmarkable uniqueinds($(Order(2)), $Iijk, $Ikl)
suite["uniqueinds"]["order0"] = @benchmarkable uniqueinds($(Order(0)), $Iij, $Iijk)
suite["uniqueinds"]["order_filter_tags"] = @benchmarkable uniqueinds($(Order(1)), $Iijk, $Ikl; tags = $(ts"i"))
suite["uniqueinds"]["order_filter_not_tags"] = @benchmarkable uniqueinds($(Order(1)), $Iijk, $Ikl; tags = $(not("i")))
suite["uniqueinds"]["order_3_inputs"] = @benchmarkable uniqueinds($(Order(1)), $Iijk, $Ijl, $Ikl)

end

BenchIndexSet.suite
