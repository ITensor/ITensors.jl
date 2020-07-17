module BenchIndexSet

using BenchmarkTools
using ITensors

suite = BenchmarkGroup()

f(i) = IndexSet(n -> i^(n-1), Order(10))
i = Index(2, "i")
is = f(i)

suite["function_constructor"] = @benchmarkable f($i)
suite["filter"] = filter(i -> plev(i) < 2, $is)
suite["filter_order"] = filter(Order(2), i -> plev(i) < 2, $is)
end

BenchIndexSet.suite
