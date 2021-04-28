module BenchContract

using BenchmarkTools
using ITensors

suite = BenchmarkGroup()

for d in 20:20:100
  i = Index(d)
  A = randomITensor(i, i')
  B = randomITensor(i', i'')
  C = randomITensor(i, i'')

  suite["matmul_$d"] = @benchmarkable $A * $B
  suite["matmul_inplace_$d"] = @benchmarkable $C .= $A .* $B
end

let
  s1 = Index(2, "s1,Site")
  s2 = Index(2, "s2,Site")
  h1 = Index(10, "h1,Link,H")
  h2 = Index(10, "h2,Link,H")
  h3 = Index(10, "h3,Link,H")
  a1 = Index(100, "a1,Link")
  a3 = Index(100, "a3,Link")
  phi = randomITensor(a1, s1, s2, a3)
  H1 = randomITensor(h1, s1', s1, h2)
  H2 = randomITensor(h2, s2', s2, h3)
  L = randomITensor(h1, a1', a1)
  R = randomITensor(h3, a3', a3)

  suite["heff_2site"] = @benchmarkable $phi * $L * $H1 * $H2 * $R
end

end

BenchContract.suite
