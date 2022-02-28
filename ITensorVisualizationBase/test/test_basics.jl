using ITensors
using ITensorVisualizationBase
using Test

@testset "Basic tests without any backend" begin
  N = 10
  s(n) = Index([QN("Sz", 0) => 1, QN("Sz", 1) => 1]; tags="S=1/2,Site,n=$n")
  l(n) = Index([QN("Sz", 0) => 10, QN("Sz", 1) => 10]; tags="Link,l=$n")
  h(n) = Index([QN("Sz", 0) => 5, QN("Sz", 1) => 5]; tags="ham,Link,l=$n")
  s⃗ = [s(n) for n in 1:N]
  l⃗ = [l(n) for n in 1:(N - 1)]
  h⃗ = [h(n) for n in 1:(N - 1)]

  # Add some more indices between two of the tensors
  x = Index([QN("Sz", 0) => 2]; tags="X")
  y = Index([QN("Sz", 0) => 2]; tags="Y")

  n = 2
  ψn1n2 = randomITensor(l⃗[n - 1], s⃗[n], s⃗[n + 1], l⃗[n + 1], dag(x), dag(y))
  hn1 = randomITensor(dag(h⃗[n - 1]), s⃗[n]', dag(s⃗[n]), h⃗[n], x, y)
  hn2 = randomITensor(dag(h⃗[n]), s⃗[n + 1]', dag(s⃗[n + 1]), h⃗[n + 1])
  ELn0 = randomITensor(l⃗[n - 1]', h⃗[n - 1], dag(l⃗[n - 1]))
  ERn2 = randomITensor(l⃗[n + 1]', dag(h⃗[n + 1]), dag(l⃗[n + 1]))

  tn = [ELn0, ψn1n2, hn1, hn2, ERn2]

  R = @visualize ELn0 * ψn1n2 * hn1 * hn2 * ERn2
  R1 = @visualize ELn0 * ψn1n2 * hn1
  R2 = @visualize R1 * hn2 * ERn2 vertex_labels = ["T1", "T2", "T3"]
  tn2 = @visualize tn
  T = @visualize ELn0

  @test R ≈ ELn0 * ψn1n2 * hn1 * hn2 * ERn2
  @test R1 ≈ ELn0 * ψn1n2 * hn1
  @test R2 ≈ ELn0 * ψn1n2 * hn1 * hn2 * ERn2
  @test all(tn .== tn2)
  @test T == ELn0

  R = @visualize figR ELn0 * ψn1n2 * hn1 * hn2 * ERn2
  R1 = @visualize figR1 ELn0 * ψn1n2 * hn1
  R2 = @visualize figR2 R1 * hn2 * ERn2 vertex_labels = ["T1", "T2", "T3"]
  T = @visualize figT T

  fig_tn = @visualize_noeval tn

  @test isnothing(figR)
  @test isnothing(figR1)
  @test isnothing(figR2)
  @test isnothing(fig_tn)
  @test isnothing(figT)
end
