using Test
using ITensors
using LinearAlgebra

@testset "Ops" begin
  o1 = Op("X", 1)
  I1 = Op(I, 1)
  y1 = Op("Y", 1)
  o2 = Op("Y", 2)
  o3 = Op("CX", 1, 2)
  o4 = Op("Ry", 4, (θ = π / 3,))

  @test 2o2 isa Ops.ScaledOp
  @test -o2 isa Ops.ScaledOp
  @test o1 * o2 isa Ops.ProdOp
  @test 2o1 * o2 isa Ops.ScaledProdOp
  @test o1 * o2 + o3 isa Ops.SumProdOp
  @test o1 * o2 + o1 * o3 isa Ops.SumProdOp
  @test o1 * o2 + 2o3 isa Ops.SumScaledProdOp
  @test o1 * o2 - o3 isa Ops.SumScaledProdOp
  @test 2o1 * o2 + 2o3 isa Ops.SumScaledProdOp
  @test 2o1 * o2 - 2o3 isa Ops.SumScaledProdOp

  N = 4
  s = siteinds("Qubit", N)
  t1 = ITensor(o1, s)
  @test hassameinds(t1, (s[1]', dag(s[1])))
  @test t1[1, 1] == 0
  @test t1[1, 2] == 1

  @test ITensor(2.3o1, s) ≈ 2.3 * t1
  @test ITensor(o1 + o1, s) ≈ 2t1
  @test ITensor(o1 + 2.3o1, s) ≈ 3.3t1

  @test ITensor(Op(I, 2), s) ≈ ITensor([1 0; 0 1], s[2]', dag(s[2]))

  c = o1 * o2 * o3
  cdag = c'
  @test c[1]' == cdag[3]
  @test c[2]' == cdag[2]
  @test c[3]' == cdag[1]

  x = randn(2, 2)
  tx = ITensor(Op(x, 3), s)
  @test tx[s[3]' => 1, s[3] => 2] == x[1, 2]

  @test ITensor(o1 * o1, s) ≈ ITensor(Op([1 0; 0 1], 1), s)
  @test ITensor(o1 * o1 * o1, s) ≈ ITensor(Op([0 1; 1 0], 1), s)
  @test ITensor(2o1 * o1, s) ≈ ITensor(Op([2 0; 0 2], 1), s)
  @test ITensor(o1 * y1, s) ≈ ITensor(Op([im 0; 0 -im], 1), s)
  @test ITensor(y1 * o1, s) ≈ ITensor(Op([-im 0; 0 im], 1), s)
  @test ITensor(2o1 * o1 + y1, s) ≈ ITensor(2 * [1 0; 0 1] + [0 -im; im 0], s[1]', dag(s[1]))

  @test y1'' == y1

  @test ITensor(y1', s) ≈ ITensor(Op([0 -im; im 0], 1), s)

  @test ITensor(exp(o1), s) ≈ ITensor(Op(exp([0 1; 1 0]), 1), s)
  @test ITensor(exp(2o1 * o1), s) ≈ ITensor(exp(2 * [1 0; 0 1]), s[1]', dag(s[1]))
  @test ITensor(exp(2o1 * o1 + y1), s) ≈ ITensor(exp(2 * [1 0; 0 1] + [0 -im; im 0]), s[1]', dag(s[1]))

  @test ITensor(I1, s) ≈ ITensor([1 0; 0 1], s[1]', dag(s[1]))

  H = Ops.OpSum() - Op("X", 1)
  @test H isa Ops.OpSum
  @test length(H) == 1
  @test coefficient(H[1]) == -1

  # MPO conversion
  H = Ops.OpSum()
  H -= 2.3, "X", 1, "X", 2
  H += 1.2, "Z", 1
  H += 1.3, "Z", 2, (θ = π / 3,)
  @test H isa Ops.OpSum
  @test length(H) == 3
  @test coefficient(H[1]) == -2.3
  @test length(H[1]) == 2
  @test Ops.sites(H[1]) == [1, 2]
  @test coefficient(H[2]) == 1.2
  @test length(H[2]) == 1
  @test Ops.sites(H[2]) == [1]
  @test coefficient(H[3]) == 1.3
  @test length(H[3]) == 1
  @test Ops.sites(H[3]) == [2]
  @test Ops.params(H[3]) == (θ = π / 3,)
end
