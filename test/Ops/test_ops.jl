using Test
using ITensors
using LinearAlgebra

using ITensors.Ops: α, ∏, ∑, expand

@testset "Ops" begin
  o1 = Op("X", 1)
  I1 = Op(I, 1)
  y1 = Op("Y", 1)
  o2 = Op("Y", 2)
  o3 = Op("CX", 1, 2)
  o4 = Op("Ry", 4, (θ=π / 3,))

  @test 2o2 isa α{Op}
  @test coefficient(2o2) == 2
  @test o2 / 2 isa α{Op}
  @test coefficient(o2 / 2) ≈ 0.5
  @test -o2 isa α{Op}
  @test 1o2 + o1 isa ∑{<:α{Op}}
  @test 1o2 + o1 isa ∑{α{Op,Int}}
  @test o1 * o2 isa ∏{Op}
  @test 2o1 * o2 isa α{∏{Op}}
  @test o1 * o2 + o3 isa ∑{∏{Op}}
  @test o1 * o2 + o1 * o3 isa ∑{∏{Op}}
  @test o1 * o2 + 2o3 isa ∑{<:α{∏{Op}}}
  @test o1 * o2 - o3 isa ∑{<:α{∏{Op}}}
  @test 2o1 * o2 + 2o3 isa ∑{<:α{∏{Op}}}
  @test 2o1 * o2 - 2o3 isa ∑{<:α{∏{Op}}}
  @test (2o1 * o2 - 2o3) / 3 isa ∑{<:α{∏{Op}}}

  o = (2o1 * o2 - 2o3) / 3
  @test coefficient(o[1]) ≈ 2 / 3
  @test coefficient(o[2]) ≈ -2 / 3

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
  @test ITensor(2o1 * o1 + y1, s) ≈
    ITensor(2 * [1 0; 0 1] + [0 -im; im 0], s[1]', dag(s[1]))

  @test y1'' == y1

  @test ITensor(y1', s) ≈ ITensor(Op([0 -im; im 0], 1), s)

  @test ITensor(exp(o1), s) ≈ ITensor(Op(exp([0 1; 1 0]), 1), s)
  @test ITensor(exp(2o1 * o1), s) ≈ ITensor(exp(2 * [1 0; 0 1]), s[1]', dag(s[1]))
  @test ITensor(exp(2o1 * o1 + y1), s) ≈
    ITensor(exp(2 * [1 0; 0 1] + [0 -im; im 0]), s[1]', dag(s[1]))

  @test ITensor(I1, s) ≈ ITensor([1 0; 0 1], s[1]', dag(s[1]))

  @test exp(Op("X", 1)) * Op("Y", 2) isa ∏{Any}
  @test ITensor(exp(Op("X", 1)) * Op("Y", 1), s) ≈
    product(exp(ITensor(Op("X", 1), s)), ITensor(Op("Y", 1), s))
  @test 2exp(Op("X", 1)) * Op("Y", 2) isa α{∏{Any}}

  H = Ops.OpSum() - Op("X", 1)
  @test H isa Ops.OpSum
  @test length(H) == 1
  @test coefficient(H[1]) == -1

  # MPO conversion
  H = Ops.OpSum()
  H -= 2.3, "X", 1, "X", 2
  H += 1.2, "Z", 1
  H += 1.3, "Z", 2, (θ=π / 3,)
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
  @test Ops.params(H[3]) == (θ=π / 3,)

  @test Ops.OpSum(("X", 1)) isa Ops.OpSum
  @test Ops.OpSum((2.3, "X", 1)) isa Ops.OpSum
  @test Ops.OpSum("X", 1) isa Ops.OpSum
  @test Ops.OpSum(2, "X", 1) isa Ops.OpSum
  @test Ops.OpSum([Op("X", 1), 2Op("Y", 1)]) isa Ops.OpSum

  @testset "Expand expression, 2 products" begin
    expr = (Op("X", 1) + Op("Y", 2)) * (Op("Z", 1) + Op("W", 2))
    expr_expanded =
      Op("X", 1) * Op("Z", 1) +
      Op("Y", 2) * Op("Z", 1) +
      Op("X", 1) * Op("W", 2) +
      Op("Y", 2) * Op("W", 2)
    @test expand(expr) == expr_expanded
  end

  @testset "Expand expression, 3 products" begin
    expr = (Op("X", 1) + Op("Y", 2)) * (Op("Z", 1) + Op("W", 2)) * (Op("A", 1) + Op("B", 2))
    expr_expanded =
      Op("X", 1) * Op("Z", 1) * Op("A", 1) +
      Op("Y", 2) * Op("Z", 1) * Op("A", 1) +
      Op("X", 1) * Op("W", 2) * Op("A", 1) +
      Op("Y", 2) * Op("W", 2) * Op("A", 1) +
      Op("X", 1) * Op("Z", 1) * Op("B", 2) +
      Op("Y", 2) * Op("Z", 1) * Op("B", 2) +
      Op("X", 1) * Op("W", 2) * Op("B", 2) +
      Op("Y", 2) * Op("W", 2) * Op("B", 2)
    @test expand(expr) == expr_expanded
  end
end
