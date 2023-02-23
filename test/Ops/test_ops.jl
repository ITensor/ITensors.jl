using Test
using ITensors
using LinearAlgebra

using ITensors.Ops #: Scaled, Prod, Sum, expand

function heisenberg(N)
  os = Sum{Op}()
  for j in 1:(N - 1)
    os += "Sz", j, "Sz", j + 1
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
  end
  return os
end

@testset "Basic Ops" begin
  x1 = Op("X", 1)
  x2 = Op("X", 2)
  I1 = Op(I, 1)
  I2 = Op(I, 2)
  y1 = Op("Y", 1)
  y2 = Op("Y", 2)
  CX12 = Op("CX", 1, 2)
  Ry4 = Op("Ry", 4; θ=π / 3)

  @test 2y2 isa Scaled{<:Number,Op}
  @test coefficient(2y2) == 2
  @test y2 / 2 isa Scaled{<:Number,Op}
  @test coefficient(y2 / 2) ≈ 0.5
  @test -y2 isa Scaled{<:Number,Op}
  @test 1y2 + x1 isa Sum{<:Scaled{<:Number,Op}}
  @test 1y2 + x1 isa Sum{Scaled{Int,Op}}
  @test x1 * y2 isa Prod{Op}
  @test 2x1 * y2 isa Scaled{<:Number,Prod{Op}}
  @test x1 * y2 + CX12 isa Sum{Prod{Op}}
  @test x1 * y2 + x1 * CX12 isa Sum{Prod{Op}}
  @test x1 * y2 + 2CX12 isa Sum{<:Scaled{<:Number,Prod{Op}}}
  @test x1 * y2 - CX12 isa Sum{<:Scaled{<:Number,Prod{Op}}}
  @test 2x1 * y2 + 2CX12 isa Sum{<:Scaled{<:Number,Prod{Op}}}
  @test 2x1 * y2 - 2CX12 isa Sum{<:Scaled{<:Number,Prod{Op}}}
  @test (2x1 * y2 - 2CX12) / 3 isa Sum{<:Scaled{<:Number,Prod{Op}}}

  o1 = Op("X", 1)
  o2 = Op("Y", 2)

  @test o1 + o2 isa Sum{Op}
  @test o1 - o2 isa Sum{Scaled{Int,Op}}
  @test 1.3 * o1 isa Scaled{Float64,Op}
  @test o1 * 1.4 isa Scaled{Float64,Op}
  @test o1 + o2 + o2 isa Sum{Op}
  @test 1.3o1 + 1.3o2 isa Sum{Scaled{Float64,Op}}
  @test 1.3o1 + o2 isa Sum{Scaled{Float64,Op}}
  @test (o1 + o2) + (o1 + o2) isa Sum{Op}
  @test 1.3o1 + 1o2 isa Sum{Scaled{Float64,Op}}
  @test 1.3 * (o1 + o2) isa Sum{Scaled{Float64,Op}}
  @test o1 + o2 + 1.3o2 isa Sum{Scaled{Float64,Op}}
  @test o1 * o2 isa Prod{Op}
  @test o1 * o2 * o2 isa Prod{Op}
  @test o1 * (o2 * o2) isa Prod{Op}
  @test 1.3 * o1 * o2 isa Scaled{Float64,Prod{Op}}
  @test 1.3 * (o1 * o2) isa Scaled{Float64,Prod{Op}}
  @test 1.3 * o1 * o2 + o1 isa Sum{Scaled{Float64,Prod{Op}}}
  @test 1.3 * o1 * o2 + o1 * o2 isa Sum{Scaled{Float64,Prod{Op}}}
  @test 1.3 * o1 * o2 + 1.3 * o1 * o2 isa Sum{Scaled{Float64,Prod{Op}}}
  @test 1.3 * o1 * o2 + 1.3 * o1 * o2 + o1 isa Sum{Scaled{Float64,Prod{Op}}}
  @test 1.3 * o1 * o2 + 1.3 * o1 * o2 + 1.2 * o1 isa Sum{Scaled{Float64,Prod{Op}}}
  @test Ops.OpSum() + o1 isa Sum{Scaled{ComplexF64,Prod{Op}}}
  @test Ops.OpSum() + 1.2 * o1 isa Sum{Scaled{ComplexF64,Prod{Op}}}
  @test Ops.OpSum() + (1.2 + 2.3im) * o1 isa Sum{Scaled{ComplexF64,Prod{Op}}}
  @test Ops.OpSum() + 1.2 * o1 * o2 isa Sum{Scaled{ComplexF64,Prod{Op}}}
  @test Ops.OpSum() - 1.2 * o1 * o2 isa Sum{Scaled{ComplexF64,Prod{Op}}}
  @test Ops.OpSum() + o1 * o2 isa Sum{Scaled{ComplexF64,Prod{Op}}}
  @test o1 + o2 + 2.3 * o1 * o2 isa Sum{Scaled{Float64,Prod{Op}}}
  @test Sum{Op}() + ("X", 1, "Y", 2) + ("Y", 2) isa Sum{Prod{Op}}
  @test Sum{Op}() + ("X", 1, "Y", 2) + (1.2, "Y", 2) isa Sum{Scaled{Float64,Prod{Op}}}
  @test OpSum() - (0.5, "Z", 1, "Z", 2) isa Sum{Scaled{ComplexF64,Prod{Op}}}

  N = 4
  s = siteinds("Qubit", N)

  @test ITensor(o1, s) ≈ op("X", s, 1)
  @test ITensor(2 * o1, s) ≈ 2 * ITensor(o1, s)
  @test ITensor(o1 * o2, s) ≈ ITensor(o1, s) * ITensor(o2, s)
  @test ITensor(2 * o1 * o2, s) ≈ 2 * ITensor(o1, s) * ITensor(o2, s)
  @test ITensor(2 * o1 * o2 + o1 * o2, s) ≈
    2 * ITensor(o1, s) * ITensor(o2, s) + ITensor(o1, s) * ITensor(o2, s)
  @test ITensor(exp(o1), s) ≈ exp(ITensor(o1, s))
  @test ITensor(exp(1.2 * o1), s) ≈ exp(1.2 * ITensor(o1, s))
  @test ITensor(1.3 * exp(1.2 * o1), s) ≈ 1.3 * exp(1.2 * ITensor(o1, s))

  o = (2x1 * y2 - 2CX12) / 3
  @test coefficient(o[1]) ≈ 2 / 3
  @test coefficient(o[2]) ≈ -2 / 3

  t1 = ITensor(x1, s)
  @test hassameinds(t1, (s[1]', dag(s[1])))
  @test t1[1, 1] == 0
  @test t1[1, 2] == 1

  @test ITensor(2.3x1, s) ≈ 2.3 * t1
  @test ITensor(x1 + x1, s) ≈ 2t1
  @test ITensor(x1 + 2.3x1, s) ≈ 3.3t1

  @test ITensor(Op(I, 2), s) ≈ ITensor([1 0; 0 1], s[2]', dag(s[2]))
  @test ITensor(Op(2I, 2), s) ≈ 2 * ITensor([1 0; 0 1], s[2]', dag(s[2]))

  c = x1 * y2 * CX12
  cdag = c'
  @test c[1]' == cdag[3]
  @test c[2]' == cdag[2]
  @test c[3]' == cdag[1]

  x = randn(2, 2)
  tx = ITensor(Op(x, 3), s)
  @test tx[s[3]' => 1, s[3] => 2] == x[1, 2]

  @test ITensor(x1 * x1, s) ≈ ITensor(Op([1 0; 0 1], 1), s)
  @test ITensor(x1 * x1 * x1, s) ≈ ITensor(Op([0 1; 1 0], 1), s)
  @test ITensor(2x1 * x1, s) ≈ ITensor(Op([2 0; 0 2], 1), s)
  @test ITensor(x1 * y1, s) ≈ ITensor(Op([im 0; 0 -im], 1), s)
  @test ITensor(y1 * x1, s) ≈ ITensor(Op([-im 0; 0 im], 1), s)
  @test ITensor(2x1 * x1 + y1, s) ≈
    ITensor(2 * [1 0; 0 1] + [0 -im; im 0], s[1]', dag(s[1]))

  # TODO: Need to add support for filling out with "Id" or "F"
  @test_broken ITensor(2y1 * x2 + x1, s) ≈
    2 * ITensor(y1, s) * ITensor(x2, s) + ITensor(x1, s) * ITensor(I2, s)

  @test y1'' == y1

  @test ITensor(y1', s) ≈ ITensor(Op([0 -im; im 0], 1), s)

  @test ITensor(exp(x1), s) ≈ ITensor(Op(exp([0 1; 1 0]), 1), s)
  @test ITensor(exp(2x1 * x1), s) ≈ ITensor(exp(2 * [1 0; 0 1]), s[1]', dag(s[1]))
  @test ITensor(exp(2x1 * x1 + y1), s) ≈
    ITensor(exp(2 * [1 0; 0 1] + [0 -im; im 0]), s[1]', dag(s[1]))

  @test ITensor(I1, s) ≈ ITensor([1 0; 0 1], s[1]', dag(s[1]))

  @test exp(Op("X", 1)) * Op("Y", 2) isa Prod{Any}
  @test ITensor(exp(Op("X", 1)) * Op("Y", 1), s) ≈
    product(exp(ITensor(Op("X", 1), s)), ITensor(Op("Y", 1), s))

  # TODO: Need to define `(::Scaled * ::Op)::Scaled`
  @test_broken 2exp(Op("X", 1)) * Op("Y", 2) isa Scaled{<:Number,Prod{Any}}

  H = Sum{Scaled{Bool,Prod{Op}}}() - Op("X", 1)
  @test H isa Sum
  @test H isa Sum{<:Scaled}
  @test H isa Sum{<:Scaled{<:Number,<:Prod}}
  @test H isa Sum{<:Scaled{<:Number,Prod{Op}}}
  @test H isa Sum{Scaled{T,Prod{Op}}} where {T}
  @test H isa Sum{Scaled{Int,Prod{Op}}}
  @test length(H) == 1
  @test coefficient(H[1]) == -1

  H = Sum{Op}() - Op("X", 1)
  @test H isa Sum
  @test H isa Sum{<:Scaled}
  @test H isa Sum{<:Scaled{<:Number,Op}}
  @test H isa Sum{Scaled{T,Op}} where {T}
  @test H isa Sum{Scaled{Int,Op}}
  @test length(H) == 1
  @test coefficient(H[1]) == -1

  # OpSum conversion
  H = Sum{Op}()
  H -= 2.3, "X", 1, "X", 2
  H += 1.2, "Z", 1
  H += 1.3, "Z", 2, (θ=π / 3,)
  @test H isa Sum{Scaled{Float64,Prod{Op}}}
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

  @test_broken Sum{Op}(("X", 1)) isa Sum{Op}
  @test_broken Sum{Op}((2.3, "X", 1)) isa Sum{Scaled{Float64,Op}}
  @test_broken Sum{Op}("X", 1) isa Sum{Op}
  @test_broken Sum{Op}(2, "X", 1) isa Sum{Scaled{Int,Op}}
  @test_broken Sum{Op}([Op("X", 1), 2Op("Y", 1)]) isa Sum
  @test_broken Sum{Op}([Op("X", 1), 2Op("Y", 1)]) isa Sum{<:Scaled}
  @test_broken Sum{Op}([Op("X", 1), 2Op("Y", 1)]) isa Sum{<:Scaled{<:Number,Op}}
  @test_broken Sum{Op}([Op("X", 1), 2Op("Y", 1)]) isa Sum{Scaled{Int,Op}}

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

  H = heisenberg(4)
  @test length(H) == 9
  @test H^2 == H * H
  @test length(H^2) == 2
  @test length(expand(H^2)) == 81

  @testset "Conversion to Sum of ITensors" begin
    H = Sum{Op}() + ("X", 1) + ("Y", 2)
    @test_broken H == Sum{Op}([("X", 1), ("Y", 2)])
    @test H == Sum{Op}() + Op("X", 1) + Op("Y", 2)
    s = siteinds("Qubit", 2)
    Hₜ = Sum{ITensor}(H, s)
    @test Hₜ isa Sum{ITensor}
    @test Hₜ[1] ≈ ITensor(Op("X", 1), s)
    @test Hₜ[2] ≈ ITensor(Op("Y", 2), s)
  end

  @testset "Conversion to Prod of ITensors" begin
    C = Prod{Op}() * ("X", 1) * ("Y", 2)
    @test_broken C == Prod{Op}([("X", 1), ("Y", 2)])
    @test C == Prod{Op}() * Op("X", 1) * Op("Y", 2)
    @test C == Op("X", 1) * Op("Y", 2)
    s = siteinds("Qubit", 2)
    Cₜ = Prod{ITensor}(C, s)
    @test Cₜ isa Prod{ITensor}
    @test Cₜ[1] ≈ ITensor(Op("X", 1), s)
    @test Cₜ[2] ≈ ITensor(Op("Y", 2), s)
  end
end
