@eval module $(gensym())
using BlockArrays: Block
using NDTensors.TensorAlgebra: ⊗
using NDTensors.GradedAxes: GradedAxes, gradedrange, sector
using Test: @test, @testset

struct U1
  dim::Int
end
Base.isless(l1::U1, l2::U1) = isless(l1.dim, l2.dim)
GradedAxes.fuse(l1::U1, l2::U1) = U1(l1.dim + l2.dim)
GradedAxes.dual(l::U1) = U1(-l.dim)

@testset "TensorAlgebraGradedAxesExt" begin
  a1 = gradedrange([U1(0) => 2, U1(1) => 3])
  a2 = gradedrange([U1(2) => 3, U1(3) => 4])
  a = a1 ⊗ a2
  @test sector(a, Block(1)) == U1(2)
  @test sector(a, Block(2)) == U1(3)
  @test sector(a, Block(3)) == U1(3)
  @test sector(a, Block(4)) == U1(4)
  @test a[Block(1)] == 1:6
  @test a[Block(2)] == 7:15
  @test a[Block(3)] == 16:23
  @test a[Block(4)] == 24:35
end
end
