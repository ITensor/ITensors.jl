@eval module $(gensym())
using NDTensors.GradedAxes: dual, fuse
using NDTensors.Sectors: U1, Z
using Test: @test, @testset

@testset "GradedAxesSectorsExt" begin
  @test fuse(U1(1), U1(2)) == U1(3)
  @test dual(U1(2)) == U1(-2)

  @test fuse(Z{2}(1), Z{2}(1)) == Z{2}(0)
  @test fuse(Z{2}(0), Z{2}(1)) == Z{2}(1)
  @test dual(Z{2}(1)) == Z{2}(1)
  @test dual(Z{2}(0)) == Z{2}(0)
end
end
