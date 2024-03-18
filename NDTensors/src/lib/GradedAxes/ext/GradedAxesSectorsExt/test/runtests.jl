@eval module $(gensym())
using NDTensors.GradedAxes: dual, fuse_labels
using NDTensors.Sectors: U1, Z
using Test: @test, @testset

@testset "GradedAxesSectorsExt" begin
  @test fuse_labels(U1(1), U1(2)) == U1(3)
  @test dual(U1(2)) == U1(-2)

  @test fuse_labels(Z{2}(1), Z{2}(1)) == Z{2}(0)
  @test fuse_labels(Z{2}(0), Z{2}(1)) == Z{2}(1)
  @test dual(Z{2}(1)) == Z{2}(1)
  @test dual(Z{2}(0)) == Z{2}(0)
end
end
