@eval module $(gensym())
using NDTensors.GradedAxes: GradedAxes, GradedUnitRange, gradedrange, tensor_product
using Test: @test, @testset
@testset "GradedAxes.tensor_product" begin
  GradedAxes.fuse_labels(x::String, y::String) = x * y
  a = gradedrange(["x" => 2, "y" => 3])
  b = tensor_product(a, a)
  @test length(b) == 25
  @test b isa GradedUnitRange
end
end
