@eval module $(gensym())
using NDTensors.GradedAxes: GradedAxes, GradedOneTo, gradedrange, tensor_product
using Test: @test, @testset
@testset "GradedAxes.tensor_product" begin
  GradedAxes.fuse_labels(x::String, y::String) = x * y
  a = gradedrange(["x" => 2, "y" => 3])
  b = tensor_product(a, a)
  @test length(b) == 25
  @test b isa GradedOneTo
end
end
