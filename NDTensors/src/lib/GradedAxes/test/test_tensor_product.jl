@eval module $(gensym())
using NDTensors.GradedAxes:
  GradedAxes,
  GradedOneTo,
  OneToOne,
  fusion_product,
  gradedrange,
  gradedisequal,
  tensor_product
using BlockArrays: blocklength, blocklengths
using Test: @test, @testset

@testset "GradedAxes.tensor_product" begin
  GradedAxes.fuse_labels(x::String, y::String) = x * y

  g0 = OneToOne()
  @test gradedisequal(tensor_product(g0, g0), g0)

  a = gradedrange(["x" => 2, "y" => 3])
  b = tensor_product(a, a)
  @test b isa GradedOneTo
  @test length(b) == 25
  @test blocklength(b) == 4
  @test blocklengths(b) == [4, 6, 6, 9]
  @test gradedisequal(b, gradedrange(["xx" => 4, "yx" => 6, "xy" => 6, "yy" => 9]))

  c = tensor_product(a, a, a)
  @test c isa GradedOneTo
  @test length(c) == 125
  @test blocklength(c) == 8
end

@testset "GradedAxes.fusion_product" begin
  GradedAxes.fuse_labels(i::Int, j::Int) = i + j

  g0 = OneToOne()
  @test gradedisequal(fusion_product(g0, g0), g0)

  a = gradedrange([1 => 1, 2 => 3, 1 => 1])

  b = fusion_product(a)
  @test gradedisequal(b, gradedrange([1 => 2, 2 => 3]))

  c = fusion_product(a, a)
  @test gradedisequal(c, gradedrange([2 => 4, 3 => 12, 4 => 9]))

  d = fusion_product(a, a, a)
  @test gradedisequal(d, gradedrange([3 => 8, 4 => 36, 5 => 54, 6 => 27]))
end
end
