@eval module $(gensym())
using Test: @test, @testset

using BlockArrays: blocklength, blocklengths

using NDTensors.GradedAxes:
  GradedAxes,
  GradedOneTo,
  OneToOne,
  dual,
  fusion_product,
  flip,
  gradedrange,
  gradedisequal,
  isdual,
  tensor_product

struct U1
  n::Int
end
GradedAxes.dual(c::U1) = U1(-c.n)
Base.isless(c1::U1, c2::U1) = c1.n < c2.n
GradedAxes.fuse_labels(x::U1, y::U1) = U1(x.n + y.n)

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
  g0 = OneToOne()
  @test gradedisequal(fusion_product(g0, g0), g0)

  a = gradedrange([U1(1) => 1, U1(2) => 3, U1(1) => 1])

  b = fusion_product(a)
  @test gradedisequal(b, gradedrange([U1(1) => 2, U1(2) => 3]))

  c = fusion_product(a, a)
  @test gradedisequal(c, gradedrange([U1(2) => 4, U1(3) => 12, U1(4) => 9]))

  d = fusion_product(a, a, a)
  @test gradedisequal(d, gradedrange([U1(3) => 8, U1(4) => 36, U1(5) => 54, U1(6) => 27]))
end

@testset "dual and tensor_product" begin
  a = gradedrange([U1(1) => 1, U1(2) => 3, U1(1) => 1])
  ad = dual(a)

  b = fusion_product(ad)
  @test b isa GradedOneTo
  @test !isdual(b)
  @test gradedisequal(b, gradedrange([U1(-2) => 3, U1(-1) => 2]))

  c = fusion_product(ad, ad)
  @test c isa GradedOneTo
  @test !isdual(c)
  @test gradedisequal(c, gradedrange([U1(-4) => 9, U1(-3) => 12, U1(-2) => 4]))

  d = fusion_product(ad, a)
  @test !isdual(d)
  @test gradedisequal(d, gradedrange([U1(-1) => 6, U1(0) => 13, U1(1) => 6]))

  e = fusion_product(a, ad)
  @test !isdual(d)
  @test gradedisequal(e, d)
end
end
