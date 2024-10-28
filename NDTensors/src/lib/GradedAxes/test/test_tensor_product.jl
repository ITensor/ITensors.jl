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
  space_isequal,
  isdual,
  tensor_product

using NDTensors.LabelledNumbers: labelled_isequal

struct U1
  n::Int
end
GradedAxes.dual(c::U1) = U1(-c.n)
Base.isless(c1::U1, c2::U1) = c1.n < c2.n
GradedAxes.fuse_labels(x::U1, y::U1) = U1(x.n + y.n)

@testset "GradedAxes.tensor_product" begin
  GradedAxes.fuse_labels(x::String, y::String) = x * y

  g0 = OneToOne()
  @test labelled_isequal(g0, g0)
  @test labelled_isequal(tensor_product(g0, g0), g0)

  a = gradedrange(["x" => 2, "y" => 3])
  b = tensor_product(a, a)
  @test b isa GradedOneTo
  @test length(b) == 25
  @test blocklength(b) == 4
  @test blocklengths(b) == [4, 6, 6, 9]
  @test labelled_isequal(b, gradedrange(["xx" => 4, "yx" => 6, "xy" => 6, "yy" => 9]))

  c = tensor_product(a, a, a)
  @test c isa GradedOneTo
  @test length(c) == 125
  @test blocklength(c) == 8
end

@testset "GradedAxes.fusion_product" begin
  g0 = OneToOne()
  @test labelled_isequal(fusion_product(g0, g0), g0)

  a = gradedrange([U1(1) => 1, U1(2) => 3, U1(1) => 1])

  b = fusion_product(a)
  @test labelled_isequal(b, gradedrange([U1(1) => 2, U1(2) => 3]))

  c = fusion_product(a, a)
  @test labelled_isequal(c, gradedrange([U1(2) => 4, U1(3) => 12, U1(4) => 9]))

  d = fusion_product(a, a, a)
  @test labelled_isequal(
    d, gradedrange([U1(3) => 8, U1(4) => 36, U1(5) => 54, U1(6) => 27])
  )
end

@testset "dual and tensor_product" begin
  a = gradedrange([U1(1) => 1, U1(2) => 3, U1(1) => 1])
  ad = dual(a)

  b = fusion_product(ad)
  @test b isa GradedOneTo
  @test !isdual(b)
  @test space_isequal(b, gradedrange([U1(-2) => 3, U1(-1) => 2]))

  c = fusion_product(ad, ad)
  @test c isa GradedOneTo
  @test !isdual(c)
  @test space_isequal(c, gradedrange([U1(-4) => 9, U1(-3) => 12, U1(-2) => 4]))

  d = fusion_product(ad, a)
  @test !isdual(d)
  @test space_isequal(d, gradedrange([U1(-1) => 6, U1(0) => 13, U1(1) => 6]))

  e = fusion_product(a, ad)
  @test !isdual(d)
  @test space_isequal(e, d)
end
end
