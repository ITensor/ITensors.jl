@eval module $(gensym())
using NDTensors.NamedDimsArrays: NamedDimsArray, dimnames
using NDTensors: NDTensors
using Test: @test, @testset

@testset "NDTensorsNamedDimsArraysExt" begin
  elt = Float64

  a = NamedDimsArray(randn(elt, 2, 2), ("i", "j"))
  b = NDTensors.similar(a)
  @test b isa NamedDimsArray{elt}
  @test eltype(b) === elt
  @test dimnames(b) == ("i", "j")
  @test size(b) == (2, 2)

  a = NamedDimsArray(randn(elt, 2, 2), ("i", "j"))
  b = NDTensors.similar(a, Float32)
  @test b isa NamedDimsArray{Float32}
  @test eltype(b) === Float32
  @test dimnames(b) == ("i", "j")
  @test size(b) == (2, 2)

  a = NamedDimsArray(randn(elt, 2, 2), ("i", "j"))
  b = copy(a)
  α = randn(elt)
  b = NDTensors.fill!!(b, α)
  @test b isa NamedDimsArray{elt}
  @test eltype(b) === elt
  @test dimnames(b) == ("i", "j")
  @test size(b) == (2, 2)
  @test all(==(α), b)
end
end
