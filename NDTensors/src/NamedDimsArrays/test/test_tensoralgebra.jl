@eval module $(gensym())
using Test: @test, @testset
using NDTensors.NamedDimsArrays: dimnames, named, unname
@testset "NamedDimsArrays $(@__FILE__) (eltype=$elt)" for elt in (
  Float32, ComplexF32, Float64, ComplexF64
)
  a = randn(elt, 2, 2)
  na = named(a, ("i", "j"))
  b = randn(elt, 2, 2)
  nb = named(b, ("j", "i"))

  nc = similar(na)
  @test size(nc) == (2, 2)
  @test eltype(nc) == elt
  @test dimnames(nc) == ("i", "j")

  nc = similar(na, (3, 3))
  @test size(nc) == (3, 3)
  @test eltype(nc) == elt
  @test dimnames(nc) == ("i", "j")

  nc = similar(na, 3, 3)
  @test size(nc) == (3, 3)
  @test eltype(nc) == elt
  @test dimnames(nc) == ("i", "j")

  nc = similar(na, Float32)
  @test size(nc) == (2, 2)
  @test eltype(nc) == Float32
  @test dimnames(nc) == ("i", "j")

  nc = similar(na, Float32, (3, 3))
  @test size(nc) == (3, 3)
  @test eltype(nc) == Float32
  @test dimnames(nc) == ("i", "j")

  nc = similar(na, Float32, 3, 3)
  @test size(nc) == (3, 3)
  @test eltype(nc) == Float32
  @test dimnames(nc) == ("i", "j")

  nc = permutedims(na, (2, 1))
  @test unname(nc) ≈ permutedims(unname(na), (2, 1))
  @test dimnames(nc) == ("j", "i")
  @test nc ≈ na

  nc = 2 * na
  @test unname(nc) ≈ 2 * a
  @test eltype(nc) === elt

  nc = 2 .* na
  @test unname(nc) ≈ 2 * a
  @test eltype(nc) === elt

  nc = na + nb
  @test unname(nc, ("i", "j")) ≈ a + permutedims(b, (2, 1))
  @test eltype(nc) === elt

  nc = na .+ nb
  @test unname(nc, ("i", "j")) ≈ a + permutedims(b, (2, 1))
  @test eltype(nc) === elt

  nc = map(+, na, nb)
  @test unname(nc, ("i", "j")) ≈ a + permutedims(b, (2, 1))
  @test eltype(nc) === elt

  nc = named(randn(elt, 2, 2), ("i", "j"))
  map!(+, nc, na, nb)
  @test unname(nc, ("i", "j")) ≈ a + permutedims(b, (2, 1))
  @test eltype(nc) === elt

  nc = na - nb
  @test unname(nc, ("i", "j")) ≈ a - permutedims(b, (2, 1))
  @test eltype(nc) === elt

  nc = na .- nb
  @test unname(nc, ("i", "j")) ≈ a - permutedims(b, (2, 1))
  @test eltype(nc) === elt
end
end
