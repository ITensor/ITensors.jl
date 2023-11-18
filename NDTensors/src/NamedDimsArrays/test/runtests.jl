using NDTensors.NamedDimsArrays:
  NamedDimsArrays, NamedDimsArray, align, dimnames, isnamed, named, unname
using Test

@testset "NamedDimsArrays" begin
  a = randn(3, 4)
  na = named(a, ("i", "j"))
  # TODO: Call `namedsize`?
  i, j = size(na)
  # TODO: Call `namedaxes`?
  ai, aj = axes(na)
  @test !isnamed(a)
  @test isnamed(na)
  @test dimnames(na) == ("i", "j")
  @test na[1, 1] == a[1, 1]
  na[1, 1] = 11
  @test na[1, 1] == 11
  @test size(a) == (named(3, "i"), named(4, "j"))
  @test length(a) == 12
  @test axes(a) == (named(1:3, "i"), named(1:4, "j"))
  @test randn(named(3, "i"), named(4, "j")) isa NamedDimsArray
  @test na["i" => 1, "j" => 2] == a[1, 2]
  @test na["j" => 2, "i" => 1] == a[1, 2]
  na["j" => 2, "i" => 1] = 12
  @test na[1, 2] == 12
  @test na[j => 1, i => 2] == a[2, 1]
  @test na[aj => 1, ai => 2] == a[2, 1]
  na[j => 1, i => 2] = 21
  @test na[2, 1] == 21
  na[aj => 1, ai => 2] = 2211
  @test na[2, 1] == 2211
  na′ = align(na, ("j", "i"))
  @test a == permutedims(unname(na′), (2, 1))
  na′ = align(na, (j, i))
  @test a == permutedims(unname(na′), (2, 1))
  na′ = align(na, (aj, ai))
  @test a == permutedims(unname(na′), (2, 1))
end
