@eval module $(gensym())
using NDTensors.NestedPermutedDimsArrays: NestedPermutedDimsArray
using Test: @test, @testset
@testset "NestedPermutedDimsArrays" for elt in (
  Float32, Float64, Complex{Float32}, Complex{Float64}
)
  a = map(_ -> randn(elt, 2, 3, 4), CartesianIndices((2, 3, 4)))
  perm = (3, 1, 2)
  p = NestedPermutedDimsArray(a, perm)
  T = PermutedDimsArray{elt,3,perm,invperm(perm),eltype(a)}
  @test typeof(p) === NestedPermutedDimsArray{T,3,perm,invperm(perm),typeof(a)}
  @test size(p) == (4, 2, 3)
  @test eltype(p) === T
  for I in eachindex(p)
    @test size(p[I]) == (4, 2, 3)
    @test p[I] == permutedims(a[CartesianIndex(map(i -> Tuple(I)[i], invperm(perm)))], perm)
  end
  x = randn(elt, 4, 2, 3)
  p[3, 1, 2] = x
  @test p[3, 1, 2] == x
  @test a[1, 2, 3] == permutedims(x, invperm(perm))
end
end
