# # DiagonalArrays.jl
#
# A Julia `DiagonalArray` type.

using NDTensors.DiagonalArrays: DiagonalArray, DiagIndex, DiagIndices
using Test

function main()
  d = DiagonalArray([1.0, 2, 3], 3, 4, 5)
  @test d[1, 1, 1] == 1
  @test d[2, 2, 2] == 2
  @test d[1, 2, 1] == 0

  d[2, 2, 2] = 22
  @test d[2, 2, 2] == 22

  @test length(d[DiagIndices()]) == 3
  @test Array(d) == d
  @test d[DiagIndex(2)] == d[2, 2, 2]

  d[DiagIndex(2)] = 222
  @test d[2, 2, 2] == 222

  a = randn(3, 4, 5)
  new_diag = randn(3)
  a[DiagIndices()] = new_diag
  d[DiagIndices()] = a[DiagIndices()]

  @test a[DiagIndices()] == new_diag
  @test d[DiagIndices()] == new_diag

  permuted_d = permutedims(d, (3, 2, 1))
  @test permuted_d isa DiagonalArray
  @test permuted_d == d

  mapped_d = map(x -> 2x, d)
  @test mapped_d isa DiagonalArray
  @test mapped_d == map(x -> 2x, Array(d))

  return nothing
end

main()

#=
You can generate this README with:
```julia
using Literate
using NDTensors.DiagonalArrays
dir = joinpath(pkgdir(DiagonalArrays), "src", "DiagonalArrays")
Literate.markdown(joinpath(dir, "examples", "README.jl"), dir; flavor=Literate.CommonMarkFlavor())
```
=#
