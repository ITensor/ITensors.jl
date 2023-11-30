# # DiagonalArrays.jl
#
# A Julia `DiagonalArray` type.

using NDTensors.DiagonalArrays:
  DiagonalArray, DiagonalMatrix, DiagIndex, DiagIndices, diaglength, isdiagindex
using Test

function main()
  d = DiagonalMatrix([1.0, 2.0, 3.0])
  @test eltype(d) == Float64
  @test diaglength(d) == 3
  @test size(d) == (3, 3)
  @test d[1, 1] == 1
  @test d[2, 2] == 2
  @test d[3, 3] == 3
  @test d[1, 2] == 0

  d = DiagonalArray([1.0, 2.0, 3.0], 3, 4, 5)
  @test eltype(d) == Float64
  @test diaglength(d) == 3
  @test d[1, 1, 1] == 1
  @test d[2, 2, 2] == 2
  @test d[3, 3, 3] == 3
  @test d[1, 2, 1] == 0

  d[2, 2, 2] = 22
  @test d[2, 2, 2] == 22

  d_r = reshape(d, 3, 20)
  @test size(d_r) == (3, 20)
  @test all(I -> d_r[I] == d[I], LinearIndices(d))

  @test length(d[DiagIndices(:)]) == 3
  @test Array(d) == d
  @test d[DiagIndex(2)] == d[2, 2, 2]

  d[DiagIndex(2)] = 222
  @test d[2, 2, 2] == 222

  a = randn(3, 4, 5)
  new_diag = randn(3)
  a[DiagIndices(:)] = new_diag
  d[DiagIndices(:)] = a[DiagIndices(:)]

  @test a[DiagIndices(:)] == new_diag
  @test d[DiagIndices(:)] == new_diag

  permuted_d = permutedims(d, (3, 2, 1))
  @test permuted_d isa DiagonalArray
  @test permuted_d[DiagIndices(:)] == d[DiagIndices(:)]
  @test size(d) == (3, 4, 5)
  @test size(permuted_d) == (5, 4, 3)
  for I in eachindex(d)
    if !isdiagindex(d, I)
      @test iszero(d[I])
    else
      @test !iszero(d[I])
    end
  end

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
dir = joinpath(pkgdir(DiagonalArrays), "src", "lib", "DiagonalArrays")
Literate.markdown(joinpath(dir, "examples", "README.jl"), dir; flavor=Literate.CommonMarkFlavor())
```
=#
