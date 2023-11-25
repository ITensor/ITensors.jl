@eval module $(gensym())
using Test: @testset
@testset "Test NDTensors lib $lib" for lib in [
  "BaseExtensions",
  "BlockSparseArrays",
  "DiagonalArrays",
  "NamedDimsArrays",
  "SetParameters",
  "SmallVectors",
  "SortedSets",
  "TagSets",
  "TensorAlgebra",
  "Unwrap",
]
  using NDTensors: NDTensors
  include(joinpath(pkgdir(NDTensors), "src", "lib", lib, "test", "runtests.jl"))
end
end
