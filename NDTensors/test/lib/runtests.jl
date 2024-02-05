@eval module $(gensym())
using Test: @testset
@testset "Test NDTensors lib $lib" for lib in [
  "AlgorithmSelection",
  "AllocateData",
  "BaseExtensions",
  "BlockSparseArrays",
  "BroadcastMapConversion",
  "DiagonalArrays",
  "GradedAxes",
  "NamedDimsArrays",
  "Sectors",
  "TypeParameterAccessors",
  "SmallVectors",
  "SortedSets",
  "SparseArrayDOKs",
  "TagSets",
  "TensorAlgebra",
  "UnallocatedArrays",
  "UnspecifiedTypes",
  "Unwrap",
]
  using NDTensors: NDTensors
  include(joinpath(pkgdir(NDTensors), "src", "lib", lib, "test", "runtests.jl"))
end
end
