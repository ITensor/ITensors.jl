@eval module $(gensym())
using NDTensors: NDTensors
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
  "SetParameters",
  "SmallVectors",
  "SortedSets",
  "SparseArrayDOKs",
  "TagSets",
  "TensorAlgebra",
  "TypeParameterAccessors",
  "UnallocatedArrays",
  "UnspecifiedTypes",
  "Expose",
]
  include(joinpath(pkgdir(NDTensors), "src", "lib", lib, "test", "runtests.jl"))
end
end
