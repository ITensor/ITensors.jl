@eval module $(gensym())
using NDTensors: NDTensors
using Test: @testset
@testset "Test NDTensors lib $lib" for lib in [
  "AllocateData",
  "AMDGPUExtensions",
  "BackendSelection",
  "BaseExtensions",
  "BlockSparseArrays",
  "BroadcastMapConversion",
  "CUDAExtensions",
  "DiagonalArrays",
  "GradedAxes",
  "GPUArraysCoreExtensions",
  "LabelledNumbers",
  "MetalExtensions",
  "NamedDimsArrays",
  "NestedPermutedDimsArrays",
  "SmallVectors",
  "SortedSets",
  "SparseArraysBase",
  "SymmetrySectors",
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
