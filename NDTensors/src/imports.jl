# Makes `cpu` available as `NDTensors.cpu`.
# TODO: Define `cpu`, `cu`, etc. in a module `DeviceAbstractions`,
# similar to:
# https://github.com/JuliaGPU/KernelAbstractions.jl
# https://github.com/oschulz/HeterogeneousComputing.jl

using Adapt
using Base.Threads
using Compat
using Dictionaries
using Folds
using InlineStrings
using Random
using LinearAlgebra
using StaticArrays
using Functors
using SimpleTraits
using SplitApplyCombine
using Strided
using TimerOutputs
using TupleTools

for lib in [
  :AllocateData,
  :BackendSelection,
  :BaseExtensions,
  :UnspecifiedTypes,
  :TypeParameterAccessors,
  :Expose,
  :GPUArraysCoreExtensions,
  :AMDGPUExtensions,
  :CUDAExtensions,
  :MetalExtensions,
  :BroadcastMapConversion,
  :RankFactorization,
  :LabelledNumbers,
  :GradedAxes,
  :SymmetrySectors,
  :TensorAlgebra,
  :SparseArrayInterface,
  :SparseArrayDOKs,
  :DiagonalArrays,
  :BlockSparseArrays,
  :NamedDimsArrays,
  :SmallVectors,
  :SortedSets,
  :TagSets,
  :UnallocatedArrays,
]
  include("lib/$(lib)/src/$(lib).jl")
  @eval using .$lib: $lib
end
# TODO: This is defined for backwards compatibility,
# delete this alias once downstream packages change over
# to using `BackendSelection`.
const AlgorithmSelection = BackendSelection

using Base: @propagate_inbounds, ReshapedArray, DimOrInd, OneTo

using Base.Cartesian: @nexprs

using Base.Threads: @spawn

using .AMDGPUExtensions: roc
using .CUDAExtensions: cu
using .GPUArraysCoreExtensions: cpu
using .MetalExtensions: mtl

import Base:
  # Types
  AbstractFloat,
  Array,
  CartesianIndex,
  Complex,
  IndexStyle,
  Tuple,
  # Symbols
  +,
  -,
  *,
  /,
  # Methods
  checkbounds,
  complex,
  convert,
  conj,
  copy,
  copyto!,
  eachindex,
  eltype,
  empty,
  fill,
  fill!,
  getindex,
  hash,
  imag,
  isempty,
  isless,
  iterate,
  length,
  map,
  permutedims,
  permutedims!,
  print,
  promote_rule,
  randn,
  real,
  reshape,
  setindex,
  setindex!,
  show,
  size,
  stride,
  strides,
  summary,
  to_indices,
  unsafe_convert,
  view,
  zero,
  zeros

import Base.Broadcast: Broadcasted, BroadcastStyle

import Adapt: adapt_structure, adapt_storage

import LinearAlgebra: diag, exp, norm, qr, svd, mul!

import TupleTools: isperm
