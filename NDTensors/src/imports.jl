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
using GPUArraysCore
using InlineStrings
using Random
using LinearAlgebra
using StaticArrays
using Functors
using HDF5
using SimpleTraits
using SplitApplyCombine
using Strided
using TimerOutputs
using TupleTools

for lib in [
  :AlgorithmSelection,
  :AllocateData,
  :BaseExtensions,
  :UnspecifiedTypes,
  :TypeParameterAccessors,
  :GPUArraysCoreExtensions,
  :CUDAExtensions,
  :MetalExtensions,
  :Expose,
  :BroadcastMapConversion,
  :RankFactorization,
  :Sectors,
  :LabelledNumbers,
  :GradedAxesNext,
  :GradedAxes,
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

using Base: @propagate_inbounds, ReshapedArray, DimOrInd, OneTo

using Base.Cartesian: @nexprs

using Base.Threads: @spawn

using .CUDAExtensions: cu
using .MetalExtensions: mtl
using .GPUArraysCoreExtensions: cpu

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
