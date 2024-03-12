# Makes `cpu` available as `NDTensors.cpu`.
# TODO: Define `cpu`, `cu`, etc. in a module `DeviceAbstractions`,
# similar to:
# https://github.com/JuliaGPU/KernelAbstractions.jl
# https://github.com/oschulz/HeterogeneousComputing.jl
using .Expose: cpu

import .CUDAExtensions: cu
import .GPUArraysCoreExtensions: storagemode
import .MetalExtensions: mtl

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
