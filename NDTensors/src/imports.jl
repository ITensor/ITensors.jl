# Makes `cpu` available as `NDTensors.cpu`.
# TODO: Define `cpu`, `cu`, etc. in a module `DeviceAbstractions`,
# similar to:
# https://github.com/JuliaGPU/KernelAbstractions.jl
# https://github.com/oschulz/HeterogeneousComputing.jl

using Adapt
using Base.Threads
using Dictionaries
using Folds
using Functors
using InlineStrings
using LinearAlgebra
using Random
using SimpleTraits
using SplitApplyCombine
using StaticArrays
using Strided
using TimerOutputs
using TupleTools

for lib in [
        :BackendSelection,
        :Expose,
        :GPUArraysCoreExtensions,
        :AMDGPUExtensions,
        :CUDAExtensions,
        :MetalExtensions,
        :RankFactorization,
    ]
    include("lib/$(lib)/src/$(lib).jl")
    @eval using .$lib: $lib
end
# TODO: This is defined for backwards compatibility,
# delete this alias once downstream packages change over
# to using `BackendSelection`.
const AlgorithmSelection = BackendSelection

import Adapt: adapt_storage, adapt_structure
import Base.Broadcast: BroadcastStyle, Broadcasted
import Base: # Methods
    checkbounds, # Symbols
    +, # Types
    AbstractFloat, *, -, /, Array, CartesianIndex, Complex, IndexStyle, Tuple, complex,
    conj, convert, copy, copyto!, eachindex, eltype, empty, fill, fill!, getindex, hash,
    imag, isempty, isless, iterate, length, map, permutedims, permutedims!, print,
    promote_rule, randn, real, reshape, setindex, setindex!, show, size, stride, strides,
    summary, to_indices, unsafe_convert, view, zero, zeros
import LinearAlgebra: diag, exp, mul!, norm, qr, svd
import TupleTools: isperm
using .AMDGPUExtensions: roc
using .CUDAExtensions: cu
using .GPUArraysCoreExtensions: cpu
using .MetalExtensions: mtl
using Base.Cartesian: @nexprs
using Base.Threads: @spawn
using Base: @propagate_inbounds, DimOrInd, OneTo, ReshapedArray
