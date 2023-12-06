module NDTensors
# TODO: List types, macros, and functions being used.
using Adapt
using Base.Threads
using Compat
using Dictionaries
using FLoops
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
  :SetParameters,
  :BroadcastMapConversion,
  :Unwrap,
  :RankFactorization,
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
]
  include("lib/$(lib)/src/$(lib).jl")
  @eval using .$lib: $lib
end

using Base: @propagate_inbounds, ReshapedArray, DimOrInd, OneTo

using Base.Cartesian: @nexprs

using Base.Threads: @spawn

#####################################
# Imports and exports
#
include("imports.jl")
include("exports.jl")

#####################################
# General functionality
#
include("default_kwargs.jl")
include("aliasstyle.jl")
include("abstractarray/set_types.jl")
include("abstractarray/to_shape.jl")
include("abstractarray/iscu.jl")
include("abstractarray/similar.jl")
include("abstractarray/ndims.jl")
include("abstractarray/mul.jl")
include("abstractarray/append.jl")
include("abstractarray/permutedims.jl")
include("abstractarray/fill.jl")
include("array/set_types.jl")
include("array/permutedims.jl")
include("array/mul.jl")
include("tupletools.jl")
include("emptynumber.jl")
include("nodata.jl")
include("tensorstorage/tensorstorage.jl")
include("tensorstorage/set_types.jl")
include("tensorstorage/default_storage.jl")
include("tensorstorage/similar.jl")
include("tensor/tensor.jl")
include("dims.jl")
include("tensor/set_types.jl")
include("tensor/similar.jl")
include("adapt.jl")
include("tensoroperations/generic_tensor_operations.jl")
include("tensoroperations/contraction_logic.jl")
include("abstractarray/tensoralgebra/contract.jl")

#####################################
# DenseTensor and DiagTensor
#
include("dense/dense.jl")
include("dense/densetensor.jl")
include("dense/tensoralgebra/contract.jl")
include("dense/linearalgebra/decompositions.jl")
include("dense/tensoralgebra/outer.jl")
include("dense/set_types.jl")
include("dense/fill.jl")
include("linearalgebra/symmetric.jl")
include("linearalgebra/linearalgebra.jl")
include("diag/diag.jl")
include("diag/set_types.jl")
include("diag/diagtensor.jl")
include("diag/similar.jl")
include("diag/tensoralgebra/contract.jl")
include("diag/tensoralgebra/outer.jl")
include("combiner/combiner.jl")
include("combiner/contract.jl")
include("truncate.jl")
include("linearalgebra/svd.jl")

#####################################
# BlockSparseTensor
#
include("blocksparse/blockdims.jl")
include("blocksparse/block.jl")
include("blocksparse/blockoffsets.jl")
include("blocksparse/blocksparse.jl")
include("blocksparse/blocksparsetensor.jl")
include("blocksparse/fermions.jl")
include("blocksparse/contract.jl")
include("blocksparse/contract_utilities.jl")
include("blocksparse/contract_generic.jl")
include("blocksparse/contract_sequential.jl")
include("blocksparse/contract_folds.jl")
include("blocksparse/contract_threads.jl")
include("blocksparse/diagblocksparse.jl")
include("blocksparse/similar.jl")
include("blocksparse/combiner.jl")
include("blocksparse/linearalgebra.jl")

#####################################
# Empty
#
include("empty/empty.jl")
include("empty/EmptyTensor.jl")
include("empty/tensoralgebra/contract.jl")
include("empty/adapt.jl")

#####################################
# Deprecations
#
include("deprecated.jl")

#####################################
# A global timer used with TimerOutputs.jl
#

const timer = TimerOutput()

#####################################
# Optional block sparse multithreading
#

blas_get_num_threads() = BLAS.get_num_threads()

const _using_threaded_blocksparse = Ref(false)

function enable_threaded_blocksparse_docstring(module_name)
  return """
             $(module_name).enable_threaded_blocksparse()
             $(module_name).disable_threaded_blocksparse()

         Enable or disable block sparse multithreading.

         Returns the current state of `$(module_name).using_threaded_blocksparse()`, i.e. `true` if threaded block sparse was previously enabled, and `false` if threaded block sparse was previously disabled. This is helpful for turning block sparse threading on or off temporarily. For example:
         ```julia
         using_threaded_blocksparse = $(module_name).enable_threaded_blocksparse()
         # Run code that you want to be threaded
         if !using_threaded_blocksparse
           $(module_name).disable_threaded_blocksparse()
         end
         ```

         Note that you need to start Julia with multiple threads. For example, to start Julia with 4 threads, you can use any of the following:
         ```
         \$ julia --threads=4

         \$ julia -t 4

         \$ JULIA_NUM_THREADS=4 julia
         ```

         In addition, we have found that it is best to disable `BLAS` and `Strided` multithreading when using block sparse multithreading. You can do that with the commands `using LinearAlgebra; BLAS.set_num_threads(1)` and `$(module_name).Strided.disable_threads()`.

         See also: `$(module_name).enable_threaded_blocksparse`, `$(module_name).disable_threaded_blocksparse`, `$(module_name).using_threaded_blocksparse`.
         """
end

function _enable_threaded_blocksparse()
  current_using_threaded_blocksparse = using_threaded_blocksparse()
  if !current_using_threaded_blocksparse
    if Threads.nthreads() == 1
      println(
        "WARNING: You are trying to enable block sparse multithreading, but you have started Julia with only a single thread. You can start Julia with `N` threads with `julia -t N`, and check the number of threads Julia can use with `Threads.nthreads()`. Your system has $(Sys.CPU_THREADS) threads available to use, which you can determine by running `Sys.CPU_THREADS`.\n",
      )
    end
    if BLAS.get_num_threads() > 1 && Threads.nthreads() > 1
      println(
        "WARNING: You are enabling block sparse multithreading, but your BLAS configuration $(BLAS.get_config()) is currently set to use $(BLAS.get_num_threads()) threads. When using block sparse multithreading, we recommend setting BLAS to use only a single thread, otherwise you may see suboptimal performance. You can set it with `using LinearAlgebra; BLAS.set_num_threads(1)`.\n",
      )
    end
    if Strided.get_num_threads() > 1
      println(
        "WARNING: You are enabling block sparse multithreading, but Strided.jl is currently set to use $(Strided.get_num_threads()) threads for performing dense tensor permutations. When using block sparse multithreading, we recommend setting Strided.jl to use only a single thread, otherwise you may see suboptimal performance. You can set it with `NDTensors.Strided.disable_threads()` and see the current number of threads it is using with `NDTensors.Strided.get_num_threads()`.\n",
      )
    end
    _using_threaded_blocksparse[] = true
  end
  return current_using_threaded_blocksparse
end

function _disable_threaded_blocksparse()
  current_using_threaded_blocksparse = using_threaded_blocksparse()
  if current_using_threaded_blocksparse
    _using_threaded_blocksparse[] = false
  end
  return current_using_threaded_blocksparse
end

"""
$(enable_threaded_blocksparse_docstring(@__MODULE__))
"""
using_threaded_blocksparse() = _using_threaded_blocksparse[]

"""
$(enable_threaded_blocksparse_docstring(@__MODULE__))
"""
enable_threaded_blocksparse() = _enable_threaded_blocksparse()

"""
$(enable_threaded_blocksparse_docstring(@__MODULE__))
"""
disable_threaded_blocksparse() = _disable_threaded_blocksparse()

#####################################
# Optional auto fermion system
#

const _using_auto_fermion = Ref(false)

using_auto_fermion() = _using_auto_fermion[]

function enable_auto_fermion()
  _using_auto_fermion[] = true
  return nothing
end

function disable_auto_fermion()
  _using_auto_fermion[] = false
  return nothing
end

#####################################
# Optional backends
#

const _using_tblis = Ref(false)

using_tblis() = _using_tblis[]

function enable_tblis()
  _using_tblis[] = true
  return nothing
end

function disable_tblis()
  _using_tblis[] = false
  return nothing
end

function backend_octavian end

using PackageExtensionCompat
function __init__()
  @require_extensions
end

end # module NDTensors
