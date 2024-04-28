"""
ITensor is a library for rapidly creating correct and efficient tensor network algorithms.

An ITensor is a tensor whose interface is independent of its memory layout.
ITensor indices are 'intelligent' meaning they carry extra information and
'recognize' each other automatically when contracting or adding ITensors.

The ITensor library includes composable and extensible algorithms for optimizing
and transforming tensor networks, such as matrix product state and matrix product operators.

# Example Usage

Define tensor indices i and j

    i = Index(2, "i")
    j = Index(3, "j")

Make an ITensor with these indices

    A = ITensor(i,j)

Set the i==2,j==1 element to -2.6

    A[j=>1,i=>2] = -2.6
    A[i=>2,j=>1] = -2.6 #this has the same effect

Make an ITensor with random elements

    B = randomITensor(j,i)

Add ITensors A and B together (ok that indices in different order)

    C = A + B

# Other Features of ITensor

  - Tools for **tensor networks**, such as matrix product states (MPS) / tensor trains (TT)
  - **Algorithms** for solving linear equations in MPS form (such as DMRG) or
    for integrating differential equations ("time evolving MPS")
  - ITensors can have **sparse data** internally, such as block sparsity or diagonal
    sparsity, while having the same interface as dense ITensors
  - ITensors can have **symmetry properties** (invariance or equivariance) under
    group transformations of the indices. In physics terminology such ITensors conserve quantum numbers.

# Documentation and Resources

ITensor website: https://itensor.org/

Documentation: https://itensor.github.io/ITensors.jl/stable/
"""
module ITensors
include("usings.jl")
include("utils.jl")
include("lib/ContractionSequenceOptimization/ContractionSequenceOptimization.jl")
using .ContractionSequenceOptimization
include("lib/LazyApply/LazyApply.jl")
using .LazyApply
include("lib/Ops/Ops.jl")
using .Ops
import .Ops: sites, name
include("exports.jl")
include("imports.jl")
include("global_variables.jl")
include("lastval.jl")
include("smallstring.jl")
include("readwrite.jl")
include("not.jl")
include("tagset.jl")
include("arrow.jl")
include("symmetrystyle.jl")
include("index.jl")
include("set_operations.jl")
include("indexset.jl")
include("itensor.jl")
include("oneitensor.jl")
include("tensor_operations/tensor_algebra.jl")
include("tensor_operations/matrix_algebra.jl")
include("tensor_operations/permutations.jl")
include("broadcast.jl")
include("tensor_operations/matrix_decomposition.jl")
include("adapt.jl")
include("set_types.jl")
include("tensor_operations/itensor_combiner.jl")
include("qn/flux.jl")
include("qn/qn.jl")
include("qn/qnindex.jl")
include("qn/qnindexset.jl")
include("qn/qnitensor.jl")
include("nullspace.jl")
include("lib/Ops/ops_itensor.jl")
include("physics/sitetype.jl")
include("physics/lattices.jl")
include("physics/site_types/aliases.jl")
include("physics/site_types/generic_sites.jl")
include("physics/site_types/qubit.jl")
include("physics/site_types/spinhalf.jl")
include("physics/site_types/spinone.jl")
include("physics/site_types/fermion.jl")
include("physics/site_types/electron.jl")
include("physics/site_types/tj.jl")
include("physics/site_types/qudit.jl")
include("physics/site_types/boson.jl")
include("physics/fermions.jl")
include("lib/ITensorMPS/ITensorMPS.jl")
@reexport using .ITensorMPS
include("lib/ITensorsNamedDimsArraysExt/src/ITensorsNamedDimsArraysExt.jl")
using .ITensorsNamedDimsArraysExt: ITensorsNamedDimsArraysExt
include("lib/Ops/trotter.jl")
include("lib/ITensorChainRules/ITensorChainRules.jl")
include("lib/ITensorNetworkMaps/ITensorNetworkMaps.jl")
include("lib/ITensorVisualizationCore/ITensorVisualizationCore.jl")
using .ITensorVisualizationCore
include("deprecated.jl")
include("argsdict/argsdict.jl")
include("packagecompile/compile.jl")
include("developer_tools.jl")

using PackageExtensionCompat: @require_extensions
function __init__()
  @require_extensions
  return resize!(empty!(INDEX_ID_RNGs), Threads.nthreads()) # ensures that we didn't save a bad object
end
end
