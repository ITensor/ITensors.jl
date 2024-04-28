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
include("lib/ContractionSequenceOptimization/src/ContractionSequenceOptimization.jl")
# TODO: `using .ContractionSequenceOptimization: ContractionSequenceOptimization`.
using .ContractionSequenceOptimization
include("lib/LazyApply/src/LazyApply.jl")
# TODO: `using .LazyApply: LazyApply`.
using .LazyApply
include("lib/Ops/src/Ops.jl")
# TODO: `using .Ops: Ops`.
using .Ops
import .Ops: sites, name
include("exports.jl")
include("imports.jl")
include("global_variables.jl")
# TODO: Move to `lib/LastVals/src/LastVals.jl`.
include("lastval.jl")
include("lib/SmallStrings/src/SmallStrings.jl")
using .SmallStrings: SmallStrings, IntChar, Tag, isint, isnull
export Tag
include("readwrite.jl")
# TODO: Move to `lib/Nots/src/Nots.jl`.
include("not.jl")
include("lib/TagSets/src/TagSets.jl")
using .TagSets: TagSets, set_strict_tags!, using_strict_tags
include("arrow.jl")
include("symmetrystyle.jl")
include("index.jl")
include("set_operations.jl")
include("indexset.jl")
include("itensor.jl")
include("val.jl")
export val
include("qn/flux.jl")
# TODO: Move to `lib/QuantumNumbers/src/QuantumNumbers.jl`.
include("qn/qn.jl")
include("oneitensor.jl")
include("tensor_operations/tensor_algebra.jl")
include("tensor_operations/matrix_algebra.jl")
include("tensor_operations/permutations.jl")
include("lib/SiteTypes/src/SiteTypes.jl")
using .SiteTypes:
  SiteTypes,
  OpName,
  SiteType,
  StateName,
  TagType,
  ValName,
  @OpName_str,
  @SiteType_str,
  @StateName_str,
  @TagType_str,
  @ValName_str,
  alias,
  has_fermion_string,
  op,
  op!,
  ops,
  state
export OpName,
  SiteType,
  StateName,
  TagType,
  ValName,
  @OpName_str,
  @SiteType_str,
  @StateName_str,
  @TagType_str,
  @ValName_str,
  has_fermion_string,
  op,
  ops,
  state,
  val
# TODO: Move to `lib/ITensorsSiteTypesExt/src/ITensorsSiteTypesExt.jl`.
include("ITensorsSiteTypesExt.jl")
include("broadcast.jl")
include("tensor_operations/matrix_decomposition.jl")
include("adapt.jl")
include("set_types.jl")
include("tensor_operations/itensor_combiner.jl")
include("qn/qnindex.jl")
include("qn/qnindexset.jl")
include("qn/qnitensor.jl")
include("nullspace.jl")

# TODO: Move to `lib/ITensorsOpsExt/src/ITensorsOpsExt.jl`?
include("lib/Ops/ops_itensor.jl")
include("fermions/fermions.jl")

include("lib/ITensorMPS/src/ITensorMPS.jl")
# TODO: `using .ITensorMPS: ITensorMPS, ...`.
@reexport using .ITensorMPS
include("lib/ITensorsNamedDimsArraysExt/src/ITensorsNamedDimsArraysExt.jl")
using .ITensorsNamedDimsArraysExt: ITensorsNamedDimsArraysExt

# TODO: Move into `Ops`.
include("lib/Ops/trotter.jl")

include("lib/ITensorChainRules/src/ITensorChainRules.jl")
include("lib/ITensorNetworkMaps/src/ITensorNetworkMaps.jl")
include("lib/ITensorVisualizationCore/src/ITensorVisualizationCore.jl")
# TODO: `using .ITensorVisualizationCore: ITensorVisualizationCore`.
using .ITensorVisualizationCore
using .ITensorVisualizationCore:
  @visualize,
  @visualize!,
  @visualize_noeval,
  @visualize_noeval!,
  @visualize_sequence,
  @visualize_sequence_noeval
export @visualize,
  @visualize!,
  @visualize_noeval,
  @visualize_noeval!,
  @visualize_sequence,
  @visualize_sequence_noeval
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
