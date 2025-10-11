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

    B = random_itensor(j,i)

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
include("lib/LazyApply/src/LazyApply.jl")
# TODO: `using .LazyApply: LazyApply, ...`.
using .LazyApply
using .LazyApply: Prod, Scaled, Sum, coefficient
include("lib/Ops/src/Ops.jl")
# TODO: `using .Ops: Ops, ...`.
using .Ops
using .Ops: Ops, Op, Trotter
import .Ops: sites, name
include("exports.jl")
include("imports.jl")
include("global_variables.jl")
# TODO: Move to `lib/LastVals/src/LastVals.jl`.
include("lastval.jl")
include("lib/SmallStrings/src/SmallStrings.jl")
using .SmallStrings: SmallStrings, IntChar, SmallString, Tag, isint, isnull
include("readwrite.jl")
export readcpp
# TODO: Move to `lib/Nots/src/Nots.jl`.
include("not.jl")
export not
include("lib/TagSets/src/TagSets.jl")
using .TagSets: TagSets, set_strict_tags!, using_strict_tags
# TODO: Move to `lib/Names/src/Names.jl`.
include("name.jl")
# TODO: Move to `lib/Vals/src/Vals.jl`.
include("val.jl")
export val
include("lib/QuantumNumbers/src/QuantumNumbers.jl")
using .QuantumNumbers:
    Arrow,
    In,
    Neither,
    Out,
    QN,
    QNVal,
    hasname,
    have_same_mods,
    have_same_qns,
    isactive,
    maxQNs,
    modulus,
    nactive
export QN, isactive, modulus
include("symmetrystyle.jl")
include("index.jl")
include("set_operations.jl")
include("indexset.jl")
include("itensor.jl")
include("qn/flux.jl")
include("oneitensor.jl")
include("tensor_operations/contraction_cost.jl")
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
    siteind,
    siteinds,
    state
include("lib/ITensorsSiteTypesExt/src/ITensorsSiteTypesExt.jl")
include("broadcast.jl")
include("tensor_operations/matrix_decomposition.jl")
include("adapt.jl")
include("set_types.jl")
include("tensor_operations/itensor_combiner.jl")
include("qn/qnindex.jl")
include("qn/qnindexset.jl")
include("qn/qnitensor.jl")
include("nullspace.jl")
include("lib/ITensorsOpsExt/src/ITensorsOpsExt.jl")
include("fermions/fermions.jl")
export fparity, isfermionic
include("../ext/ITensorsChainRulesCoreExt/ITensorsChainRulesCoreExt.jl")
include("lib/ITensorVisualizationCore/src/ITensorVisualizationCore.jl")
# TODO: `using .ITensorVisualizationCore: ITensorVisualizationCore, ...`.
using .ITensorVisualizationCore
using .ITensorVisualizationCore:
    @visualize,
    @visualize!,
    @visualize_noeval,
    @visualize_noeval!,
    @visualize_sequence,
    @visualize_sequence_noeval
include("deprecated.jl")
include("argsdict/argsdict.jl")
include("packagecompile/compile.jl")
include("developer_tools.jl")
end
