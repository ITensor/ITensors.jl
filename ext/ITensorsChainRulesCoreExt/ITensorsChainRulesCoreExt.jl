module ITensorsChainRulesCoreExt
import ChainRulesCore: rrule
using ChainRulesCore
using ITensors
using ITensors.Adapt
using ITensors.NDTensors
using ITensors.NDTensors: datatype
using ITensors.Ops
using ITensors: Indices
include("utils.jl")
include("projection.jl")
include("NDTensors/tensor.jl")
include("NDTensors/dense.jl")
include("indexset.jl")
include("itensor.jl")
include("LazyApply/LazyApply.jl")
include("non_differentiable.jl")
include("smallstrings.jl")
end
