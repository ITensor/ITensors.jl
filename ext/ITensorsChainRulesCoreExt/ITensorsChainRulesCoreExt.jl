module ITensorsChainRulesCoreExt
using ChainRulesCore
import ChainRulesCore: rrule
using ITensors
using ITensors: Indices
using ITensors.Adapt
using ITensors.NDTensors
using ITensors.NDTensors: datatype
using ITensors.Ops
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
