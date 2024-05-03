module ITensorsChainRulesCoreExt
## using ITensors.Adapt
## using ITensors.NDTensors
## using ITensors.Ops
##
## using ITensors: Indices
##
## using ITensors.NDTensors: datatype
##
## using ChainRulesCore
## using ..ITensors
##
## import ChainRulesCore: rrule

include("utils.jl")
include("projection.jl")
include("NDTensors/tensor.jl")
include("NDTensors/dense.jl")
include("indexset.jl")
include("itensor.jl")
include("LazyApply/LazyApply.jl")
include("non_differentiable.jl")
include("itensormps.jl")
end
