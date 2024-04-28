module ITensorChainRules

using ITensors.Adapt
using ITensors.NDTensors
using ITensors.Ops

using ITensors: Indices

using ITensors.NDTensors: datatype

using ChainRulesCore
using ..ITensors

import ChainRulesCore: rrule

ITensors.dag(z::AbstractZero) = z

if VERSION < v"1.7"
  map_notangent(a) = map(_ -> NoTangent(), a)
else
  map_notangent(a) = map(Returns(NoTangent()), a)
end

include("projection.jl")
include(joinpath("NDTensors", "tensor.jl"))
include(joinpath("NDTensors", "dense.jl"))
include("indexset.jl")
include("itensor.jl")
include(joinpath("physics", "sitetype.jl"))
include(joinpath("mps", "abstractmps.jl"))
include(joinpath("mps", "mps.jl"))
include(joinpath("mps", "mpo.jl"))
include(joinpath("LazyApply", "LazyApply.jl"))
include("zygoterules.jl")

@non_differentiable map_notangent(::Any)
@non_differentiable Index(::Any...)
@non_differentiable delta(::Any...)
@non_differentiable dag(::Index)
@non_differentiable inds(::Any...)
@non_differentiable commoninds(::Any...)
@non_differentiable noncommoninds(::Any...)
@non_differentiable uniqueinds(::Any...)
@non_differentiable SiteType(::Any)
@non_differentiable ITensors._sitetypes(::Any)
@non_differentiable addtags(::TagSet, ::Any)
@non_differentiable ITensors.filter_inds_set_function(::Function, ::Function, ::Any...)
@non_differentiable ITensors.filter_inds_set_function(::Function, ::Any...)
@non_differentiable ITensors.indpairs(::Any...)
@non_differentiable onehot(::Any...)
@non_differentiable Base.convert(::Type{TagSet}, str::String)

end
