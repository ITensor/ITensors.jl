module ITensorChainRules

using ITensors.NDTensors
using ITensors: Indices

using ChainRulesCore
using ..ITensors

import ChainRulesCore: rrule

ITensors.dag(z::AbstractZero) = z
broadcast_notangent(a) = broadcast(_ -> NoTangent(), a)

include("zygoterules.jl")
include("indexset.jl")
include("itensor.jl")
include("mps/abstractmps.jl")
include("mps/mpo.jl")

@non_differentiable broadcast_notangent(::Any)
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
@non_differentiable has_fermion_string(::AbstractString, ::Index)
@non_differentiable permute(::Indices, ::Indices)
@non_differentiable combiner(::Indices)
@non_differentiable ITensors.filter_inds_set_function(::Function, ::Function, ::Any...)
@non_differentiable ITensors.filter_inds_set_function(::Function, ::Any...)

end
