using ChainRulesCore: @non_differentiable
using ITensors:
    ITensors, Index, addtags, commoninds, dag, delta, inds, noncommoninds, onehot, uniqueinds
using ITensors.TagSets: TagSet

@non_differentiable map_notangent(::Any)
@non_differentiable Index(::Any...)
@non_differentiable delta(::Any...)
@non_differentiable dag(::Index)
@non_differentiable inds(::Any...)
@non_differentiable commoninds(::Any...)
@non_differentiable noncommoninds(::Any...)
@non_differentiable uniqueinds(::Any...)
@non_differentiable addtags(::TagSet, ::Any)
@non_differentiable ITensors.filter_inds_set_function(::Function, ::Function, ::Any...)
@non_differentiable ITensors.filter_inds_set_function(::Function, ::Any...)
@non_differentiable ITensors.indpairs(::Any...)
@non_differentiable onehot(::Any...)
@non_differentiable Base.convert(::Type{TagSet}, str::String)
