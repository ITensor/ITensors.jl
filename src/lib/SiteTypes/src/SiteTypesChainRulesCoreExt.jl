module SiteTypesChainRulesCoreExt
using ..SiteTypes: SiteType, _sitetypes, has_fermion_string
using ChainRulesCore: @non_differentiable
@non_differentiable has_fermion_string(::AbstractString, ::Any)
@non_differentiable SiteType(::Any)
@non_differentiable _sitetypes(::Any)
end
