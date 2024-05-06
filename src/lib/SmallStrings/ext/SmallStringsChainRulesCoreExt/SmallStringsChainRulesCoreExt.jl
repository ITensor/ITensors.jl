module SmallStringsChainRulesCoreExt
using ChainRulesCore: @non_differentiable
using ...SmallStrings: SmallString
@non_differentiable SmallString(::Any)
end
