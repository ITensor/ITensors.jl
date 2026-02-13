module SmallStringsChainRulesCoreExt
using ...SmallStrings: SmallString
using ChainRulesCore: @non_differentiable
@non_differentiable SmallString(::Any)
end
