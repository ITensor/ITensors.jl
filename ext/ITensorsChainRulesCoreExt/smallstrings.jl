using ITensors: ITensors
include(
    joinpath(
        pkgdir(ITensors),
        "src",
        "lib",
        "SmallStrings",
        "ext",
        "SmallStringsChainRulesCoreExt",
        "SmallStringsChainRulesCoreExt.jl",
    ),
)
