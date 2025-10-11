module RankFactorization

module Vendored
    include(
        joinpath(
            "..", "..", "..", "vendored", "TypeParameterAccessors", "src",
            "TypeParameterAccessors.jl",
        )
    )
end

include("default_kwargs.jl")
include("truncate_spectrum.jl")
include("spectrum.jl")

end
