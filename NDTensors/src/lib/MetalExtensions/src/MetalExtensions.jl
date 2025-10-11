module MetalExtensions

module Vendored
    include(
        joinpath(
            "..", "..", "..", "vendored", "TypeParameterAccessors", "src",
            "TypeParameterAccessors.jl",
        )
    )
end

include("metal.jl")

end
