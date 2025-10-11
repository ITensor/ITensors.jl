module GPUArraysCoreExtensions

module Vendored
    include(
        joinpath(
            "..", "..", "..", "vendored", "TypeParameterAccessors", "src",
            "TypeParameterAccessors.jl",
        )
    )
end

include("gpuarrayscore.jl")

end
