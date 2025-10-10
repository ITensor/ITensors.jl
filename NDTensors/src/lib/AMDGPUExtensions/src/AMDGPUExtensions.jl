module AMDGPUExtensions

module Vendored
    include(joinpath(
        "..", "..", "..", "vendored", "TypeParameterAccessors", "src",
        "TypeParameterAccessors.jl",
    ))
end

include("roc.jl")

end
