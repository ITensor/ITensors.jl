module CUDAExtensions

module Vendored
    include(joinpath(
        "..", "..", "..", "vendored", "TypeParameterAccessors", "src",
        "TypeParameterAccessors.jl",
    ))
end

include("cuda.jl")

end
