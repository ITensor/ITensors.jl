module NDTensorsJLArraysExt

module Vendored
    include(
        joinpath(
            "..", "..", "src", "vendored", "TypeParameterAccessors", "ext",
            "TypeParameterAccessorsJLArraysExt.jl"
        )
    )
end

include("copyto.jl")
include("indexing.jl")
include("linearalgebra.jl")
include("mul.jl")
include("permutedims.jl")

end
