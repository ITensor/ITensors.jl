module NDTensorsMetalExt

module Vendored
    include(
        joinpath(
            "..", "..", "src", "vendored", "TypeParameterAccessors", "ext",
            "TypeParameterAccessorsMetalExt.jl"
        )
    )
end

include("adapt.jl")
include("set_types.jl")
include("indexing.jl")
include("linearalgebra.jl")
include("copyto.jl")
include("append.jl")
include("permutedims.jl")
include("mul.jl")

end
