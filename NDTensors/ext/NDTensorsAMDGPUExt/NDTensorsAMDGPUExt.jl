module NDTensorsAMDGPUExt

module Vendored
    include(
        joinpath(
            "..", "..", "src", "vendored", "TypeParameterAccessors", "ext",
            "TypeParameterAccessorsAMDGPUExt.jl"
        )
    )
end

include("append.jl")
include("copyto.jl")
include("set_types.jl")
include("adapt.jl")
include("indexing.jl")
include("linearalgebra.jl")
include("mul.jl")
include("permutedims.jl")

end
