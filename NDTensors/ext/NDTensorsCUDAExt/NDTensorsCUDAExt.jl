module NDTensorsCUDAExt

module Vendored
    include(
        joinpath(
            "..", "..", "src", "vendored", "TypeParameterAccessors", "ext",
            "TypeParameterAccessorsCUDAExt.jl"
        )
    )
end

include("append.jl")
include("default_kwargs.jl")
include("copyto.jl")
include("set_types.jl")
include("iscu.jl")
include("adapt.jl")
include("indexing.jl")
include("linearalgebra.jl")
include("mul.jl")
include("permutedims.jl")

end
