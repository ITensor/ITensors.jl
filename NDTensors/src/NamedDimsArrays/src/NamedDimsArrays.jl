module NamedDimsArrays
include("BaseExtension/BaseExtension.jl")

include("traits.jl")
include("abstractnamedint.jl")
include("abstractnamedunitrange.jl")
include("abstractnameddimsarray.jl")
include("namedint.jl")
include("namedunitrange.jl")
include("nameddimsarray.jl")
include("constructors.jl")
include("tensoralgebra.jl")

# Extensions
include("../ext/NamedDimsArraysAdaptExt/src/NamedDimsArraysAdaptExt.jl")
include("../ext/NamedDimsArraysTensorAlgebraExt/src/NamedDimsArraysTensorAlgebraExt.jl")
end
