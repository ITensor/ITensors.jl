module NamedDimsArrays
include("traits.jl")
include("abstractnamedint.jl")
include("abstractnamedunitrange.jl")
include("abstractnameddimsarray.jl")
include("namedint.jl")
include("namedunitrange.jl")
include("nameddimsarray.jl")

# Extensions
include("../ext/NamedDimsArraysTensorAlgebraExt/src/NamedDimsArraysTensorAlgebraExt.jl")
end
