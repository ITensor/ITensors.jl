@eval module $(gensym())
include("basics.jl")
include("../ext/BlockSparseArraysTensorAlgebraExt/test/runtests.jl")
include("../ext/BlockSparseArraysGradedAxesExt/test/runtests.jl")
end
