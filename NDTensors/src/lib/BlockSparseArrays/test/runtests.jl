@eval module $(gensym())
include("basics.jl")
include("../ext/BlockSparseArraysTensorAlgebraExt/test/runtest.jl")
include("../ext/BlockSparseArraysGradedAxesExt/test/runtest.jl")
end
