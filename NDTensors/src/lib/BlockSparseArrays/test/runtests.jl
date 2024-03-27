@eval module $(gensym())
include("test_basics.jl")
include("../ext/BlockSparseArraysTensorAlgebraExt/test/runtests.jl")
include("../ext/BlockSparseArraysGradedAxesExt/test/runtests.jl")
end
