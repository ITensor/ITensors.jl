@eval module $(gensym())
include("basics.jl")
include("../ext/BlockSparseArraysTensorAlgebraExt/test/runtest.jl")
end
