@eval module $(gensym())
include("test_basics.jl")
include("../ext/TensorAlgebraGradedAxesExt/test/runtests.jl")
end
