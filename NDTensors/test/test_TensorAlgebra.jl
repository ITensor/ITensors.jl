@eval module $(gensym())
using NDTensors

include(joinpath(pkgdir(NDTensors), "src", "TensorAlgebra", "test", "runtests.jl"))
end