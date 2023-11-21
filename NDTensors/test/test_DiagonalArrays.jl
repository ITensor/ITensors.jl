@eval module $(gensym())
using NDTensors

include(joinpath(pkgdir(NDTensors), "src", "DiagonalArrays", "test", "runtests.jl"))
end
