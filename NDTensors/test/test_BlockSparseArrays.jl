@eval module $(gensym())
using NDTensors

include(joinpath(pkgdir(NDTensors), "src", "BlockSparseArrays", "test", "runtests.jl"))
end