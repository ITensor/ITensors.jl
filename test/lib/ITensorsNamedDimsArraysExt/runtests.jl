@eval module $(gensym())
using ITensors: ITensors
include(
  joinpath(
    pkgdir(ITensors), "src", "lib", "ITensorsNamedDimsArraysExt", "test", "runtests.jl"
  ),
)
end
