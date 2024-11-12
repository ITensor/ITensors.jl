@eval module $(gensym())
using ITensors: ITensors
using Suppressor: @suppress
using Test: @testset
@testset "examples" begin
  @suppress include(
    joinpath(
      pkgdir(ITensors),
      "src",
      "lib",
      "ITensorsNamedDimsArraysExt",
      "examples",
      "example_readme.jl",
    ),
  )
end
end
