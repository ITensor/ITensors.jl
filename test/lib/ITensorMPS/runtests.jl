@eval module $(gensym())
using ITensors: ITensors
using Test: @testset
@testset "ITensorMPS tests" begin
  include(joinpath(pkgdir(ITensors), "src", "lib", "ITensorMPS", "test", "runtests.jl"))
end
end
