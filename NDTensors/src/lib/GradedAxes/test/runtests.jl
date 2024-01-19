@eval module $(gensym())
using Test: @testset
@testset "GradedAxes" begin
  include("test_basics.jl")
  include("../ext/GradedAxesSectorsExt/test/runtests.jl")
end
end
