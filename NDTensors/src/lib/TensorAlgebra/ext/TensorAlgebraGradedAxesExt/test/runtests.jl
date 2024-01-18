@eval module $(gensym())
using Test: @testset
@testset "TensorAlgebraGradedAxesExt" begin
  include("test_basics.jl")
  include("test_contract.jl")
end
end
