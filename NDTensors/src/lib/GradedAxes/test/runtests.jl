@eval module $(gensym())
using Test: @testset
@testset "GradedAxes" begin
  include("test_basics.jl")
  include("test_tensor_product.jl")
  include("test_dual.jl")
end
end
