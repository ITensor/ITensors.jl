@eval module $(gensym())
using Test: @testset
@testset "TypeParameterAccessors.jl" begin
  include("test_basics.jl")
  include("test_defaults.jl")
  include("test_custom_types.jl")
  include("test_wrappers.jl")
end
end
