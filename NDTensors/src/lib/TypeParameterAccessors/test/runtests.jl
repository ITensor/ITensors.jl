@eval module $(gensym())
using Test: @testset
@testset "TypeParameterAccessors.jl" begin
  include("test_basics.jl")
  include("test_defaults.jl")
  include("test_custom_types.jl")
  include("test_wrappers.jl")
  include("test_similartype.jl")
end
end
