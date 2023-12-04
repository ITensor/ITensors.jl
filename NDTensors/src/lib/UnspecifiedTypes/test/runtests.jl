## TODO This needs work here
@eval module $(gensym())
using NDTensors.UnspecifiedTypes
using Test: @test, @testset

@testset "Testing UnspecifiedTypes" begin
  UA = UnspecifiedArray{}
end
end
