@eval module $(gensym())
using Test: @testset, @test
using NDTensors.MetalExtensions
@testset "mtl function exists" begin
  @test mtl isa Function
end
end
