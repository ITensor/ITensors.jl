@eval module $(gensym())
using Test: @testset, @test
using NDTensors.MetalExtensions: mtl
@testset "mtl function exists" begin
    @test mtl isa Function
end
end
