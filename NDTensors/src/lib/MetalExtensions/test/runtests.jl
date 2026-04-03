@eval module $(gensym())
using NDTensors.MetalExtensions: mtl
using Test: @test, @testset
@testset "mtl function exists" begin
    @test mtl isa Function
end
end
