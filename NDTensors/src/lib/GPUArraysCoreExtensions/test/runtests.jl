@eval module $(gensym())
using Test: @testset, @test
using NDTensors.GPUArraysCoreExtensions: storagemode
@testset "Test Base" begin
    @test storagemode isa Function
end
end
