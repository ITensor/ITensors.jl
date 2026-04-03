@eval module $(gensym())
using NDTensors.GPUArraysCoreExtensions: storagemode
using Test: @test, @testset
@testset "Test Base" begin
    @test storagemode isa Function
end
end
