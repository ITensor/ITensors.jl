@eval module $(gensym())
using NDTensors.AMDGPUExtensions: ROCArrayAdaptor, roc
using NDTensors.GPUArraysCoreExtensions: storagemode
using Test: @test, @testset
@testset "roc and ROCArrayAdaptor" begin
    @test roc isa Function
    @test storagemode(ROCArrayAdaptor{1}) == 1
end
end
