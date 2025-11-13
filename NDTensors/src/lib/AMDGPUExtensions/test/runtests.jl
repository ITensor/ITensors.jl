@eval module $(gensym())
using Test: @testset, @test
using NDTensors.AMDGPUExtensions: roc, ROCArrayAdaptor
using NDTensors.GPUArraysCoreExtensions: storagemode
@testset "roc and ROCArrayAdaptor" begin
    @test roc isa Function
    @test storagemode(ROCArrayAdaptor{1}) == 1
end
end
