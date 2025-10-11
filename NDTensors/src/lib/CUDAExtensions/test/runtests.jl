@eval module $(gensym())
using Test: @testset, @test
using NDTensors.CUDAExtensions: cu, CuArrayAdaptor
using NDTensors.GPUArraysCoreExtensions: storagemode
@testset "cu function exists" begin
    @test cu isa Function
    @test storagemode(CuArrayAdaptor{1}) == 1
end
end
