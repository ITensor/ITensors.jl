@eval module $(gensym())
using NDTensors.CUDAExtensions: CuArrayAdaptor, cu
using NDTensors.GPUArraysCoreExtensions: storagemode
using Test: @test, @testset
@testset "cu function exists" begin
    @test cu isa Function
    @test storagemode(CuArrayAdaptor{1}) == 1
end
end
