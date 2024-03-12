@eval module $(gensym())
using Test: @testset, @test
using NDTensors.CUDAExtensions
@testset "cu function exists" begin
  @test cu isa Function
end
end
