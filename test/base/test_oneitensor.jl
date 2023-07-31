using ITensors
using Test

@testset "OneITensor" begin
  let i = Index(2), it = ITensor(i), OneITensor = ITensors.OneITensor
    @test OneITensor() isa OneITensor
    @test inds(OneITensor()) == ()
    @test eltype(OneITensor()) <: Bool
    @test isone(dim(OneITensor()))
    @test ITensors.isoneitensor(OneITensor())
    @test !ITensors.isoneitensor(it)
    @test dag(OneITensor()) == OneITensor()
    @test OneITensor() * it == it
    @test it * OneITensor() == it
    @test *(OneITensor()) == OneITensor()
    @test contract([it, OneITensor(), OneITensor()]) == it
  end
end
