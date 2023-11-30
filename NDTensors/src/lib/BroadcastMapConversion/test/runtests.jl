@eval module $(gensym())
using Test: @test, @testset
using NDTensors.BroadcastMapConversion: map_function, map_args
@testset "BroadcastMapConversion" begin
  using Base.Broadcast: Broadcasted
  c = 2.2
  a = randn(2, 3)
  b = randn(2, 3)
  bc = Broadcasted(*, (c, a))
  @test copy(bc) ≈ c * a ≈ map(map_function(bc), map_args(bc)...)
  bc = Broadcasted(+, (a, b))
  @test copy(bc) ≈ a + b ≈ map(map_function(bc), map_args(bc)...)
end
end
