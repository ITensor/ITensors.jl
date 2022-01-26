using Test
using NDTensors

@testset "Test non-exported tuple tools" begin
  @test NDTensors.diff((1, 3, 6, 4)) == (2, 3, -2)
  @test NDTensors.diff((1, 2, 3)) == (1, 1)
end

nothing
