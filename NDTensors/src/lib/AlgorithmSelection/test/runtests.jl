@eval module $(gensym())
using Test: @test, @testset
using NDTensors.AlgorithmSelection: Algorithm, @Algorithm_str
@testset "AlgorithmSelection" begin
  @test Algorithm"alg"() isa Algorithm{:alg}
  @test Algorithm("alg") isa Algorithm{:alg}
  @test Algorithm(:alg) isa Algorithm{:alg}
  alg = Algorithm"alg"(; x=2, y=3)
  @test alg isa Algorithm{:alg}
  @test alg.kwargs == (; x=2, y=3)
end
end
