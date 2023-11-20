using Dictionaries
using NDTensors
using Test

@testset "UniformDiagBlockSparseTensor basic functionality" begin
  NeverAlias = NDTensors.NeverAlias
  AllowAlias = NDTensors.AllowAlias

  storage = DiagBlockSparse(1.0, Dictionary([Block(1, 1), Block(2, 2)], [0, 1]))
  tensor = Tensor(storage, ([1, 1], [1, 1]))

  @test conj(tensor) == tensor
  @test conj(NeverAlias(), tensor) == tensor
  @test conj(AllowAlias(), tensor) == tensor

  c = 1 + 2im
  tensor *= c

  @test tensor[1, 1] == c
  @test conj(tensor) ≠ tensor
  @test conj(NeverAlias(), tensor) ≠ tensor
  @test conj(AllowAlias(), tensor) ≠ tensor
  @test conj(tensor)[1, 1] == conj(c)
  @test conj(NeverAlias(), tensor)[1, 1] == conj(c)
  @test conj(AllowAlias(), tensor)[1, 1] == conj(c)
end
