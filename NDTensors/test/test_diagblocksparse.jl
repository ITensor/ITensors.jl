@eval module $(gensym())
using Dictionaries: Dictionary
using NDTensors:
  NDTensors,
  Block,
  BlockSparseTensor,
  DiagBlockSparse,
  Tensor,
  blockoffsets,
  contract,
  nzblocks
using Random: randn!
using Test: @test, @test_throws, @testset
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
@testset "DiagBlockSparse off-diagonal (eltype=$elt)" for elt in (
  Float32, Float64, Complex{Float32}, Complex{Float64}
)
  inds1 = ([1, 1], [1, 1])
  inds2 = ([1, 1], [1, 1])
  blocks = [(1, 2), (2, 1)]
  a1 = BlockSparseTensor{elt}(blocks, inds1...)
  for b in nzblocks(a1)
    randn!(a1[b])
  end
  a2 = Tensor(DiagBlockSparse(one(elt), blockoffsets(a1)), inds2)
  for (labels1, labels2) in (((1, -1), (-1, 2)), ((-1, -2), (-1, -2)))
    @test_throws ErrorException contract(a1, labels1, a2, labels2)
  end
end
end
