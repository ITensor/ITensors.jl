using ITensors
using ITensors.NDTensors
using Test

@testset "NDTensors compatibility" begin
  i = Index([QN(0) => 1, QN(1) => 1])

  T = BlockSparseTensor(Float64, [Block(1, 1)], (i', dag(i)))
  @test nnzblocks(T) == 1
  @test nzblocks(T) == [Block(1, 1)]

  T = BlockSparseTensor(Float64, [Block(1, 1)], [i', dag(i)])
  @test nnzblocks(T) == 1
  @test nzblocks(T) == [Block(1, 1)]

  T = BlockSparseTensor(Float64, [Block(1, 1)], IndexSet(i', dag(i)))
  @test nnzblocks(T) == 1
  @test nzblocks(T) == [Block(1, 1)]

  @testset "blockdim" begin
    i = Index(2)
    @test_throws ErrorException blockdim(i, Block(1))
    @test_throws ErrorException blockdim(i, 1)
    @test_throws ErrorException blockdim(1, Block(1))
    @test_throws ErrorException blockdim(1, 1)
  end
end
