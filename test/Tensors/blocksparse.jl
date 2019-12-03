using ITensors,
      Test

@testset "BlockSparseTensor basic functionality" begin

  # Indices
  indsA = ([2,3],[4,5])

  # Locations of non-zero blocks
  locs = [(1,2),(2,1)]

  A = BlockSparseTensor(locs,indsA...)
  randn!(A)

  @test blockdims(A,(1,2)) == (2,5)
  @test blockdims(A,(2,1)) == (3,4)
  @test nnzblocks(A) == 2
  @test nnz(A) == 2*5+3*4
  @test inds(A) == ([2,3],[4,5])
  @test isblocknz(A,(2,1))
  @test isblocknz(A,(1,2))
  @test !isblocknz(A,(1,1))
  @test !isblocknz(A,(2,2))
  @test findblock(A,(2,1))==1
  @test findblock(A,(1,2))==2
  @test isnothing(findblock(A,(1,1)))
  @test isnothing(findblock(A,(2,2)))

  # Test different ways of getting nnz
  @test nnz(blockoffsets(A),inds(A)) == nnz(A)

  A[1,5] = 15
  A[2,5] = 25

  @test A[1,1] == 0
  @test A[1,5] == 15
  @test A[2,5] == 25

  D = dense(A)

  @test D == A

  for I in CartesianIndices(A)
    @test D[I] == A[I]
  end

  A12 = blockview(A,(1,2))

  @test dims(A12) == (2,5)

  for I in CartesianIndices(A12)
    blockstart = CartesianIndex(0,4)
    @test A12[I] == A[I+blockstart]
  end

  B = BlockSparseTensor(undef,locs,indsA)
  randn!(B)

  C = A+B

  for I in CartesianIndices(C)
    @test C[I] == A[I]+B[I]
  end

  Ap = permutedims(A,(2,1))

  @test blockdims(Ap,(1,2)) == (4,3)
  @test blockdims(Ap,(2,1)) == (5,2)
  @test nnz(A) == nnz(Ap)
  @test nnzblocks(A) == nnzblocks(Ap)

  for I in CartesianIndices(C)
    @test A[I] == Ap[permute(I,(2,1))]
  end

  @testset "BlockSparseTensor setindex! add block" begin
    T = BlockSparseTensor([2,3],[4,5])
    #T[1,1] = 2.0
  end
 
end

