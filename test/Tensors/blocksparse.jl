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

  for I in eachindex(A)
    @test D[I] == A[I]
  end

  A12 = blockview(A,(1,2))

  @test dims(A12) == (2,5)

  for I in eachindex(A12)
    @test A12[I] == A[I+CartesianIndex(0,4)]
  end

  B = BlockSparseTensor(undef,locs,indsA)
  randn!(B)

  C = A+B

  for I in eachindex(C)
    @test C[I] == A[I]+B[I]
  end

  Ap = permutedims(A,(2,1))

  @test blockdims(Ap,(1,2)) == (4,3)
  @test blockdims(Ap,(2,1)) == (5,2)
  @test nnz(A) == nnz(Ap)
  @test nnzblocks(A) == nnzblocks(Ap)

  for I in eachindex(C)
    @test A[I] == Ap[permute(I,(2,1))]
  end

  @testset "BlockSparseTensor setindex! add block" begin
    T = BlockSparseTensor([2,3],[4,5])

    for I in eachindex(C)
      @test T[I] == 0.0
    end
    @test nnz(T) == 0
    @test nnzblocks(T) == 0
    @test !isblocknz(T,(1,1))
    @test !isblocknz(T,(2,1))
    @test !isblocknz(T,(1,2))
    @test !isblocknz(T,(2,2))

    T[1,1] = 1.0

    @test T[1,1] == 1.0
    @test nnz(T) == 8
    @test nnzblocks(T) == 1
    @test isblocknz(T,(1,1))
    @test !isblocknz(T,(2,1))
    @test !isblocknz(T,(1,2))
    @test !isblocknz(T,(2,2))

    T[4,8] = 2.0

    @test T[4,8] == 2.0
    @test nnz(T) == 8+15
    @test nnzblocks(T) == 2
    @test isblocknz(T,(1,1))
    @test !isblocknz(T,(2,1))
    @test !isblocknz(T,(1,2))
    @test isblocknz(T,(2,2))

    T[1,6] = 3.0

    @test T[1,6] == 3.0
    @test nnz(T) == 8+15+10
    @test nnzblocks(T) == 3
    @test isblocknz(T,(1,1))
    @test !isblocknz(T,(2,1))
    @test isblocknz(T,(1,2))
    @test isblocknz(T,(2,2))

    T[4,2] = 4.0

    @test T[4,2] == 4.0
    @test nnz(T) == 8+15+10+12
    @test nnzblocks(T) == 4
    @test isblocknz(T,(1,1))
    @test isblocknz(T,(2,1))
    @test isblocknz(T,(1,2))
    @test isblocknz(T,(2,2))
  end

  @testset "Add with different blocks" begin
    # Indices
    inds = ([2,3],[4,5])

    # Locations of non-zero blocks
    locsA = [(1,1),(1,2),(2,2)]
    A = BlockSparseTensor(locsA,inds...)
    randn!(A)

    locsB = [(1,2),(2,1)]
    B = BlockSparseTensor(locsB,inds...)
    randn!(B)

    R = A+B

    @test nnz(R) == dim(R)
    for I in eachindex(R)
      @test R[I] == A[I] + B[I]
    end
  end
end

