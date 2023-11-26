@eval module $(gensym())
using Test: @test, @testset, @test_broken
using NDTensors.SparseArrayDOKs: SparseArrayDOK, nonzero_keys, nonzero_length
using SparseArrays: nnz
@testset "SparseArrayDOK (eltype=$elt)" for elt in
                                            (Float32, ComplexF32, Float64, ComplexF64)
  @testset "Basics" begin
    a = SparseArrayDOK{elt}(3, 4)
    @test a == SparseArrayDOK{elt}((3, 4))
    @test a == SparseArrayDOK{elt}(undef, 3, 4)
    @test a == SparseArrayDOK{elt}(undef, (3, 4))
    @test iszero(a)
    @test iszero(nnz(a))
    @test nonzero_length(a) == nnz(a)
    @test size(a) == (3, 4)
    @test eltype(a) == elt
    for I in eachindex(a)
      @test iszero(a[I])
      @test a[I] isa elt
    end
    @test isempty(nonzero_keys(a))

    x12 = randn(elt)
    x23 = randn(elt)
    b = copy(a)
    @test b isa SparseArrayDOK{elt}
    @test iszero(b)
    b[1, 2] = x12
    b[2, 3] = x23
    @test iszero(a)
    @test !iszero(b)
    @test b[1, 2] == x12
    @test b[2, 3] == x23
    @test iszero(nonzero_length(a))
    @test_broken nonzero_length(b) == 2

    # To test:
    # reshape
    # zero (PermutedDimsArray)
    # map[!]
    # broadcast
    # Custom zero type
    # conversion to `SparseMatrixCSC`
  end
  @testset "map/broadcast" begin
    a = SparseArrayDOK{elt}(3, 4)
    a[1, 1] = 11
    a[3, 4] = 34
    @test nonzero_length(a) == 2
    2 * a
  end
end
end
