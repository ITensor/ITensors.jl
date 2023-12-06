@eval module $(gensym())

# TODO: Test:
# zero (PermutedDimsArray)
# Custom zero type
# Slicing

using Test: @test, @testset, @test_broken
using NDTensors.SparseArrayDOKs: SparseArrayDOK, SparseMatrixDOK
using NDTensors.SparseArrayInterface: storage_indices, nstored
using SparseArrays: SparseMatrixCSC, nnz
@testset "SparseArrayDOK (eltype=$elt)" for elt in
                                            (Float32, ComplexF32, Float64, ComplexF64)
  @testset "Basics" begin
    a = SparseArrayDOK{elt}(3, 4)
    @test a == SparseArrayDOK{elt}((3, 4))
    @test a == SparseArrayDOK{elt}(undef, 3, 4)
    @test a == SparseArrayDOK{elt}(undef, (3, 4))
    @test iszero(a)
    @test iszero(nnz(a))
    @test nstored(a) == nnz(a)
    @test size(a) == (3, 4)
    @test eltype(a) == elt
    for I in eachindex(a)
      @test iszero(a[I])
      @test a[I] isa elt
    end
    @test isempty(storage_indices(a))

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
    @test iszero(nstored(a))
    @test nstored(b) == 2
  end
  @testset "map/broadcast" begin
    a = SparseArrayDOK{elt}(3, 4)
    a[1, 1] = 11
    a[3, 4] = 34
    @test nstored(a) == 2
    b = 2 * a
    @test nstored(b) == 2
    @test b[1, 1] == 2 * 11
    @test b[3, 4] == 2 * 34
  end
  @testset "reshape" begin
    a = SparseArrayDOK{elt}(2, 2, 2)
    a[1, 2, 2] = 122
    b = reshape(a, 2, 4)
    @test b[1, 4] == 122
  end
  @testset "Matrix multiplication" begin
    a1 = SparseArrayDOK{elt}(2, 3)
    a1[1, 2] = 12
    a1[2, 1] = 21
    a2 = SparseArrayDOK{elt}(3, 4)
    a2[1, 1] = 11
    a2[2, 2] = 22
    a2[3, 3] = 33
    a_dest = a1 * a2
    # TODO: Use `densearray` to make generic to GPU.
    @test Array(a_dest) ≈ Array(a1) * Array(a2)
    # TODO: Make this work with `ArrayLayouts`.
    @test nstored(a_dest) == 2
    @test a_dest isa SparseMatrixDOK{elt}

    a2 = randn(elt, (3, 4))
    a_dest = a1 * a2
    # TODO: Use `densearray` to make generic to GPU.
    @test Array(a_dest) ≈ Array(a1) * Array(a2)
    @test nstored(a_dest) == 8
    @test a_dest isa Matrix{elt}
  end
  @testset "SparseMatrixCSC" begin
    a = SparseArrayDOK{elt}(2, 2)
    a[1, 2] = 12
    for (type, a′) in ((SparseMatrixCSC, a), (SparseArrayDOK, SparseMatrixCSC(a)))
      b = type(a′)
      @test b isa type{elt}
      @test b[1, 2] == 12
      @test isone(nnz(b))
      for I in eachindex(b)
        if I ≠ CartesianIndex(1, 2)
          @test iszero(b[I])
        end
      end
    end
  end
end
end
