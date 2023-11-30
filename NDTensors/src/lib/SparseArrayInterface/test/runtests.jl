@eval module $(gensym())
using Compat: Returns, allequal
using LinearAlgebra: norm
include("SparseArrayInterfaceTestUtils/SparseArrayInterfaceTestUtils.jl")
using .SparseArrayInterfaceTestUtils.DiagonalArrays: DiagonalArray
using .SparseArrayInterfaceTestUtils.SparseArrays: SparseArray
using Test: @test, @testset, @test_broken, @test_throws

@testset "SparseArrayInterface (eltype=$elt)" for elt in
                                                  (Float32, ComplexF32, Float64, ComplexF64)
  @testset "Array" begin
    using NDTensors.SparseArrayInterface: SparseArrayInterface
    a = randn(2, 3)
    @test SparseArrayInterface.storage(a) == a
    @test SparseArrayInterface.index_to_storage_index(a, CartesianIndex(1, 2)) ==
      CartesianIndex(1, 2)
    @test SparseArrayInterface.storage_index_to_index(a, CartesianIndex(1, 2)) ==
      CartesianIndex(1, 2)
  end
  @testset "Custom SparseArray" begin
    a = SparseArray{elt}(2, 3)
    @test size(a) == (2, 3)
    @test axes(a) == (1:2, 1:3)
    @test SparseArrayInterface.storage(a) == elt[]
    @test iszero(SparseArrayInterface.nstored(a))
    @test collect(SparseArrayInterface.stored_indices(a)) == CartesianIndex{2}[]
    @test iszero(a)
    @test iszero(norm(a))
    for I in eachindex(a)
      @test iszero(a)
    end

    a = SparseArray{elt}(2, 3)
    fill!(a, 0)
    @test size(a) == (2, 3)
    @test iszero(a)
    @test iszero(SparseArrayInterface.nstored(a))

    a_dense = SparseArrayInterface.densearray(a)
    @test a_dense == a
    @test a_dense isa Array{elt,ndims(a)}

    a = SparseArray{elt}(2, 3)
    fill!(a, 2)
    @test size(a) == (2, 3)
    @test !iszero(a)
    @test SparseArrayInterface.nstored(a) == length(a)
    for I in eachindex(a)
      @test a[I] == 2
    end

    a = SparseArray{elt}(2, 3)
    a[1, 2] = 12
    @test a[1, 2] == 12
    @test a[3] == 12 # linear indexing
    @test size(a) == (2, 3)
    @test axes(a) == (1:2, 1:3)
    @test a[SparseArrayInterface.StorageIndex(1)] == 12
    @test SparseArrayInterface.storage(a) == elt[12]
    @test isone(SparseArrayInterface.nstored(a))
    @test collect(SparseArrayInterface.stored_indices(a)) == [CartesianIndex(1, 2)]
    @test !iszero(a)
    @test !iszero(norm(a))
    for I in eachindex(a)
      if I == CartesianIndex(1, 2)
        @test a[I] == 12
      else
        @test iszero(a[I])
      end
    end

    a = SparseArray{elt}(2, 3)
    a[1, 2] = 12
    a = map(x -> 2x, a)
    for I in eachindex(a)
      if I == CartesianIndex(1, 2)
        @test a[I] == 2 * 12
      else
        @test iszero(a[I])
      end
    end

    a = SparseArray{elt}(2, 2, 2)
    a[1, 2, 2] = 122
    a_r = reshape(a, 2, 4)
    @test a_r[1, 4] == a[1, 2, 2] == 122

    a = SparseArray{elt}(2, 3)
    a[1, 2] = 12
    a = zero(a)
    @test size(a) == (2, 3)
    @test iszero(SparseArrayInterface.nstored(a))

    a = SparseArray{elt}(2, 3)
    a[1, 2] = 12
    b = SparseArray{elt}(2, 3)
    b[2, 1] = 21
    @test a == a
    @test a ≠ b

    a = SparseArray{elt}(2, 3)
    a[1, 2] = 12
    @test isreal(a)

    a = SparseArray{elt}(2, 3)
    a[1, 2] = randn(elt)
    b = copy(a)
    conj!(b)
    for I in eachindex(a)
      @test conj(a[I]) == b[I]
    end

    a = SparseArray{elt}(2, 3)
    a[1, 2] = randn(elt)
    b = conj(a)
    for I in eachindex(a)
      @test conj(a[I]) == b[I]
    end

    if !(elt <: Real)
      a = SparseArray{elt}(2, 3)
      a[1, 2] = 12 + 12im
      @test !isreal(a)
    end

    a = SparseArray{elt}(2, 2)
    a[1, 2] = 12
    a = one(a)
    @test size(a) == (2, 2)
    @test isone(a[1, 1])
    @test isone(a[2, 2])
    @test iszero(a[1, 2])
    @test iszero(a[2, 1])

    a = SparseArray{elt}(2, 3)
    a[1, 2] = 12
    a = zero(a)
    @test size(a) == (2, 3)
    @test iszero(SparseArrayInterface.nstored(a))

    a = SparseArray{elt}(2, 3)
    a[1, 2] = 12
    a = copy(a)
    @test size(a) == (2, 3)
    @test axes(a) == (1:2, 1:3)
    @test SparseArrayInterface.storage(a) == elt[12]
    @test isone(SparseArrayInterface.nstored(a))
    @test SparseArrayInterface.storage_indices(a) == 1:1
    @test collect(SparseArrayInterface.stored_indices(a)) == [CartesianIndex(1, 2)]
    @test !iszero(a)
    @test !iszero(norm(a))
    for I in eachindex(a)
      if I == CartesianIndex(1, 2)
        @test a[I] == 12
      else
        @test iszero(a[I])
      end
    end

    a = SparseArray{elt}(2, 3)
    a[1, 2] = 12
    a = 2 * a
    @test size(a) == (2, 3)
    @test axes(a) == (1:2, 1:3)
    @test SparseArrayInterface.storage(a) == elt[24]
    @test isone(SparseArrayInterface.nstored(a))
    @test collect(SparseArrayInterface.stored_indices(a)) == [CartesianIndex(1, 2)]
    @test !iszero(a)
    @test !iszero(norm(a))
    for I in eachindex(a)
      if I == CartesianIndex(1, 2)
        @test a[I] == 24
      else
        @test iszero(a[I])
      end
    end

    a = SparseArray{elt}(2, 3)
    a[1, 2] = 12
    b = SparseArray{elt}(2, 3)
    b[2, 1] = 21
    c = a + b
    @test size(c) == (2, 3)
    @test axes(c) == (1:2, 1:3)
    @test SparseArrayInterface.storage(c) == elt[12, 21]
    @test SparseArrayInterface.nstored(c) == 2
    @test collect(SparseArrayInterface.stored_indices(c)) ==
      [CartesianIndex(1, 2), CartesianIndex(2, 1)]
    @test !iszero(c)
    @test !iszero(norm(c))
    for I in eachindex(c)
      if I == CartesianIndex(1, 2)
        @test c[I] == 12
      elseif I == CartesianIndex(2, 1)
        @test c[I] == 21
      else
        @test iszero(c[I])
      end
    end

    a = SparseArray{elt}(2, 3)
    a[1, 2] = 12
    b = permutedims(a, (2, 1))
    @test size(b) == (3, 2)
    @test axes(b) == (1:3, 1:2)
    @test SparseArrayInterface.storage(b) == elt[12]
    @test SparseArrayInterface.nstored(b) == 1
    @test collect(SparseArrayInterface.stored_indices(b)) == [CartesianIndex(2, 1)]
    @test !iszero(b)
    @test !iszero(norm(b))
    for I in eachindex(b)
      if I == CartesianIndex(2, 1)
        @test b[I] == 12
      else
        @test iszero(b[I])
      end
    end

    a = SparseArray{elt}(2, 3)
    a[1, 2] = 12
    b = randn(elt, 2, 3)
    b .= a
    @test a == b
    for I in eachindex(a)
      @test a[I] == b[I]
    end

    a = SparseArray{elt}(2, 3)
    a[1, 2] = 12
    b = randn(elt, 2, 3)
    b .= 2 .* a
    @test 2 * a == b
    for I in eachindex(a)
      @test 2 * a[I] == b[I]
    end

    a = SparseArray{elt}(2, 3)
    a[1, 2] = 12
    b = randn(elt, 2, 3)
    b .= 2 .+ a
    @test 2 .+ a == b
    for I in eachindex(a)
      @test 2 + a[I] == b[I]
    end

    a = SparseArray{elt}(2, 3)
    a[1, 2] = 12
    b = randn(elt, 2, 3)
    map!(x -> 2x, b, a)
    @test 2 * a == b
    for I in eachindex(a)
      @test 2 * a[I] == b[I]
    end

    a = SparseArray{elt}(2, 3)
    a[1, 2] = 12
    b = zeros(elt, 2, 3)
    b[2, 1] = 21
    @test Array(a) == a
    @test a + b == Array(a) + b
    @test b + a == Array(a) + b
    @test b .+ 2 .* a == 2 * Array(a) + b
    @test a .+ 2 .* b == Array(a) + 2b
    @test a + b isa Matrix{elt}
    @test b + a isa Matrix{elt}
    @test SparseArrayInterface.nstored(a + b) == length(a)

    a = SparseArray{elt}(2, 3)
    a[1, 2] = 12
    b = zeros(elt, 2, 3)
    b[2, 1] = 21
    a′ = copy(a)
    a′ .+= b
    @test a′ == a + b
    @test SparseArrayInterface.nstored(a′) == 2
  end
  @testset "Custom DiagonalArray" begin
    # TODO: Test `fill!`.

    # Test
    a = DiagonalArray{elt}(undef, 2, 3)
    @test size(a) == (2, 3)
    a[1, 1] = 11
    a[2, 2] = 22
    @test a[1, 1] == 11
    @test a[2, 2] == 22
    @test_throws ArgumentError a[1, 2] = 12
    @test SparseArrayInterface.storage_indices(a) == 1:2
    @test collect(SparseArrayInterface.stored_indices(a)) ==
      [CartesianIndex(1, 1), CartesianIndex(2, 2)]
    a[1, 2] = 0
    @test a[1, 1] == 11
    @test a[2, 2] == 22

    a_dense = SparseArrayInterface.densearray(a)
    @test a_dense == a
    @test a_dense isa Array{elt,ndims(a)}

    b = similar(a)
    @test b isa DiagonalArray
    @test size(b) == (2, 3)

    a = DiagonalArray(elt[1, 2, 3], (3, 3))
    @test size(a) == (3, 3)
    @test a[1, 1] == 1
    @test a[2, 2] == 2
    @test a[3, 3] == 3
    @test a[SparseArrayInterface.StorageIndex(1)] == 1
    @test a[SparseArrayInterface.StorageIndex(2)] == 2
    @test a[SparseArrayInterface.StorageIndex(3)] == 3
    @test iszero(a[1, 2])

    a = DiagonalArray(elt[1, 2, 3], (3, 3))
    a = 2 * a
    @test size(a) == (3, 3)
    @test a[1, 1] == 2
    @test a[2, 2] == 4
    @test a[3, 3] == 6
    @test iszero(a[1, 2])

    a = DiagonalArray(elt[1, 2, 3], (3, 3))
    a_r = reshape(a, 9)
    @test a_r isa DiagonalArray{elt,1}
    for I in LinearIndices(a)
      @test a[I] == a_r[I]
    end

    # This needs `Base.reshape` with a custom destination
    # calling `SparseArrayInterface.sparse_reshape!`
    # in order to specify an appropriate output
    # type to work.
    a = DiagonalArray(elt[1, 2], (2, 2, 2))
    a_r = reshape(a, 2, 4)
    @test a_r isa Matrix{elt}
    for I in LinearIndices(a)
      @test a[I] == a_r[I]
    end
  end
end
end
