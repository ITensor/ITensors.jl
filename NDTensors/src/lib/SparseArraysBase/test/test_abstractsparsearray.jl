@eval module $(gensym())
using LinearAlgebra: dot, mul!, norm
using NDTensors.SparseArraysBase: SparseArraysBase
using NDTensors.NestedPermutedDimsArrays: NestedPermutedDimsArray
include("SparseArraysBaseTestUtils/SparseArraysBaseTestUtils.jl")
using .SparseArraysBaseTestUtils.AbstractSparseArrays: AbstractSparseArrays
using .SparseArraysBaseTestUtils.SparseArrays: SparseArrays
using Test: @test, @testset
@testset "AbstractSparseArray (arraytype=$SparseArray, eltype=$elt)" for SparseArray in (
    AbstractSparseArrays.SparseArray, SparseArrays.SparseArray
  ),
  elt in (Float32, ComplexF32, Float64, ComplexF64)

  a = SparseArray{elt}(2, 3)
  @test size(a) == (2, 3)
  @test axes(a) == (1:2, 1:3)
  @test SparseArraysBase.sparse_storage(a) == elt[]
  @test iszero(SparseArraysBase.stored_length(a))
  @test collect(SparseArraysBase.stored_indices(a)) == CartesianIndex{2}[]
  @test iszero(a)
  @test iszero(norm(a))
  for I in eachindex(a)
    @test iszero(a)
  end
  for I in CartesianIndices(a)
    @test isassigned(a, Tuple(I)...)
    @test isassigned(a, I)
  end
  @test !isassigned(a, 0, 1)
  @test !isassigned(a, CartesianIndex(0, 1))
  @test !isassigned(a, 1, 4)
  @test !isassigned(a, CartesianIndex(1, 4))

  a = SparseArray{elt}(2, 3)
  fill!(a, 0)
  @test size(a) == (2, 3)
  @test iszero(a)
  @test iszero(SparseArraysBase.stored_length(a))

  a_dense = SparseArraysBase.densearray(a)
  @test a_dense == a
  @test a_dense isa Array{elt,ndims(a)}

  a = SparseArray{elt}(2, 3)
  fill!(a, 2)
  @test size(a) == (2, 3)
  @test !iszero(a)
  @test SparseArraysBase.stored_length(a) == length(a)
  for I in eachindex(a)
    @test a[I] == 2
  end

  a = SparseArray{elt}(2, 3)
  a[1, 2] = 12
  @test a[1, 2] == 12
  @test a[3] == 12 # linear indexing
  @test size(a) == (2, 3)
  @test axes(a) == (1:2, 1:3)
  @test a[SparseArraysBase.StorageIndex(1)] == 12
  @test SparseArraysBase.sparse_storage(a) == elt[12]
  @test isone(SparseArraysBase.stored_length(a))
  @test collect(SparseArraysBase.stored_indices(a)) == [CartesianIndex(1, 2)]
  @test !iszero(a)
  @test !iszero(norm(a))
  for I in eachindex(a)
    if I == CartesianIndex(1, 2)
      @test a[I] == 12
    else
      @test iszero(a[I])
    end
  end
  for I in CartesianIndices(a)
    @test isassigned(a, Tuple(I)...)
    @test isassigned(a, I)
  end
  @test !isassigned(a, 0, 1)
  @test !isassigned(a, CartesianIndex(0, 1))
  @test !isassigned(a, 1, 4)
  @test !isassigned(a, CartesianIndex(1, 4))

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
  @test iszero(SparseArraysBase.stored_length(a))

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
  @test iszero(SparseArraysBase.stored_length(a))

  a = SparseArray{elt}(2, 3)
  a[1, 2] = 12
  a = copy(a)
  @test size(a) == (2, 3)
  @test axes(a) == (1:2, 1:3)
  @test SparseArraysBase.sparse_storage(a) == elt[12]
  @test isone(SparseArraysBase.stored_length(a))
  @test SparseArraysBase.storage_indices(a) == 1:1
  @test collect(SparseArraysBase.stored_indices(a)) == [CartesianIndex(1, 2)]
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
  @test SparseArraysBase.sparse_storage(a) == elt[24]
  @test isone(SparseArraysBase.stored_length(a))
  @test collect(SparseArraysBase.stored_indices(a)) == [CartesianIndex(1, 2)]
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
  @test SparseArraysBase.sparse_storage(c) == elt[12, 21]
  @test SparseArraysBase.stored_length(c) == 2
  @test collect(SparseArraysBase.stored_indices(c)) ==
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
  @test SparseArraysBase.sparse_storage(b) == elt[12]
  @test SparseArraysBase.stored_length(b) == 1
  @test collect(SparseArraysBase.stored_indices(b)) == [CartesianIndex(2, 1)]
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
  b = PermutedDimsArray(a, (2, 1))
  @test size(b) == (3, 2)
  @test axes(b) == (1:3, 1:2)
  @test SparseArraysBase.sparse_storage(b) == elt[12]
  @test SparseArraysBase.stored_length(b) == 1
  @test collect(SparseArraysBase.stored_indices(b)) == [CartesianIndex(2, 1)]
  @test !iszero(b)
  @test !iszero(norm(b))
  for I in eachindex(b)
    if I == CartesianIndex(2, 1)
      @test b[I] == 12
    else
      @test iszero(b[I])
    end
  end

  a = SparseArray{Matrix{elt}}(
    2, 3; zero=(a, I) -> (z = similar(eltype(a), 2, 3); fill!(z, false); z)
  )
  a[1, 2] = randn(elt, 2, 3)
  b = NestedPermutedDimsArray(a, (2, 1))
  @test size(b) == (3, 2)
  @test axes(b) == (1:3, 1:2)
  @test SparseArraysBase.sparse_storage(b) == [a[1, 2]]
  @test SparseArraysBase.stored_length(b) == 1
  @test collect(SparseArraysBase.stored_indices(b)) == [CartesianIndex(2, 1)]
  @test !iszero(b)
  @test !iszero(norm(b))
  for I in eachindex(b)
    if I == CartesianIndex(2, 1)
      @test b[I] == permutedims(a[1, 2], (2, 1))
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
  @test SparseArraysBase.stored_length(a + b) == length(a)

  a = SparseArray{elt}(2, 3)
  a[1, 2] = 12
  b = zeros(elt, 2, 3)
  b[2, 1] = 21
  a′ = copy(a)
  a′ .+= b
  @test a′ == a + b
  # TODO: Should this be:
  # ```julia
  # @test SparseArraysBase.stored_length(a′) == 2
  # ```
  # ? I.e. should it only store the nonzero values?
  @test SparseArraysBase.stored_length(a′) == 6

  # Matrix multiplication
  a1 = SparseArray{elt}(2, 3)
  a1[1, 2] = 12
  a1[2, 1] = 21
  a2 = SparseArray{elt}(3, 4)
  a2[1, 1] = 11
  a2[2, 2] = 22
  a_dest = a1 * a2
  @test Array(a_dest) ≈ Array(a1) * Array(a2)
  @test a_dest isa SparseArray{elt}
  @test SparseArraysBase.stored_length(a_dest) == 2

  # Dot product
  a1 = SparseArray{elt}(4)
  a1[1] = randn()
  a1[3] = randn()
  a2 = SparseArray{elt}(4)
  a2[2] = randn()
  a2[3] = randn()
  a_dest = a1' * a2
  @test a_dest isa elt
  @test a_dest ≈ Array(a1)' * Array(a2)
  @test a_dest ≈ dot(a1, a2)

  # In-place matrix multiplication
  a1 = SparseArray{elt}(2, 3)
  a1[1, 2] = 12
  a1[2, 1] = 21
  a2 = SparseArray{elt}(3, 4)
  a2[1, 1] = 11
  a2[2, 2] = 22
  a_dest = SparseArray{elt}(2, 4)
  mul!(a_dest, a1, a2)
  @test Array(a_dest) ≈ Array(a1) * Array(a2)
  @test a_dest isa SparseArray{elt}
  @test SparseArraysBase.stored_length(a_dest) == 2

  # In-place matrix multiplication
  a1 = SparseArray{elt}(2, 3)
  a1[1, 2] = 12
  a1[2, 1] = 21
  a2 = SparseArray{elt}(3, 4)
  a2[1, 1] = 11
  a2[2, 2] = 22
  a_dest = SparseArray{elt}(2, 4)
  a_dest[1, 2] = 12
  a_dest[2, 1] = 21
  α = elt(2)
  β = elt(3)
  a_dest′ = copy(a_dest)
  mul!(a_dest, a1, a2, α, β)
  @test Array(a_dest) ≈ Array(a1) * Array(a2) * α + Array(a_dest′) * β
  @test a_dest isa SparseArray{elt}
  @test SparseArraysBase.stored_length(a_dest) == 2

  # cat
  a1 = SparseArray{elt}(2, 3)
  a1[1, 2] = 12
  a1[2, 1] = 21
  a2 = SparseArray{elt}(2, 3)
  a2[1, 1] = 11
  a2[2, 2] = 22

  a_dest = cat(a1, a2; dims=1)
  @test size(a_dest) == (4, 3)
  @test SparseArraysBase.stored_length(a_dest) == 4
  @test a_dest[1, 2] == a1[1, 2]
  @test a_dest[2, 1] == a1[2, 1]
  @test a_dest[3, 1] == a2[1, 1]
  @test a_dest[4, 2] == a2[2, 2]

  a_dest = cat(a1, a2; dims=2)
  @test size(a_dest) == (2, 6)
  @test SparseArraysBase.stored_length(a_dest) == 4
  @test a_dest[1, 2] == a1[1, 2]
  @test a_dest[2, 1] == a1[2, 1]
  @test a_dest[1, 4] == a2[1, 1]
  @test a_dest[2, 5] == a2[2, 2]

  a_dest = cat(a1, a2; dims=(1, 2))
  @test size(a_dest) == (4, 6)
  @test SparseArraysBase.stored_length(a_dest) == 4
  @test a_dest[1, 2] == a1[1, 2]
  @test a_dest[2, 1] == a1[2, 1]
  @test a_dest[3, 4] == a2[1, 1]
  @test a_dest[4, 5] == a2[2, 2]

  ## # Sparse matrix of matrix multiplication
  ## TODO: Make this work, seems to require
  ## a custom zero constructor.
  ## a1 = SparseArray{Matrix{elt}}(2, 3)
  ## a1[1, 1] = zeros(elt, (2, 3))
  ## a1[1, 2] = randn(elt, (2, 3))
  ## a1[2, 1] = randn(elt, (2, 3))
  ## a1[2, 2] = zeros(elt, (2, 3))
  ## a2 = SparseArray{Matrix{elt}}(3, 4)
  ## a2[1, 1] = randn(elt, (3, 4))
  ## a2[1, 2] = zeros(elt, (3, 4))
  ## a2[2, 2] = randn(elt, (3, 4))
  ## a2[2, 2] = zeros(elt, (3, 4))
  ## a_dest = SparseArray{Matrix{elt}}(2, 4)
  ## a_dest[1, 1] = zeros(elt, (3, 4))
  ## a_dest[1, 2] = zeros(elt, (3, 4))
  ## a_dest[2, 2] = zeros(elt, (3, 4))
  ## a_dest[2, 2] = zeros(elt, (3, 4))
  ## mul!(a_dest, a1, a2)
  ## @test Array(a_dest) ≈ Array(a1) * Array(a2)
  ## @test a_dest isa SparseArray{Matrix{elt}}
  ## @test SparseArraysBase.stored_length(a_dest) == 2
end
end
