@eval module $(gensym())
using Test: @test, @testset, @test_broken, @test_throws

@testset "SparseArrayInterface (eltype=$elt)" for elt in
                                                  (Float32, ComplexF32, Float64, ComplexF64)
  @testset "Array" begin
    using NDTensors.SparseArrayInterface: SparseArrayInterface
    a = randn(2, 3)
    @test SparseArrayInterface.nonzeros(a) == a
    @test SparseArrayInterface.index_to_nonzero_index(a, CartesianIndex(1, 2)) ==
      CartesianIndex(1, 2)
    @test SparseArrayInterface.nonzero_index_to_index(a, CartesianIndex(1, 2)) ==
      CartesianIndex(1, 2)
  end
  @testset "Custom SparseArray" begin
    using LinearAlgebra: norm
    using NDTensors.SparseArrayInterface: SparseArrayInterface
    struct SparseArray{T,N} <: AbstractArray{T,N}
      data::Vector{T}
      dims::Tuple{Vararg{Int,N}}
      index_to_dataindex::Dict{CartesianIndex{N},Int}
      dataindex_to_index::Vector{CartesianIndex{N}}
    end
    SparseArray{T,N}(dims::Tuple{Vararg{Int,N}}) where {T,N} = SparseArray{T,N}(
      T[], dims, Dict{CartesianIndex{N},Int}(), Vector{CartesianIndex{N}}()
    )
    SparseArray{T,N}(dims::Vararg{Int,N}) where {T,N} = SparseArray{T,N}(dims)
    SparseArray{T}(dims::Tuple{Vararg{Int}}) where {T} = SparseArray{T,length(dims)}(dims)
    SparseArray{T}(dims::Vararg{Int}) where {T} = SparseArray{T}(dims)

    # AbstractArray interface
    Base.size(a::SparseArray) = a.dims
    function Base.getindex(a::SparseArray, I...)
      return SparseArrayInterface.sparse_getindex(a, I...)
    end
    function Base.setindex!(a::SparseArray, I...)
      return SparseArrayInterface.sparse_setindex!(a, I...)
    end
    function Base.similar(a::SparseArray, elt::Type, dims::Tuple{Vararg{Int}})
      return SparseArray{elt}(dims)
    end

    # Minimal interface
    SparseArrayInterface.nonzeros(a::SparseArray) = a.data
    function SparseArrayInterface.index_to_nonzero_index(
      a::SparseArray{<:Any,N}, I::CartesianIndex{N}
    ) where {N}
      return get(a.index_to_dataindex, I, nothing)
    end
    SparseArrayInterface.nonzero_index_to_index(a::SparseArray, I) = a.dataindex_to_index[I]
    function SparseArrayInterface.setindex_zero!(
      a::SparseArray{<:Any,N}, value, I::CartesianIndex{N}
    ) where {N}
      push!(a.data, value)
      push!(a.dataindex_to_index, I)
      a.index_to_dataindex[I] = length(a.data)
      return a
    end

    # Broadcasting
    function Broadcast.BroadcastStyle(arraytype::Type{<:SparseArray})
      return SparseArrayInterface.SparseArrayStyle{ndims(arraytype)}()
    end

    # Base
    function Base.iszero(a::SparseArray)
      return SparseArrayInterface.sparse_iszero(a)
    end
    function Base.isreal(a::SparseArray)
      return SparseArrayInterface.sparse_isreal(a)
    end
    function Base.zero(a::SparseArray)
      return SparseArrayInterface.sparse_zero(a)
    end
    function Base.one(a::SparseArray)
      return SparseArrayInterface.sparse_one(a)
    end
    function Base.:(==)(a1::SparseArray, a2::SparseArray)
      return SparseArrayInterface.sparse_isequal(a1, a2)
    end
    function Base.reshape(a::SparseArray, dims::Tuple{Vararg{Int}})
      return SparseArrayInterface.sparse_reshape(a, dims)
    end

    # Map
    function Base.map!(f, dest::AbstractArray, src::SparseArray)
      SparseArrayInterface.sparse_map!(f, dest, src)
      return dest
    end
    function Base.copy!(dest::AbstractArray, src::SparseArray)
      SparseArrayInterface.sparse_copy!(dest, src)
      return dest
    end
    function Base.copyto!(dest::AbstractArray, src::SparseArray)
      SparseArrayInterface.sparse_copyto!(dest, src)
      return dest
    end
    function Base.permutedims!(dest::AbstractArray, src::SparseArray, perm)
      SparseArrayInterface.sparse_permutedims!(dest, src, perm)
      return dest
    end

    # TODO: Test `fill!`.

    # Test
    a = SparseArray{elt}(2, 3)
    @test size(a) == (2, 3)
    @test axes(a) == (1:2, 1:3)
    @test SparseArrayInterface.nonzeros(a) == elt[]
    @test iszero(SparseArrayInterface.nonzero_length(a))
    @test collect(SparseArrayInterface.nonzero_indices(a)) == CartesianIndex{2}[]
    @test iszero(a)
    @test iszero(norm(a))
    for I in eachindex(a)
      @test iszero(a)
    end

    a = SparseArray{elt}(2, 3)
    a[1, 2] = 12
    @test size(a) == (2, 3)
    @test axes(a) == (1:2, 1:3)
    @test SparseArrayInterface.nonzeros(a) == elt[12]
    @test isone(SparseArrayInterface.nonzero_length(a))
    @test collect(SparseArrayInterface.nonzero_indices(a)) == [CartesianIndex(1, 2)]
    @test !iszero(a)
    @test !iszero(norm(a))
    for I in eachindex(a)
      if I == CartesianIndex(1, 2)
        @test a[I] == 12
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
    @test iszero(SparseArrayInterface.nonzero_length(a))

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
    @test iszero(SparseArrayInterface.nonzero_length(a))

    a = SparseArray{elt}(2, 3)
    a[1, 2] = 12
    a = copy(a)
    @test size(a) == (2, 3)
    @test axes(a) == (1:2, 1:3)
    @test SparseArrayInterface.nonzeros(a) == elt[12]
    @test isone(SparseArrayInterface.nonzero_length(a))
    @test collect(SparseArrayInterface.nonzero_indices(a)) == [CartesianIndex(1, 2)]
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
    @test SparseArrayInterface.nonzeros(a) == elt[24]
    @test isone(SparseArrayInterface.nonzero_length(a))
    @test collect(SparseArrayInterface.nonzero_indices(a)) == [CartesianIndex(1, 2)]
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
    @test SparseArrayInterface.nonzeros(c) == elt[12, 21]
    @test SparseArrayInterface.nonzero_length(c) == 2
    @test collect(SparseArrayInterface.nonzero_indices(c)) ==
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
    @test SparseArrayInterface.nonzeros(b) == elt[12]
    @test SparseArrayInterface.nonzero_length(b) == 1
    @test collect(SparseArrayInterface.nonzero_indices(b)) == [CartesianIndex(2, 1)]
    @test !iszero(b)
    @test !iszero(norm(b))
    for I in eachindex(b)
      if I == CartesianIndex(2, 1)
        @test b[I] == 12
      else
        @test iszero(b[I])
      end
    end
  end
  @testset "Custom DiagonalArray" begin
    struct DiagonalArray{T,N} <: AbstractArray{T,N}
      data::Vector{T}
      dims::Tuple{Vararg{Int,N}}
    end
    DiagonalArray{T,N}(::UndefInitializer, dims::Tuple{Vararg{Int,N}}) where {T,N} =
      DiagonalArray{T,N}(Vector{T}(undef, minimum(dims)), dims)
    DiagonalArray{T,N}(::UndefInitializer, dims::Vararg{Int,N}) where {T,N} =
      DiagonalArray{T,N}(undef, dims)
    DiagonalArray{T}(::UndefInitializer, dims::Tuple{Vararg{Int}}) where {T} =
      DiagonalArray{T,length(dims)}(undef, dims)
    DiagonalArray{T}(::UndefInitializer, dims::Vararg{Int}) where {T} =
      DiagonalArray{T}(undef, dims)

    # AbstractArray interface
    Base.size(a::DiagonalArray) = a.dims
    function Base.getindex(a::DiagonalArray, I...)
      return SparseArrayInterface.sparse_getindex(a, I...)
    end
    function Base.setindex!(a::DiagonalArray, I...)
      return SparseArrayInterface.sparse_setindex!(a, I...)
    end
    function Base.similar(a::DiagonalArray, elt::Type, dims::Tuple{Vararg{Int}})
      return DiagonalArray{elt}(undef, dims)
    end

    # Minimal interface
    SparseArrayInterface.nonzeros(a::DiagonalArray) = a.data
    function SparseArrayInterface.index_to_nonzero_index(
      a::DiagonalArray{<:Any,N}, I::CartesianIndex{N}
    ) where {N}
      !allequal(Tuple(I)) && return nothing
      return first(Tuple(I))
    end
    function SparseArrayInterface.nonzero_index_to_index(a::DiagonalArray, I)
      return CartesianIndex(ntuple(Returns(I), ndims(a)))
    end
    function SparseArrayInterface.setindex_zero!(
      a::DiagonalArray{<:Any,N}, value, I::CartesianIndex{N}
    ) where {N}
      return throw(ArgumentError("Can't set off-diagonal element of DiagonalArray"))
    end
    function SparseArrayInterface.sparse_zero(a::DiagonalArray, elt::Type, dims::Tuple)
      # TODO: Need to make generic to GPU.
      return DiagonalArray(zeros(elt, minimum(dims)), dims)
    end

    # Broadcasting
    function Broadcast.BroadcastStyle(arraytype::Type{<:DiagonalArray})
      return SparseArrayInterface.SparseArrayStyle{ndims(arraytype)}()
    end

    # Base
    function Base.iszero(a::DiagonalArray)
      return SparseArrayInterface.sparse_iszero(a)
    end
    function Base.isreal(a::DiagonalArray)
      return SparseArrayInterface.sparse_isreal(a)
    end
    function Base.zero(a::DiagonalArray)
      return SparseArrayInterface.sparse_zero(a)
    end
    function Base.one(a::DiagonalArray)
      return SparseArrayInterface.sparse_one(a)
    end
    function Base.:(==)(a1::DiagonalArray, a2::DiagonalArray)
      return SparseArrayInterface.sparse_isequal(a1, a2)
    end
    function Base.reshape(a::DiagonalArray, dims::Tuple{Vararg{Int}})
      return SparseArrayInterface.sparse_reshape(a, dims)
    end

    # Map
    function Base.map!(f, dest::AbstractArray, src::DiagonalArray)
      SparseArrayInterface.sparse_map!(f, dest, src)
      return dest
    end
    function Base.copy!(dest::AbstractArray, src::DiagonalArray)
      SparseArrayInterface.sparse_copy!(dest, src)
      return dest
    end
    function Base.copyto!(dest::AbstractArray, src::DiagonalArray)
      SparseArrayInterface.sparse_copyto!(dest, src)
      return dest
    end
    function Base.permutedims!(dest::AbstractArray, src::DiagonalArray, perm)
      SparseArrayInterface.sparse_permutedims!(dest, src, perm)
      return dest
    end

    # TODO: Test `fill!`.

    # Test
    a = DiagonalArray{elt}(undef, 2, 3)
    @test size(a) == (2, 3)
    a[1, 1] = 11
    a[2, 2] = 22
    @test a[1, 1] == 11
    @test a[2, 2] == 22
    @test_throws ArgumentError a[1, 2] = 12
    a[1, 2] = 0
    @test a[1, 1] == 11
    @test a[2, 2] == 22

    b = similar(a)
    @test b isa DiagonalArray
    @test size(b) == (2, 3)

    a = DiagonalArray(elt[1, 2, 3], (3, 3))
    @test size(a) == (3, 3)
    @test a[1, 1] == 1
    @test a[2, 2] == 2
    @test a[3, 3] == 3
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

    a = DiagonalArray(elt[1, 2], (2, 2, 2))
    # TODO: Make `reshape` for `DiagonalArray` return `SparseArray`.
    @test_broken reshape(a, 2, 4)
  end
end
end
