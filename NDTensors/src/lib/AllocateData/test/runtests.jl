@eval module $(gensym())
using NDTensors.AllocateData: AllocateData, allocate, allocate_zeros, zero_init
using LinearAlgebra: Diagonal, Hermitian
using NDTensors.DiagonalArrays: DiagonalArray
using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.SparseArraysBase: SparseArrayDOK
using Test: @test, @testset, @test_broken, @test_throws

const arraytypes = (
  Array, Diagonal, Hermitian, DiagonalArray, BlockSparseArray, SparseArrayDOK
)
const elts = (Float32, Float64, ComplexF32, ComplexF64)
const initializerss = ((undef,), (AllocateData.undef,), (zero_init,), ())
const axess = ((2, 2), (1:2, 1:2))
@testset "AllocateData (arraytype=$arraytype, eltype=$elt, initializer=$initializers, axes=$axes)" for arraytype in
                                                                                                       arraytypes,
  elt in elts,
  initializers in initializerss,
  axes in axess

  a = allocate(arraytype{elt}, initializers..., axes)
  @test a isa arraytype{elt}
  @test ndims(a) == length(axes)
  @test size(a) == (2, 2)
  if !isempty(initializers) && only(initializers) isa AllocateData.ZeroInitializer
    @test iszero(a)
  end
  a = allocate_zeros(arraytype{elt}, axes)
  @test a isa arraytype{elt}
  @test ndims(a) == length(axes)
  @test size(a) == (2, 2)
  @test iszero(a)
  if !(arraytype <: BlockSparseArray)
    @test_throws AssertionError allocate(arraytype{elt}, (1:2, 0:2))
  else
    @test_broken error("Constructor should throw error for non-one-based axes.")
  end
end
end
