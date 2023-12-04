@eval module $(gensym())
using NDTensors.AllocateData: AllocateData, allocate, zero_init
using LinearAlgebra: Diagonal, Hermitian
using NDTensors.DiagonalArrays: DiagonalArray
using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.SparseArrayDOKs: SparseArrayDOK
using Test: @test, @testset, @test_throws
@testset "AllocateData" begin
  for arraytype in (
    Array,
    Diagonal,
    Hermitian,
    DiagonalArray,
    BlockSparseArray,
    SparseArrayDOK,
  ),
    elt in (
      Float32,
      Float64,
      ComplexF32,
      ComplexF64,
    ),
  initializers in (
    (undef,),
    (AllocateData.undef,),
    (zero_init,),
    (),
  ), axes in (
      (2, 2),
      (1:2, 1:2),
    )
      a = allocate(arraytype{elt}, initializers..., axes)
      @test a isa arraytype{elt,length(axes)}
      @test size(a) == (2, 2)
      if !isempty(initializers) && only(initializers) isa AllocateData.ZeroInitializer
        @test iszero(a)
      end
      a = allocate_zeros(arraytype{elt}, axes)
      @test a isa arraytype{elt,length(axes)}
      @test size(a) == (2, 2)
      @test iszero(a)
  end
  @test_throws AssertionError allocate(arraytype{elt}, (1:2, 0:2))
end
end
