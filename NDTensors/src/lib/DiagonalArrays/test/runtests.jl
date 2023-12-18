@eval module $(gensym())
using Test: @test, @testset, @test_broken
using NDTensors.DiagonalArrays: DiagonalArrays, DiagonalArray, DiagonalMatrix, diaglength
using NDTensors.SparseArrayDOKs: SparseArrayDOK
using NDTensors.SparseArrayInterface: nstored
using NDTensors: NDTensors
using GPUArraysCore: @allowscalar
include(joinpath(pkgdir(NDTensors), "test", "NDTensorsTestUtils", "NDTensorsTestUtils.jl"))
using .NDTensorsTestUtils: devices_list
@testset "Test NDTensors.DiagonalArrays" begin
  @testset "README" begin
    @test include(
      joinpath(
        pkgdir(DiagonalArrays), "src", "lib", "DiagonalArrays", "examples", "README.jl"
      ),
    ) isa Any
  end
  @testset "DiagonalArray (eltype=$elt)" for dev in devices_list(ARGS),
    elt in (Float32, Float64, ComplexF32, ComplexF64)

    @testset "Basics" begin
      a = dev(fill(one(elt), 2, 3))
      @test diaglength(a) == 2
      a = dev(fill(one(elt)))
      @test diaglength(a) == 1
    end
    @testset "Matrix multiplication" begin
      a1 = dev(DiagonalArray{elt}(undef, (2, 3)))
      a2 = dev(DiagonalArray{elt}(undef, (3, 4)))
      @allowscalar begin
        a1[1, 1] = 11
        a1[2, 2] = 22
        a2[1, 1] = 11
        a2[2, 2] = 22
        a2[3, 3] = 33
      end
      ## So far this fails on GPU with scalar indexing because * calls muladd!
      ## which causl default_blasmul! which calls getindex
      a_dest = a1 * a2
      # TODO: Use `densearray` to make generic to GPU.
      @test Array(a_dest) ≈ Array(a1) * Array(a2)
      # TODO: Make this work with `ArrayLayouts`.
      @test nstored(a_dest) == 2
      @test a_dest isa DiagonalMatrix{elt}

      # TODO: Make generic to GPU, use `allocate_randn`?
      a2 = dev(randn(elt, (3, 4)))
      a_dest = a1 * a2
      # TODO: Use `densearray` to make generic to GPU.
      @test Array(a_dest) ≈ Array(a1) * Array(a2)
      @test nstored(a_dest) == 8
      @test a_dest isa Matrix{elt}

      a2 = SparseArrayDOK{elt}(3, 4)
      a2[1, 1] = 11
      a2[2, 2] = 22
      a2[3, 3] = 33
      a_dest = a1 * a2
      # TODO: Use `densearray` to make generic to GPU.
      @test Array(a_dest) ≈ Array(a1) * Array(a2)
      # TODO: Define `SparseMatrixDOK`.
      # TODO: Make this work with `ArrayLayouts`.
      @test nstored(a_dest) == 2
      @test a_dest isa SparseArrayDOK{elt,2}
    end
  end
end
end
