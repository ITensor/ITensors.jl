using NDTensors: UnallocatedZeros, allocate, UnspecifiedZero
using Test

@testset "UnallocatedZeros" for T in [Float64, ComplexF64, UnspecifiedZero]
    for v = [Vector]
      N = ndims(v)
      z = UnallocatedZeros{T, N, v{T}}(())
      @test length(z) == 1
      @test norm(z) == zero(T)
      @test z[] == zero(T)

      @test NDTensors.allocate(z) isa v{T}
      z = UnallocatedZeros{T, N, v{T}}((10,))
      @test length(z) == 10
      @test norm(z) == zero(T)
      @test z[9] == zero(T)
    end
  end