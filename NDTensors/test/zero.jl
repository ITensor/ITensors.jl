using NDTensors: UnallocatedZeros, allocate, UnspecifiedZero
using Test
using FillArrays
## TODO right now fource allocated zeros to be a 
@testset "UnallocatedZeros" for T in [Float64, ComplexF64, UnspecifiedZero]
  T = Float64
  v = Vector
  N = ndims(v)
  z = UnallocatedZeros{T,N,v{T}}(())
  @test length(z) == 1
  @test norm(z) == zero(T)
  @test z[] == zero(T)

  @test NDTensors.allocate(z) isa v{T}
  z = UnallocatedZeros{T,N,v{T}}((10,))
  @test length(z) == 10
  @test norm(z) == zero(T)
  @test z[9] == zero(T)

  is = (33, 5, 34)
  N = 3
  z = UnallocatedZeros{T,N,Array{T,N}}(is)
  @test norm(z) == zero(T)
  @test z[3, 5, 24] == zero(T)
  fz = FillArrays.Zeros{T,N}(is)
  @test_broken z = UnallocatedZeros{T,1,Base.axes(()),Array{T,1}}(())
end
