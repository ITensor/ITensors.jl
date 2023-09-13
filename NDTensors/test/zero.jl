using NDTensors: UnallocatedZeros, allocate, UnspecifiedZero
using Test
using FillArrays

## TODO right now fource allocated zeros to be a 
@testset "UnallocatedZeros" for T in [Float64, ComplexF64, UnspecifiedZero]
  T = ComplexF64
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
  #@test_broken z = UnallocatedZeros{T,1,Base.axes(()),Array{T,1}}(())

  z = UnallocatedZeros{T,1,Base.axes(()),Vector{T}}()
  @test length(z) == 0

  z = UnallocatedZeros(Vector, 10)
  @test NDTensors.alloctype(z) == Vector{NDTensors.default_eltype()}
  @test length(z) == 10

  z = UnallocatedZeros{T}(Array, 43, 20, 3)
  @test dim(z) == 43 * 20 * 3
  @test eltype(z) == T
  @test NDTensors.alloctype(z) == Array{T,3}
  @test ndims(z) == 3
  @test axes(z) == (Base.OneTo(43), Base.OneTo(20), Base.OneTo(3))

  @test typeof(NDTensors.data(z)) ==
    Zeros{T,3,Tuple{Base.OneTo{Int64},Base.OneTo{Int64},Base.OneTo{Int64}}}
  @test array(z) == NDTensors.allocate(z)
  zp = copy(z)
  @test zp == z
  @test eltype(complex(z)) == eltype(NDTensors.alloctype(complex(z))) == complex(T)

  @test sum(z) == zero(T)
end
