using NDTensors
using Test

@testset "EmptyStorage" for op in ops
  T = op(Tensor(EmptyStorage(NDTensors.EmptyNumber), (2, 2)))
  @test size(T) == (2, 2)
  @test eltype(T) == NDTensors.EmptyNumber
  @test T[1, 1] == NDTensors.EmptyNumber()
  @test T[1, 2] == NDTensors.EmptyNumber()
  # TODO: This should fail with an out of bounds error!
  #@test T[1, 3] == NDTensors.EmptyNumber()

  Tc = complex(T)
  @test size(Tc) == (2, 2)
  @test eltype(Tc) == Complex{NDTensors.EmptyNumber}
  @test Tc[1, 1] == Complex(NDTensors.EmptyNumber(), NDTensors.EmptyNumber())
  @test Tc[1, 2] == Complex(NDTensors.EmptyNumber(), NDTensors.EmptyNumber())
end
