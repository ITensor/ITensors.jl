using NDTensors
using Test
if "cuda" in ARGS || "all" in ARGS
  using CUDA
end
if "metal" in ARGS || "all" in ARGS
  using Metal
end

@testset "EmptyStorage test" begin
  include("device_list.jl")
  devs = devices_list(copy(ARGS))
  @testset "test device: $dev" for dev in devs
    T = dev(Tensor(EmptyStorage(NDTensors.EmptyNumber), (2, 2)))
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
end
