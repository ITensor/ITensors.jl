using NDTensors
using Test

@testset "EmptyStorage test" begin
  include("device_list.jl")
  devs = devices_list(copy(ARGS))
  @testset "test device: $dev" for dev in devs
    T = dev(Tensor(EmptyStorage(NDTensors.UnspecifiedZero), (2, 2)))
    @test size(T) == (2, 2)
    @test eltype(T) == NDTensors.UnspecifiedZero
    @test T[1, 1] == NDTensors.UnspecifiedZero()
    @test T[1, 2] == NDTensors.UnspecifiedZero()
    # TODO: This should fail with an out of bounds error!
    #@test T[1, 3] == NDTensors.UnspecifiedZero()

    Tc = complex(T)
    @test size(Tc) == (2, 2)
    @test eltype(Tc) == Complex{NDTensors.UnspecifiedZero}
    @test Tc[1, 1] == Complex(NDTensors.UnspecifiedZero(), NDTensors.UnspecifiedZero())
    @test Tc[1, 2] == Complex(NDTensors.UnspecifiedZero(), NDTensors.UnspecifiedZero())

    T = dev(EmptyTensor(Float64, (2, 2)))
    @test blockoffsets(T) == BlockOffsets{2}()
    T = dev(EmptyBlockSparseTensor(Float64, ([1, 1], [1, 1])))
    @test blockoffsets(T) == BlockOffsets{2}()

    T = dev(EmptyStorage(NDTensors.UnspecifiedZero))
    @test zero(T) isa typeof(T)

    T = dev(EmptyTensor(NDTensors.UnspecifiedZero, (2, 2)))
    @test zero(T) isa typeof(T)
  end
end
