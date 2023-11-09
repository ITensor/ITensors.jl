using NDTensors
using Test

@testset "DiagTensor basic functionality" begin
  include("device_list.jl")
  devs = devices_list(copy(ARGS))
  @testset "test device: $dev" for dev in devs,
    elt in (Float32, ComplexF32, Float64, ComplexF64)

    if dev == NDTensors.mtl && real(elt) ≠ Float32
      # Metal doesn't support double precision
      continue
    end
    t = dev(tensor(Diag(rand(elt, 100)), (100, 100)))
    @test conj(data(store(t))) == data(store(conj(t)))
    @test typeof(conj(t)) <: DiagTensor

    d = rand(real(elt), 10)
    D = dev(Diag{elt}(d))
    @test eltype(D) == elt
    @test dev(Array(dense(D))) == convert.(elt, d)
    simD = similar(D)
    @test length(simD) == length(D)
    @test eltype(simD) == eltype(D)
    D = dev(Diag(one(elt)))
    @test eltype(D) == elt
    @test complex(D) == Diag(one(complex(elt)))
    @test similar(D) == Diag(0.0)

    D = Tensor(Diag(1), (2, 2))
    @test norm(D) == √2
    d = 3
    vr = rand(elt, d)
    D = dev(tensor(Diag(vr), (d, d)))
    @test Array(D) == NDTensors.LinearAlgebra.diagm(0 => vr)
    @test matrix(D) == NDTensors.LinearAlgebra.diagm(0 => vr)
    @test permutedims(D, (2, 1)) == D

    # Regression test for https://github.com/ITensor/ITensors.jl/issues/1199
    S = dev(tensor(Diag(randn(elt, 2)), (2, 2)))
    V = dev(tensor(Dense(randn(elt, 12, 2)'), (3, 4, 2)))
    @test contract(S, (2, -1), V, (3, 4, -1)) ≈
      contract(dense(S), (2, -1), copy(V), (3, 4, -1))
  end
end
@testset "DiagTensor contractions" begin
  t = tensor(Diag([1.0, 1.0, 1.0]), (3, 3))
  A = randomTensor(Dense, (3, 3))

  @test contract(t, (1, -2), t, (-2, 3)) == t
  @test contract(A, (1, -2), t, (-2, 3)) == A
  @test contract(A, (-2, 1), t, (-2, 3)) == transpose(A)
end
nothing
