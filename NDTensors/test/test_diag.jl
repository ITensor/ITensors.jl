@eval module $(gensym())
using NDTensors
using Test: @testset, @test, @test_throws
using GPUArraysCore: @allowscalar
include("NDTensorsTestUtils/NDTensorsTestUtils.jl")
using .NDTensorsTestUtils: NDTensorsTestUtils

@testset "DiagTensor basic functionality" begin
  @testset "test device: $dev" for dev in NDTensorsTestUtils.devices_list(copy(ARGS)),
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
    @test @allowscalar dev(Array(dense(D))) == convert.(elt, d)
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
    Da = Array(D)
    Dm = Matrix(D)
    @allowscalar begin
      @test Da == NDTensors.LinearAlgebra.diagm(0 => vr)
      @test Da == NDTensors.LinearAlgebra.diagm(0 => vr)

      ## TODO Currently this permutedims requires scalar indexing on GPU. 
      Da = permutedims(D, (2, 1))
      @test Da == D
    end

    # Regression test for https://github.com/ITensor/ITensors.jl/issues/1199
    S = dev(tensor(Diag(randn(elt, 2)), (2, 2)))
    ## This was creating a `Dense{ReshapedArray{Adjoint{Matrix}}}` which, in mul!, was 
    ## becoming a Transpose{ReshapedArray{Adjoint{Matrix}}} which was causing issues on
    ## dispatching GPU mul!
    V = dev(tensor(Dense(randn(elt, 12, 2)'), (3, 4, 2)))
    S1 = contract(S, (2, -1), V, (3, 4, -1))
    S2 = contract(dense(S), (2, -1), copy(V), (3, 4, -1))
    @test @allowscalar S1 ≈ S2
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
end