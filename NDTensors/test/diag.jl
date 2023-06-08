using NDTensors
using Test
for dev in devs
  @testset "DiagTensor basic functionality" begin
    t = dev(tensor(Diag(rand(ComplexF64, 100)), (100, 100)))
    @test conj(data(store(t))) == data(store(conj(t)))
    @test typeof(conj(t)) <: DiagTensor

    d = rand(Float32, 10)
    D = dev(Diag{ComplexF64}(d))
    @test eltype(D) == ComplexF64
    @test dev(Array(dense(D))) == convert.(ComplexF64, d)
    simD = similar(D)
    @test length(simD) == length(D)
    @test eltype(simD) == eltype(D)
    D = dev(Diag(1.0))
    @test eltype(D) == Float64
    @test complex(D) == Diag(one(ComplexF64))
    @test similar(D) == Diag(0.0)

    d = 3
    vr = rand(d)
    D = dev(tensor(Diag(vr), (d, d)))
    @test Array(D) == NDTensors.LinearAlgebra.diagm(0 => vr)
    @test matrix(D) == NDTensors.LinearAlgebra.diagm(0 => vr)
    @test permutedims(D, (2, 1)) == D
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
