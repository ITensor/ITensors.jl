using NDTensors
using Test

@testset "DiagTensor basic functionality" begin
  t = tensor(Diag(rand(ComplexF64, 100)), (100, 100))
  @test conj(data(store(t))) == data(store(conj(t)))
  @test typeof(conj(t)) <: DiagTensor

  d = rand(Float32, 10)
  D = Diag{ComplexF64}(d)
  @test eltype(D) == ComplexF64
  @test Array(dense(D)) == convert.(ComplexF64, d)
  simD = similar(D)
  @test length(simD) == length(D)
  @test eltype(simD) == eltype(D)
  D = Diag(1.0)
  @test eltype(D) == Float64
  @test complex(D) == Diag(one(ComplexF64))
  @test similar(D) == Diag(0.0)

  d = 3
  vr = rand(d)
  D = tensor(Diag(vr), (d, d))
  @test Array(D) == NDTensors.LinearAlgebra.diagm(0 => vr)
  @test matrix(D) == NDTensors.LinearAlgebra.diagm(0 => vr)
  @test permutedims(D, (2, 1)) == D
end

nothing
