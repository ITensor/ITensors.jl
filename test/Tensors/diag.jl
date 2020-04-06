using ITensors,
      Test, LinearAlgebra

@testset "DiagTensor basic functionality" begin

  t = Tensor(Diag(rand(ComplexF64,100)), (100,100))
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
  i = Index(d,"i")
  j = Index(d,"j")
  vr = rand(d)
  D = tensor(diagITensor(vr, i,j))
  @test Array(D) == diagm(0=>vr) 
  @test matrix(D) == diagm(0=>vr)
  # fails because of missing similar method for NonuniformDiag :(
  #@test permutedims(D, (2, 1)) == tensor(diagITensor(vr, j, i))
  #@test permutedims(tensor(diagITensor(2.0, j, i)), (2, 1)) == tensor(diagITensor(2.0, j, i))
end
