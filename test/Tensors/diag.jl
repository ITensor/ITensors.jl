using ITensors,
      Test

@testset "DiagTensor basic functionality" begin

  t = Tensor(Diag(rand(ComplexF64,100)), (100,100))
  @test conj(data(store(t))) == data(store(conj(t)))
  @test typeof(conj(t)) <: DiagTensor

end
