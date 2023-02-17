using ITensors, Test

@testset "QN Combiner" begin
  d = 1
  i = Index([QN(0) => d, QN(0) => d])
  A = randomITensor(i)
  C = combiner(i)
  AC = A * C

  Ã = AC * dag(C)
  @test Ã ≈ A
end

nothing
