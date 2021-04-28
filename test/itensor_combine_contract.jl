using ITensors
using Test
import Random: seed!

seed!(12345)

@testset "ITensor combine contract order $(2*N) tensors" for N in 1:5
  d = 1
  i = Index(QN(0) => d, QN(1) => d)
  is = IndexSet(ntuple(n -> settags(i, "i$n"), Val(N)))
  A = randomITensor(is'..., dag(is)...)
  B = randomITensor(is'..., dag(is)...)

  @test !ITensors.using_combine_contract()

  C_contract = A' * B

  ITensors.enable_combine_contract()
  @test ITensors.using_combine_contract()

  C_combine_contract = A' * B

  ITensors.disable_combine_contract()
  @test !ITensors.using_combine_contract()

  @test C_contract ≈ C_combine_contract
end
