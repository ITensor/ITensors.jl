using ITensors
using ITensors.NDTensors
using Test

@testset "NDTensors Dense contract" begin
  i = Index(2)
  T1 = randomTensor((i'', i'))
  T2 = randomTensor((i', i))
  R = randomTensor((i'', i))

  labelsT1 = (1, -1)
  labelsT2 = (-1, 2)
  labelsR = (1, 2)

  @test @inferred(NDTensors.contraction_output(T1, labelsT1, T2, labelsT2, labelsR)) isa DenseTensor{Float64, 2, Tuple{Index{Int64}, Index{Int64}}, Dense{Float64, Vector{Float64}}}
  @test @inferred(NDTensors.contract(T1, labelsT1, T2, labelsT2, labelsR)) isa DenseTensor{Float64, 2, Tuple{Index{Int64}, Index{Int64}}, Dense{Float64, Vector{Float64}}}
  @test @inferred(NDTensors.contract!!(R, labelsR, T1, labelsT1, T2, labelsT2)) isa DenseTensor{Float64, 2, Tuple{Index{Int64}, Index{Int64}}, Dense{Float64, Vector{Float64}}}

  A = Base.ReshapedArray(randn(4), (2, 2), ())
  B = Base.ReshapedArray(randn(4), (2, 2), ())
  @inferred NDTensors._contract_scalar_perm!(B, A, (2, 1), 1.0, 1.0)
  @inferred NDTensors._contract_scalar_perm!(B, A, (2, 1), 1.0, 1)
end

@testset "NDTensors BlockSparse contract" begin
  i = Index([QN(0) => 2, QN(1) => 2])
  IT1 = randomITensor(i'', dag(i)')
  IT2 = randomITensor(i', dag(i))
  IR = randomITensor(i'', dag(i))
  T1, T2, R = Tensor.((IT1, IT2, IR))

  labelsT1 = (1, -1)
  labelsT2 = (-1, 2)
  labelsR = (1, 2)

  @test @inferred(NDTensors.contraction_output(T1, labelsT1, T2, labelsT2, labelsR)) isa Tuple{BlockSparseTensor, Vector{Tuple{Block{2}, Block{2}, Block{2}}}}
  @test @inferred(NDTensors.contract(T1, labelsT1, T2, labelsT2, labelsR)) isa BlockSparseTensor
  # TODO: this function doesn't exist yet
  #@test @inferred(NDTensors.contract!!(R, labelsR, T1, labelsT1, T2, labelsT2)) isa BlockSparseTensor

  b = Block(1, 1)
  B1 = T1[b]
  B2 = T2[b]
  BR = R[b]
  @test @inferred(NDTensors.contract!(BR, labelsR, B1, labelsT1, B2, labelsT2, 1.0, 0.0)) isa DenseTensor
end

