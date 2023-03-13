using ITensors
using ITensors.NDTensors
using Test

@testset "ITensors priming and tagging" begin
  i = Index(2)
  T1 = randomITensor(i'', i')
  T2 = randomITensor(i', i)

  @test inds(@inferred(adjoint(T1))) == (i''', i'')
  @test inds(@inferred(prime(T1, 2))) == (i'''', i''')
  @test inds(@inferred(addtags(T1, "x"))) == (addtags(i, "x")'', addtags(i, "x")')
  @test inds(@inferred(T1 * T2)) == (i'', i)

  @test @inferred(order(T1)) == 2
  @test @inferred(ndims(T1)) == 2
  @test @inferred(dim(T1)) == 4
  @test @inferred(maxdim(T1)) == 2
end

@testset "NDTensors Dense contract" begin
  i = Index(2)
  T1 = randomTensor((i'', i'))
  T2 = randomTensor((i', i))
  R = randomTensor((i'', i))

  labelsT1 = (1, -1)
  labelsT2 = (-1, 2)
  labelsR = (1, 2)

  @test @inferred(NDTensors.contraction_output(T1, labelsT1, T2, labelsT2, labelsR)) isa
    DenseTensor{
    Float64,2,Tuple{Index{Int64},Index{Int64}},Dense{Float64,Vector{Float64}}
  }
  @test @inferred(NDTensors.contract(T1, labelsT1, T2, labelsT2, labelsR)) isa DenseTensor{
    Float64,2,Tuple{Index{Int64},Index{Int64}},Dense{Float64,Vector{Float64}}
  }
  @test @inferred(NDTensors.contract!!(R, labelsR, T1, labelsT1, T2, labelsT2)) isa
    DenseTensor{
    Float64,2,Tuple{Index{Int64},Index{Int64}},Dense{Float64,Vector{Float64}}
  }

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

  indsR = @inferred(
    NDTensors.contract_inds(inds(T1), labelsT1, inds(T2), labelsT2, labelsR)
  )
  @test indsR isa Tuple{Index{Vector{Pair{QN,Int}}},Index{Vector{Pair{QN,Int}}}}

  TensorT = @inferred(NDTensors.contraction_output_type(typeof(T1), typeof(T2), indsR))
  @test TensorT <: Tensor{Float64,2,BlockSparse{Float64,Vector{Float64},2},typeof(indsR)}

  blockoffsetsR, contraction_plan = @inferred(
    NDTensors.contract_blockoffsets(
      blockoffsets(T1),
      inds(T1),
      labelsT1,
      blockoffsets(T2),
      inds(T2),
      labelsT2,
      indsR,
      labelsR,
    )
  )
  @test blockoffsetsR isa BlockOffsets{2}
  @test contraction_plan isa Vector{Tuple{Block{2},Block{2},Block{2}}}

  @test @inferred(NDTensors.contraction_output(T1, labelsT1, T2, labelsT2, labelsR)) isa
    Tuple{BlockSparseTensor,Vector{Tuple{Block{2},Block{2},Block{2}}}}

  if VERSION ≥ v"1.7"
    # Only properly inferred in Julia 1.7 and later
    @test @inferred(NDTensors.contract(T1, labelsT1, T2, labelsT2, labelsR)) isa
      BlockSparseTensor
  end

  # TODO: this function doesn't exist yet
  #@test @inferred(NDTensors.contract!!(R, labelsR, T1, labelsT1, T2, labelsT2)) isa BlockSparseTensor

  b = Block(1, 1)
  B1 = T1[b]
  B2 = T2[b]
  BR = R[b]
  @test @inferred(
    NDTensors.contract!(BR, labelsR, B1, labelsT1, B2, labelsT2, 1.0, 0.0)
  ) isa DenseTensor
end
