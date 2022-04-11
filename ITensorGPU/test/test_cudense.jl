using ITensors,
  ITensorGPU,
  LinearAlgebra, # For tr()
  Combinatorics, # For permutations()
  CUDA,
  Test

# gpu tests!
@testset "cuITensor, Dense{$SType} storage" for SType in (Float64, ComplexF64)
  mi, mj, mk, ml, ma = 2, 3, 4, 5, 6, 7
  i = Index(mi, "i")
  j = Index(mj, "j")
  k = Index(mk, "k")
  l = Index(ml, "l")
  a = Index(ma, "a")
  @testset "Test add CuDense" begin
    A = [SType(1.0) for ii in 1:dim(i), jj in 1:dim(j)]
    dA = ITensorGPU.CuDense{SType,CuVector{SType}}(SType(1.0), dim(i) * dim(j))
    B = [SType(2.0) for ii in 1:dim(i), jj in 1:dim(j)]
    dB = ITensorGPU.CuDense{SType,CuVector{SType}}(SType(2.0), dim(i) * dim(j))
    dC = +(dA, IndexSet(i, j), dB, IndexSet(j, i))
    hC = collect(dC)
    @test collect(A + B) ≈ hC
  end
  @testset "Test subtract CuDense" begin
    A = [SType(1.0) for ii in 1:dim(i), jj in 1:dim(j)]
    dA = ITensorGPU.CuDense{SType,CuVector{SType}}(SType(1.0), dim(i) * dim(j))
    B = [SType(2.0) for ii in 1:dim(i), jj in 1:dim(j)]
    dB = ITensorGPU.CuDense{SType,CuVector{SType}}(SType(2.0), dim(i) * dim(j))
    dC = -(dA, IndexSet(i, j), dB, IndexSet(i, j))
    hC = collect(dC)
    @test A - B ≈ hC
  end
  @testset "Test permute CuDense" begin
    A = [SType(ii * jj) for ii in 1:dim(i), jj in 1:dim(j)]
    dA = ITensorGPU.CuDense{SType,CuVector{SType}}(NDTensors.Dense(vec(A)))
    B = [SType(0.0) for ii in 1:dim(j), jj in 1:dim(j)]
    dB = ITensorGPU.CuDense{SType,CuVector{SType}}(SType(0.0), dim(i) * dim(j))
    dC = permute!(dB, IndexSet(j, i), dA, IndexSet(i, j))
    hC = cpu(dC)
    @test transpose(A) == hC
  end
  @testset "Test move CuDense on/off GPU" begin
    A = [SType(1.0) for ii in 1:dim(i), jj in 1:dim(j)]
    dA = ITensorGPU.CuDense{SType,CuVector{SType}}(NDTensors.Dense(vec(A)))
    dB = ITensorGPU.Dense{SType,Vector{SType}}(dA)
    @test NDTensors.data(dB) == vec(A)
  end
  @testset "Test basic CuDense features" begin
    @test NDTensors.Dense{SType,CuVector{SType}}(10) isa ITensorGPU.CuDense{SType}
    @test complex(NDTensors.Dense{SType,CuVector{SType}}) ==
      NDTensors.Dense{complex(SType),CuVector{complex(SType)}}
  end
  if SType == Float64
    @testset "Test CuDense complex" begin
      A = CUDA.rand(SType, dim(i) * dim(j))
      dA = ITensorGPU.CuDense{SType,CuVector{SType}}(A)
      dC = complex(dA)
      @test typeof(dC) !== typeof(dA)
      cdC = CuArray(dC)
      hC = collect(cdC)
      ccA = complex.(A)
      @test hC == collect(ccA)
    end
  end
end # End Dense storage test
