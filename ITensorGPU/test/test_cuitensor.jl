using ITensors,
  ITensorGPU,
  LinearAlgebra, # For tr()
  Combinatorics, # For permutations()
  Random,
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
  @testset "Constructor" begin
    A = cuITensor(one(SType), i, j, k)
    @test collect(CuArray(A, i, j, k)) == ones(SType, dim(i), dim(j), dim(k))
    A = randomCuITensor(IndexSet(i, j, k))
    @test inds(A) == IndexSet(i, j, k)
    @test ITensorGPU.storage(A) isa ITensorGPU.CuDense
    Aarr = rand(SType, dim(i) * dim(j) * dim(k))
    @test cpu(ITensor(Aarr, i, j, k)) == cpu(cuITensor(Aarr, i, j, k))
    @test cuITensor(SType, i, j, k) isa ITensor
    @test storage(cuITensor(SType, i, j, k)) isa ITensorGPU.CuDense{SType}
    @test vec(collect(CuArray(ITensor(Aarr, i, j, k), i, j, k))) == Aarr
  end
  @testset "Test permute(cuITensor,Index...)" begin
    CA = randomCuITensor(SType, i, k, j)
    permCA = permute(CA, k, j, i)
    permA = cpu(permCA)
    @test k == inds(permA)[1]
    @test j == inds(permA)[2]
    @test i == inds(permA)[3]
    A = cpu(CA)
    for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k)
      @test A[k => kk, i => ii, j => jj] == permA[i => ii, j => jj, k => kk]
    end
  end
  @testset "Test permute(cuITensor,Index...) for large tensors" begin
    inds = [Index(2) for ii in 1:14]
    A = randomITensor(SType, IndexSet(inds))
    CA = cuITensor(A)
    for shuffle_count in 1:20
      perm_inds = shuffle(inds)
      permCA = permute(CA, perm_inds...)
      permA = cpu(permCA)
      pA = permute(A, perm_inds...)
      for ci in CartesianIndices(pA)
        @test pA[ci] == permA[ci]
      end
    end
  end
  #=@testset "Test scalar(cuITensor)" begin
    x = SType(34)
    A = randomCuITensor(a)
    @test x==scalar(A)
  end=#
  @testset "Test CuVector(cuITensor)" begin
    v = CuVector(ones(SType, dim(a)))
    A = cuITensor(v, a)
    @test v == CuVector(A)
  end
  @testset "Test CuMatrix(cuITensor)" begin
    v = CuMatrix(ones(SType, dim(a), dim(l)))
    A = cuITensor(vec(v), a, l)
    @test v == CuMatrix(A, a, l)
    A = cuITensor(vec(v), a, l)
    @test v == CuMatrix(A)
    A = cuITensor(vec(v), a, l)
    @test v == CuArray(A, a, l)
    @test v == CuArray(A)
  end
  @testset "Test norm(cuITensor)" begin
    A = randomCuITensor(SType, i, j, k)
    B = dag(A) * A
    @test norm(A) ≈ sqrt(real(scalar(B)))
  end
  @testset "Test complex(cuITensor)" begin
    A = randomCuITensor(SType, i, j, k)
    cA = complex(A)
    @test complex.(CuArray(A)) == CuArray(cA)
  end
  #@testset "Test exp(cuITensor)" begin
  #  A  = randomCuITensor(SType,i,i')
  #  @test CuArray(exp(A,i,i')) ≈ exp(CuArray(A))
  #end
  @testset "Test add cuITensors" begin
    dA = randomCuITensor(SType, i, j, k)
    dB = randomCuITensor(SType, k, i, j)
    A = cpu(dA)
    B = cpu(dB)
    C = cpu(dA + dB)
    @test CuArray(permute(C, i, j, k)) ==
      CuArray(permute(A, i, j, k)) + CuArray(permute(B, i, j, k))
    for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k)
      @test C[i => ii, j => jj, k => kk] ==
        A[j => jj, i => ii, k => kk] + B[i => ii, k => kk, j => jj]
    end
  end

  @testset "Test factorizations of a cuITensor" begin
    A = randomCuITensor(SType, i, j, k, l)
    @testset "Test SVD of a cuITensor" begin
      U, S, V = svd(A, (j, l))
      u = commonind(U, S)
      v = commonind(S, V)
      @test cpu(A) ≈ cpu(U * S * V)
      @test cpu(U * dag(prime(U, u))) ≈ δ(SType, u, u') rtol = 1e-14
      @test cpu(V * dag(prime(V, v))) ≈ δ(SType, v, v') rtol = 1e-14
    end

    A = randomCuITensor(SType, i, j, k, l)
    @testset "Test SVD consistency between CPU and GPU" begin
      U_gpu, S_gpu, V_gpu = svd(A, (j, l))
      U_cpu, S_cpu, V_cpu = svd(cpu(A), (j, l))
      @test cpu(U_gpu) * cpu(S_gpu) * cpu(V_gpu) ≈ U_cpu * S_cpu * V_cpu
    end

    #=@testset "Test SVD truncation" begin 
        M = randn(4,4)
        (U,s,V) = svd(M)
        ii = Index(4)
        jj = Index(4)
        S = Diagonal(s)
        T = cuITensor(vec(CuArray(U*S*V')),IndexSet(ii,jj))
        (U,S,V) = svd(T,ii;maxm=2)
        @test norm(U*S*V-T)≈sqrt(s[3]^2+s[4]^2)
    end=#

    @testset "Test QR decomposition of a cuITensor" begin
      Q, R = qr(A, (i, l))
      q = commonind(Q, R)
      @test cpu(A) ≈ cpu(Q * R)
      @test cpu(Q * dag(prime(Q, q))) ≈ δ(SType, q, q') atol = 1e-14
    end

    #=@testset "Test polar decomposition of a cuITensor" begin
      U,P = polar(A,(k,l))
      @test cpu(A)≈cpu(U*P)
    end=#

  end # End ITensor factorization testset
end # End Dense storage test
