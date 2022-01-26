using ITensors,
  ITensorGPU,
  LinearAlgebra, # For tr()
  Combinatorics, # For permutations()
  Random,
  CUDA,
  Test

@testset "cuITensor $T Contractions" for T in (Float64, ComplexF64)
  mi, mj, mk, ml, ma = 2, 3, 4, 5, 6, 7
  i = Index(mi, "i")
  j = Index(mj, "j")
  k = Index(mk, "k")
  l = Index(ml, "l")
  a = Index(ma, "a")
  @testset "Test contract cuITensors" begin
    A = cuITensor(randomITensor(T))
    B = cuITensor(randomITensor(T))
    Ai = cuITensor(randomITensor(T, i))
    Bi = cuITensor(randomITensor(T, i))
    Aj = cuITensor(randomITensor(T, j))
    Aij = cuITensor(randomITensor(T, i, j))
    Aji = cuITensor(randomITensor(T, j, i))
    Bij = cuITensor(randomITensor(T, i, j))
    Aik = cuITensor(randomITensor(T, i, k))
    Ajk = cuITensor(randomITensor(T, j, k))
    Ajl = cuITensor(randomITensor(T, j, l))
    Akl = cuITensor(randomITensor(T, k, l))
    Aijk = cuITensor(randomITensor(T, i, j, k))
    Ajkl = cuITensor(randomITensor(T, j, k, l))
    Aikl = cuITensor(randomITensor(T, i, k, l))
    Akla = cuITensor(randomITensor(T, k, l, a))
    Aijkl = cuITensor(randomITensor(T, i, j, k, l))
    @testset "Test contract cuITensor (Scalar*Scalar -> Scalar)" begin
      C = A * B
      @test scalar(C) ≈ scalar(A) * scalar(B)
      C = cuITensor(T(2.0)) * cuITensor(T(2.0))
      @test scalar(C) ≈ T(4.0)
    end
    @testset "Test contract cuITensor (Scalar*Vector -> Vector)" begin
      C = A * Ai
      @test cpu(C) ≈ scalar(A) * cpu(Ai)
      C = cuITensor(T(2.0)) * Ai
      @test cpu(C) ≈ T(2.0) * cpu(Ai)
      C = Ai * cuITensor(T(2.0))
      @test cpu(C) ≈ T(2.0) * cpu(Ai)
    end
    @testset "Test contract cuITensor (Vector*Scalar -> Vector)" begin
      C = Aj * A
      @test cpu(C) ≈ scalar(A) * cpu(Aj)
    end
    @testset "Test contract cuITensors (Vectorᵀ*Vector -> Scalar)" begin
      C = Ai * Bi
      Ccpu = cpu(Ai) * cpu(Bi)
      @test scalar(Ccpu) ≈ scalar(C)
    end
    @testset "Test contract cuITensors (Vector*Vectorᵀ -> Matrix)" begin
      C = Ai * Aj
      Ccpu = cpu(Ai) * cpu(Aj)
      @test Ccpu ≈ cpu(permute(C, i, j))
    end
    @testset "Test contract cuITensors (Matrix*Scalar -> Matrix)" begin
      Aij = permute(Aij, i, j)
      C = Aij * A
      @test cpu(permute(C, i, j)) ≈ scalar(A) * cpu(Aij)
    end
    @testset "Test contract cuITensors (Matrixᵀ*Vector -> Vector)" begin
      cAij = permute(copy(Aij), j, i)
      Ccpu = cpu(Aij) * cpu(Aj)
      C = cAij * Aj
      @test Ccpu ≈ cpu(C)
    end
    @testset "Test contract cuITensors (Matrix*Vector -> Vector)" begin
      cpAij = permute(copy(Aij), i, j)
      Ccpu = cpu(cpAij) * cpu(Aj)
      C = copy(cpAij) * copy(Aj)
      @test Ccpu ≈ cpu(C)
    end
    @testset "Test contract cuITensors (Vector*Matrix -> Vector)" begin
      Aij = permute(Aij, i, j)
      C = Ai * Aij
      Ccpu = cpu(Ai) * cpu(Aij)
      @test Ccpu ≈ cpu(C)
    end
    @testset "Test contract cuITensors (Vector*Matrixᵀ -> Vector)" begin
      C = Ai * Aji
      Ccpu = cpu(Ai) * cpu(Aji)
      @test Ccpu ≈ cpu(C)
    end
    @testset "Test contract cuITensors (Matrix*Matrix -> Scalar)" begin
      Aij = permute(Aij, i, j)
      Bij = permute(Bij, i, j)
      C = Aij * Bij
      Ccpu = cpu(Aij) * cpu(Bij)
      @test scalar(Ccpu) ≈ scalar(C)
    end
    @testset "Test contract cuITensors (Matrix*Matrix -> Matrix)" begin
      Aij = permute(Aij, i, j)
      Ajk = permute(Ajk, j, k)
      C = Aij * Ajk
      Ccpu = cpu(Aij) * cpu(Ajk)
      @test Ccpu ≈ cpu(C)
    end
    @testset "Test contract cuITensors (Matrixᵀ*Matrix -> Matrix)" begin
      Aij = permute(Aij, j, i)
      Ajk = permute(Ajk, j, k)
      C = Aij * Ajk
      Ccpu = cpu(Aij) * cpu(Ajk)
      @test Ccpu ≈ cpu(C)
    end
    @testset "Test contract cuITensors (Matrix*Matrixᵀ -> Matrix)" begin
      Aij = permute(Aij, i, j)
      Ajk = permute(Ajk, k, j)
      C = Aij * Ajk
      Ccpu = cpu(Aij) * cpu(Ajk)
      @test Ccpu ≈ cpu(C)
    end
    @testset "Test contract cuITensors (Matrixᵀ*Matrixᵀ -> Matrix)" begin
      Aij = permute(Aij, j, i)
      Ajk = permute(Ajk, k, j)
      C = Aij * Ajk
      Ccpu = cpu(Aij) * cpu(Ajk)
      @test Ccpu ≈ cpu(C)
    end
    @testset "Test contract cuITensors (3-Tensor*Scalar -> 3-Tensor)" begin
      Aijk = permute(Aijk, i, j, k)
      C = Aijk * A
      @test cpu(permute(C, i, j, k)) ≈ scalar(A) * cpu(Aijk)
    end
    @testset "Test contract cuITensors (3-Tensor*Vector -> Matrix)" begin
      cAijk = permute(copy(Aijk), i, j, k)
      C = cAijk * Ai
      Ccpu = cpu(cAijk) * cpu(Ai)
      @test Ccpu ≈ cpu(C)
    end
    @testset "Test contract cuITensors (Vector*3-Tensor -> Matrix)" begin
      Aijk = permute(Aijk, i, j, k)
      C = Aj * Aijk
      Ccpu = cpu(Aj) * cpu(Aijk)
      @test Ccpu ≈ cpu(C)
    end
    @testset "Test contract cuITensors (3-Tensor*Matrix -> Vector)" begin
      Aijk = permute(Aijk, i, j, k)
      Aik = permute(Aik, i, k)
      C = Aijk * Aik
      Ccpu = cpu(Aijk) * cpu(Aik)
      @test Ccpu ≈ cpu(C)
    end
    @testset "Test contract cuITensors (3-Tensor*Matrix -> 3-Tensor)" begin
      Aijk = permute(Aijk, i, j, k)
      Ajl = permute(Ajl, j, l)
      C = Aijk * Ajl
      Ccpu = cpu(Aijk) * cpu(Ajl)
      @test Ccpu ≈ cpu(permute(C, i, k, l))
    end
    @testset "Test contract cuITensors (Matrix*3-Tensor -> 3-Tensor)" begin
      Aijk = permute(Aijk, i, j, k)
      Akl = permute(Akl, k, l)
      C = Akl * Aijk
      Ccpu = cpu(Aijk) * cpu(Akl)
      @test Ccpu ≈ cpu(permute(C, l, i, j))
    end
    @testset "Test contract cuITensors (3-Tensor*3-Tensor -> 3-Tensor)" begin
      Aijk = permute(Aijk, i, j, k)
      Ajkl = permute(Ajkl, j, k, l)
      C = Aijk * Ajkl
      Ccpu = cpu(Aijk) * cpu(Ajkl)
      @test Ccpu ≈ cpu(permute(C, i, l))
    end
    @testset "Test contract cuITensors (3-Tensor*3-Tensor -> 3-Tensor)" begin
      for inds_ijk in permutations([i, j, k]), inds_jkl in permutations([j, k, l])
        Aijk = permute(Aijk, inds_ijk...)
        Ajkl = permute(Ajkl, inds_jkl...)
        C = Ajkl * Aijk
        Ccpu = cpu(Ajkl) * cpu(Aijk)
        @test Ccpu ≈ cpu(C)
      end
    end
    @testset "Test contract cuITensors (4-Tensor*3-Tensor -> 3-Tensor)" begin
      for inds_ijkl in permutations([i, j, k, l]), inds_kla in permutations([k, l, a])
        Aijkl = permute(Aijkl, inds_ijkl...)
        Akla = permute(Akla, inds_kla...)
        C = Akla * Aijkl
        Ccpu = cpu(Akla) * cpu(Aijkl)
        @test Ccpu ≈ cpu(C)
      end
    end
    @testset "Test contract cuITensors (4-Tensor*3-Tensor -> 1-Tensor)" begin
      for inds_ijkl in permutations([i, j, k, l]), inds_jkl in permutations([j, k, l])
        Aijkl = permute(Aijkl, inds_ijkl...)
        Ajkl = permute(Ajkl, inds_jkl...)
        C = Ajkl * Aijkl
        Ccpu = cpu(Ajkl) * cpu(Aijkl)
        @test Ccpu ≈ cpu(C)
      end
    end
    @testset "Test supersized contract cuITensors (14-Tensor*14-Tensor -> 14-Tensor)" begin
      a_only_inds = [Index(2) for ii in 1:7]
      b_only_inds = [Index(2) for ii in 1:7]
      shared_inds = [Index(2) for ii in 1:7]
      A = randomITensor(IndexSet(vcat(a_only_inds, shared_inds)))
      B = randomITensor(IndexSet(vcat(b_only_inds, shared_inds)))
      cA = cuITensor(A)
      cB = cuITensor(B)
      inds_a = vcat(a_only_inds, shared_inds)
      inds_b = vcat(b_only_inds, shared_inds)
      cA_ = permute(cA, inds_a...)
      cB_ = permute(cB, inds_b...)
      @disable_warn_order begin
        C = cA_ * cB_
        Ccpu = cpu(cA_) * cpu(cB_)
      end
      @test Ccpu ≈ cpu(C)
      for shuffles in 1:1 # too many permutations to test all
        inds_a = shuffle(vcat(a_only_inds, shared_inds))
        inds_b = shuffle(vcat(b_only_inds, shared_inds))
        cA_ = permute(cA, inds_a...)
        cB_ = permute(cB, inds_b...)
        @disable_warn_order begin
          C = cA_ * cB_
          Ccpu = cpu(cA_) * cpu(cB_)
        end
        @test Ccpu ≈ cpu(C)
      end
    end
  end # End contraction testset
end
