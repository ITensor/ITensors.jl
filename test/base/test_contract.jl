using ITensors
using Test
using Combinatorics: Combinatorics

digits(::Type{T}, i, j, k) where {T} = T(i * 10^2 + j * 10 + k)

@testset "ITensor $T Contractions" for T in (Float64, ComplexF64)
  mi, mj, mk, ml, mα = 2, 3, 4, 5, 6, 7
  i = Index(mi, "i")
  j = Index(mj, "j")
  k = Index(mk, "k")
  l = Index(ml, "l")
  α = Index(mα, "alpha")
  @testset "Test contract ITensors" begin
    A = randomITensor(T)
    B = randomITensor(T)
    Ai = randomITensor(T, i)
    Bi = randomITensor(T, i)
    Aj = randomITensor(T, j)
    Aij = randomITensor(T, i, j)
    Bij = randomITensor(T, i, j)
    Aik = randomITensor(T, i, k)
    Ajk = randomITensor(T, j, k)
    Ajl = randomITensor(T, j, l)
    Akl = randomITensor(T, k, l)
    Aijk = randomITensor(T, i, j, k)
    Ajkl = randomITensor(T, j, k, l)
    Aikl = randomITensor(T, i, k, l)
    Aklα = randomITensor(T, k, l, α)
    Aijkl = randomITensor(T, i, j, k, l)
    @testset "Test contract ITensor (Scalar*Scalar -> Scalar)" begin
      C = A * B
      @test scalar(C) ≈ scalar(A) * scalar(B)
    end
    @testset "Test contract ITensor (Scalar*Vector -> Vector)" begin
      C = A * Ai
      @test array(C) ≈ scalar(A) * array(Ai)
    end
    @testset "Test contract ITensor (Vector*Scalar -> Vector)" begin
      C = Aj * A
      @test array(C) ≈ scalar(A) * array(Aj)
    end
    @testset "Test contract ITensors (Vectorᵀ*Vector -> Scalar)" begin
      C = Ai * Bi
      CArray = transpose(array(Ai)) * array(Bi)
      @test CArray ≈ scalar(C)
    end
    @testset "Test Matrix{ITensor} * Matrix{ITensor}" begin
      M1 = [Aij Aij; Aij Aij]
      M2 = [Ajk Ajk; Ajk Ajk]
      M12 = M1 * M2
      for x in 1:2, y in 1:2
        @test M12[x, y] ≈ 2 * Aij * Ajk
      end
    end
    @testset "Test contract ITensors (Vector*Vectorᵀ -> Matrix)" begin
      C = Ai * Aj
      for ii in 1:dim(i), jj in 1:dim(j)
        @test C[i => ii, j => jj] ≈ Ai[i => ii] * Aj[j => jj]
      end
    end
    @testset "Test contract ITensors (Matrix*Scalar -> Matrix)" begin
      Aij = permute(Aij, i, j)
      C = Aij * A
      @test array(permute(C, i, j)) ≈ scalar(A) * array(Aij)
    end
    @testset "Test contract ITensors (Matrix*Vector -> Vector)" begin
      Aij = permute(Aij, i, j)
      C = Aij * Aj
      CArray = array(permute(Aij, i, j)) * array(Aj)
      @test CArray ≈ array(C)
    end
    @testset "Test contract ITensors (Matrixᵀ*Vector -> Vector)" begin
      Aij = permute(Aij, j, i)
      C = Aij * Aj
      CArray = transpose(array(Aij)) * array(Aj)
      @test CArray ≈ array(C)
    end
    @testset "Test contract ITensors (Vector*Matrix -> Vector)" begin
      Aij = permute(Aij, i, j)
      C = Ai * Aij
      CArray = transpose(transpose(array(Ai)) * array(Aij))
      @test CArray ≈ array(C)
    end
    @testset "Test contract ITensors (Vector*Matrixᵀ -> Vector)" begin
      Aij = permute(Aij, j, i)
      C = Ai * Aij
      CArray = transpose(transpose(array(Ai)) * transpose(array(Aij)))
      @test CArray ≈ array(C)
    end
    @testset "Test contract ITensors (Matrix*Matrix -> Scalar)" begin
      Aij = permute(Aij, i, j)
      Bij = permute(Bij, i, j)
      C = Aij * Bij
      CArray = LinearAlgebra.tr(array(Aij) * transpose(array(Bij)))
      @test CArray ≈ scalar(C)
    end
    @testset "Test contract ITensors (Matrix*Matrix -> Matrix)" begin
      Aij = permute(Aij, i, j)
      Ajk = permute(Ajk, j, k)
      C = Aij * Ajk
      CArray = array(Aij) * array(Ajk)
      @test CArray ≈ array(C)
    end
    @testset "Test contract ITensors (Matrixᵀ*Matrix -> Matrix)" begin
      Aij = permute(Aij, j, i)
      Ajk = permute(Ajk, j, k)
      C = Aij * Ajk
      CArray = transpose(array(Aij)) * array(Ajk)
      @test CArray ≈ array(C)
    end
    @testset "Test contract ITensors (Matrix*Matrixᵀ -> Matrix)" begin
      Aij = permute(Aij, i, j)
      Ajk = permute(Ajk, k, j)
      C = Aij * Ajk
      CArray = array(Aij) * transpose(array(Ajk))
      @test CArray ≈ array(C)
    end
    @testset "Test contract ITensors (Matrixᵀ*Matrixᵀ -> Matrix)" begin
      Aij = permute(Aij, j, i)
      Ajk = permute(Ajk, k, j)
      C = Aij * Ajk
      CArray = transpose(array(Aij)) * transpose(array(Ajk))
      @test CArray ≈ array(C)
    end
    @testset "Test contract ITensors (Matrix⊗Matrix -> 4-tensor)" begin
      C = Aij * Akl
      for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k), ll in 1:dim(l)
        @test C[i => ii, j => jj, k => kk, l => ll] ≈
          Aij[i => ii, j => jj] * Akl[k => kk, l => ll]
      end
    end
    @testset "Test contract ITensors (3-Tensor*Scalar -> 3-Tensor)" begin
      Aijk = permute(Aijk, i, j, k)
      C = Aijk * A
      @test array(permute(C, i, j, k)) ≈ scalar(A) * array(Aijk) rtol = 1e-12
    end
    @testset "Test contract ITensors (3-Tensor*Vector -> Matrix)" begin
      Aijk = permute(Aijk, i, j, k)
      C = Aijk * Ai
      CArray = reshape(
        reshape(array(permute(Aijk, j, k, i)), dim(j) * dim(k), dim(i)) * array(Ai),
        dim(j),
        dim(k),
      )
      @test CArray ≈ array(permute(C, j, k))
    end
    @testset "Test contract ITensors (Vector*3-Tensor -> Matrix)" begin
      Aijk = permute(Aijk, i, j, k)
      C = Aj * Aijk
      CArray = reshape(
        transpose(array(Aj)) *
        reshape(array(permute(Aijk, j, i, k)), dim(j), dim(i) * dim(k)),
        dim(i),
        dim(k),
      )
      @test CArray ≈ array(permute(C, i, k))
    end
    @testset "Test contract ITensors (3-Tensor*Matrix -> Vector)" begin
      Aijk = permute(Aijk, i, j, k)
      Aik = permute(Aik, i, k)
      C = Aijk * Aik
      CArray =
        reshape(array(permute(Aijk, j, i, k)), dim(j), dim(i) * dim(k)) * vec(array(Aik))
      @test CArray ≈ array(C)
    end
    @testset "Test contract ITensors (3-Tensor*Matrix -> 3-Tensor)" begin
      Aijk = permute(Aijk, i, j, k)
      Ajl = permute(Ajl, j, l)
      C = Aijk * Ajl
      CArray = reshape(
        reshape(array(permute(Aijk, i, k, j)), dim(i) * dim(k), dim(j)) * array(Ajl),
        dim(i),
        dim(k),
        dim(l),
      )
      @test CArray ≈ array(permute(C, i, k, l))
    end
    @testset "Test contract ITensors (Matrix*3-Tensor -> 3-Tensor)" begin
      Aijk = permute(Aijk, i, j, k)
      Akl = permute(Akl, k, l)
      C = Akl * Aijk
      CArray = reshape(
        array(permute(Akl, l, k)) *
        reshape(array(permute(Aijk, k, i, j)), dim(k), dim(i) * dim(j)),
        dim(l),
        dim(i),
        dim(j),
      )
      @test CArray ≈ array(permute(C, l, i, j))
    end
    @testset "Test contract ITensors (3-Tensor*3-Tensor -> 3-Tensor)" begin
      Aijk = permute(Aijk, i, j, k)
      Ajkl = permute(Ajkl, j, k, l)
      C = Aijk * Ajkl
      CArray =
        reshape(array(permute(Aijk, i, j, k)), dim(i), dim(j) * dim(k)) *
        reshape(array(permute(Ajkl, j, k, l)), dim(j) * dim(k), dim(l))
      @test CArray ≈ array(permute(C, i, l))
    end
    @testset "Test contract ITensors (3-Tensor*3-Tensor -> 3-Tensor)" begin
      for inds_ijk in Combinatorics.permutations([i, j, k]),
        inds_jkl in Combinatorics.permutations([j, k, l])

        Aijk = permute(Aijk, inds_ijk...)
        Ajkl = permute(Ajkl, inds_jkl...)
        C = Ajkl * Aijk
        CArray =
          reshape(array(permute(Ajkl, l, j, k)), dim(l), dim(j) * dim(k)) *
          reshape(array(permute(Aijk, j, k, i)), dim(j) * dim(k), dim(i))
        @test CArray ≈ array(permute(C, l, i))
      end
    end
    @testset "Test contract ITensors (4-Tensor*3-Tensor -> 1-Tensor)" begin
      for inds_ijkl in Combinatorics.permutations([i, j, k, l]),
        inds_jkl in Combinatorics.permutations([j, k, l])

        Aijkl = permute(Aijkl, inds_ijkl...)
        Ajkl = permute(Ajkl, inds_jkl...)
        C = Ajkl * Aijkl
        CArray =
          reshape(array(permute(Ajkl, j, k, l)), 1, dim(j) * dim(k) * dim(l)) *
          reshape(array(permute(Aijkl, j, k, l, i)), dim(j) * dim(k) * dim(l), dim(i))
        @test vec(CArray) ≈ array(permute(C, i))
      end
    end
    @testset "Test contract ITensors (4-Tensor*3-Tensor -> 3-Tensor)" begin
      for inds_ijkl in Combinatorics.permutations([i, j, k, l]),
        inds_klα in Combinatorics.permutations([k, l, α])

        Aijkl = permute(Aijkl, inds_ijkl...)
        Aklα = permute(Aklα, inds_klα...)
        C = Aklα * Aijkl
        CArray = reshape(
          reshape(array(permute(Aklα, α, k, l)), dim(α), dim(k) * dim(l)) *
          reshape(array(permute(Aijkl, k, l, i, j)), dim(k) * dim(l), dim(i) * dim(j)),
          dim(α),
          dim(i),
          dim(j),
        )
        @test CArray ≈ array(permute(C, α, i, j))
      end
    end
    @testset "Test contract in-place ITensors (4-Tensor*Matrix -> 4-Tensor)" begin
      A = randomITensor(T, (j, i))
      B = randomITensor(T, (j, k, l, α))
      C = ITensor(zero(T), (i, k, α, l))
      ITensors.contract!(C, B, A, 1.0, 0.0)
      ITensors.contract!(C, B, A, 1.0, 1.0)
      D = A * B
      D .+= A * B
      @test C ≈ D
    end
  end # End contraction testset
end

@testset "Contraction conversions" begin
  @testset "Real scalar * Complex ITensor" begin
    i = Index(2, "i")
    j = Index(2, "j")
    x = rand(Float64)
    A = randomITensor(ComplexF64, i, j)
    B = x * A
    for ii in dim(i), jj in dim(j)
      @test B[i => ii, j => jj] == x * A[i => ii, j => jj]
    end
  end
  @testset "Complex scalar * Real ITensor" begin
    i = Index(2, "i")
    j = Index(2, "j")
    x = rand(ComplexF64)
    A = randomITensor(Float64, i, j)
    B = x * A
    for ii in dim(i), jj in dim(j)
      @test B[i => ii, j => jj] == x * A[i => ii, j => jj]
    end
  end
  @testset "Real ITensor * Complex ITensor" begin
    i = Index(2, "i")
    j = Index(2, "j")
    k = Index(2, "k")
    A = randomITensor(Float64, i, j)
    B = randomITensor(ComplexF64, j, k)
    C = A * B
    @test array(permute(C, i, k)) ≈ array(A) * array(B)
  end
  @testset "Complex ITensor * Real ITensor" begin
    i = Index(2, "i")
    j = Index(2, "j")
    k = Index(2, "k")
    A = randomITensor(ComplexF64, i, j)
    B = randomITensor(Float64, j, k)
    C = A * B
    @test array(permute(C, i, k)) ≈ array(A) * array(B)
  end

  @testset "Outer Product Real ITensor * Complex ITensor" begin
    i = Index(2, "i")
    j = Index(2, "j")
    A = randomITensor(Float64, i)
    B = randomITensor(ComplexF64, j)
    C = A * B
    @test array(permute(C, i, j)) ≈ kron(array(A), transpose(array(B)))
  end

  @testset "Outer Product: Complex ITensor * Real ITensor" begin
    i = Index(2, "i")
    j = Index(2, "j")
    A = randomITensor(ComplexF64, i)
    B = randomITensor(Float64, j)
    C = A * B
    @test array(permute(C, i, j)) ≈ kron(array(A), transpose(array(B)))
  end
end

nothing
