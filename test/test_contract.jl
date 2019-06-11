using ITensors,
      LinearAlgebra, # For tr()
      Combinatorics, # For permutations()
      Test

digits(::Type{T},i,j,k) where {T} = T(i*10^2+j*10+k)

@testset "ITensor $T Contractions" for T ∈ (Float64,) #ComplexF64)
  mi,mj,mk,ml,mα = 2,3,4,5,6,7
  i = Index(mi,"i")
  j = Index(mj,"j")
  k = Index(mk,"k")
  l = Index(ml,"l")
  α = Index(mα,"α") 
  @testset "Test contract ITensors" begin
    A = randomITensor(T)
    B = randomITensor(T)
    Ai = randomITensor(T,i)
    Bi = randomITensor(T,i)
    Aj = randomITensor(T,j)
    Aij = randomITensor(T,i,j)
    Bij = randomITensor(T,i,j)
    Aik = randomITensor(T,i,k)
    Ajk = randomITensor(T,j,k)
    Ajl = randomITensor(T,j,l)
    Akl = randomITensor(T,k,l)
    Aijk = randomITensor(T,i,j,k)
    Ajkl = randomITensor(T,j,k,l)
    Aikl = randomITensor(T,i,k,l)
    Aklα = randomITensor(T,k,l,α)
    Aijkl = randomITensor(T,i,j,k,l)
    @testset "Test contract ITensor (Scalar*Scalar -> Scalar)" begin
      C = A*B
      @test scalar(C)≈scalar(A)*scalar(B)
    end
    @testset "Test contract ITensor (Scalar*Vector -> Vector)" begin
      C = A*Ai
      @test Array(C)≈scalar(A)*Array(Ai)
    end
    @testset "Test contract ITensor (Vector*Scalar -> Vector)" begin
      C = Aj*A
      @test Array(C)≈scalar(A)*Array(Aj)
    end
    @testset "Test contract ITensors (Vectorᵀ*Vector -> Scalar)" begin
      C = Ai*Bi
      CArray = transpose(Array(Ai))*Array(Bi)
      @test CArray≈scalar(C)
    end
    @testset "Test contract ITensors (Vector*Vectorᵀ -> Matrix)" begin
      C = Ai*Aj
      for ii ∈ 1:dim(i), jj ∈ 1:dim(j)
        @test C[i(ii),j(jj)] ≈ Ai[i(ii)]*Aj[j(jj)]
      end
    end
    @testset "Test contract ITensors (Matrix*Scalar -> Matrix)" begin
      Aij = permute(Aij,i,j)
      C = Aij*A
      @test Array(permute(C,i,j))≈scalar(A)*Array(Aij)
    end
    @testset "Test contract ITensors (Matrix*Vector -> Vector)" begin
      Aij = permute(Aij,i,j)
      C = Aij*Aj
      CArray = Array(permute(Aij,i,j))*Array(Aj)
      @test CArray≈Array(C)
    end
    @testset "Test contract ITensors (Matrixᵀ*Vector -> Vector)" begin
      Aij = permute(Aij,j,i)
      C = Aij*Aj
      CArray = transpose(Array(Aij))*Array(Aj)
      @test CArray≈Array(C)
    end
    @testset "Test contract ITensors (Vector*Matrix -> Vector)" begin
      Aij = permute(Aij,i,j)
      C = Ai*Aij
      CArray = transpose(transpose(Array(Ai))*Array(Aij))
      @test CArray≈Array(C)
    end
    @testset "Test contract ITensors (Vector*Matrixᵀ -> Vector)" begin
      Aij = permute(Aij,j,i)
      C = Ai*Aij
      CArray = transpose(transpose(Array(Ai))*transpose(Array(Aij)))
      @test CArray≈Array(C)
    end
    @testset "Test contract ITensors (Matrix*Matrix -> Scalar)" begin
      Aij = permute(Aij,i,j)
      Bij = permute(Bij,i,j)
      C = Aij*Bij
      CArray = tr(Array(Aij)*transpose(Array(Bij)))
      @test CArray≈scalar(C)
    end
    @testset "Test contract ITensors (Matrix*Matrix -> Matrix)" begin
      Aij = permute(Aij,i,j)
      Ajk = permute(Ajk,j,k)
      C = Aij*Ajk
      CArray = Array(Aij)*Array(Ajk)
      @test CArray≈Array(C)
    end
    @testset "Test contract ITensors (Matrixᵀ*Matrix -> Matrix)" begin
      Aij = permute(Aij,j,i)
      Ajk = permute(Ajk,j,k)
      C = Aij*Ajk
      CArray = transpose(Array(Aij))*Array(Ajk)
      @test CArray≈Array(C)
    end
    @testset "Test contract ITensors (Matrix*Matrixᵀ -> Matrix)" begin
      Aij = permute(Aij,i,j)
      Ajk = permute(Ajk,k,j)
      C = Aij*Ajk
      CArray = Array(Aij)*transpose(Array(Ajk))
      @test CArray≈Array(C)
    end
    @testset "Test contract ITensors (Matrixᵀ*Matrixᵀ -> Matrix)" begin
      Aij = permute(Aij,j,i)
      Ajk = permute(Ajk,k,j)
      C = Aij*Ajk
      CArray = transpose(Array(Aij))*transpose(Array(Ajk))
      @test CArray≈Array(C)
    end
    @testset "Test contract ITensors (Matrix⊗Matrix -> 4-tensor)" begin
      C = Aij*Akl
      for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k), ll ∈ 1:dim(l)
        @test C[i(ii),j(jj),k(kk),l(ll)] ≈ Aij[i(ii),j(jj)]*Akl[k(kk),l(ll)]
      end
    end
    @testset "Test contract ITensors (3-Tensor*Scalar -> 3-Tensor)" begin
      Aijk = permute(Aijk,i,j,k)
      C = Aijk*A
      @test Array(permute(C,i,j,k))==scalar(A)*Array(Aijk)
    end
    @testset "Test contract ITensors (3-Tensor*Vector -> Matrix)" begin
      Aijk = permute(Aijk,i,j,k)
      C = Aijk*Ai
      CArray = reshape(reshape(Array(permute(Aijk,j,k,i)),dim(j)*dim(k),dim(i))*Array(Ai),dim(j),dim(k))
      @test CArray≈Array(permute(C,j,k))
    end
    @testset "Test contract ITensors (Vector*3-Tensor -> Matrix)" begin
      Aijk = permute(Aijk,i,j,k)
      C = Aj*Aijk
      CArray = reshape(transpose(Array(Aj))*reshape(Array(permute(Aijk,j,i,k)),dim(j),dim(i)*dim(k)),dim(i),dim(k))
      @test CArray≈Array(permute(C,i,k))
    end
    @testset "Test contract ITensors (3-Tensor*Matrix -> Vector)" begin
      Aijk = permute(Aijk,i,j,k)
      Aik = permute(Aik,i,k)
      C = Aijk*Aik
      CArray = reshape(Array(permute(Aijk,j,i,k)),dim(j),dim(i)*dim(k))*vec(Array(Aik))
      @test CArray≈Array(C)
    end
    @testset "Test contract ITensors (3-Tensor*Matrix -> 3-Tensor)" begin
      Aijk = permute(Aijk,i,j,k)
      Ajl = permute(Ajl,j,l)
      C = Aijk*Ajl
      CArray = reshape(reshape(Array(permute(Aijk,i,k,j)),dim(i)*dim(k),dim(j))*Array(Ajl),dim(i),dim(k),dim(l))
      @test CArray≈Array(permute(C,i,k,l))
    end
    @testset "Test contract ITensors (Matrix*3-Tensor -> 3-Tensor)" begin
      Aijk = permute(Aijk,i,j,k)
      Akl = permute(Akl,k,l)
      C = Akl*Aijk
      CArray = reshape(Array(permute(Akl,l,k))*reshape(Array(permute(Aijk,k,i,j)),dim(k),dim(i)*dim(j)),dim(l),dim(i),dim(j))
      @test CArray≈Array(permute(C,l,i,j))
    end
    @testset "Test contract ITensors (3-Tensor*3-Tensor -> 3-Tensor)" begin
      Aijk = permute(Aijk,i,j,k)
      Ajkl = permute(Ajkl,j,k,l)
      C = Aijk*Ajkl
      CArray = reshape(Array(permute(Aijk,i,j,k)),dim(i),dim(j)*dim(k))*reshape(Array(permute(Ajkl,j,k,l)),dim(j)*dim(k),dim(l))
      @test CArray≈Array(permute(C,i,l))
    end
    @testset "Test contract ITensors (3-Tensor*3-Tensor -> 3-Tensor)" begin
      for inds_ijk ∈ permutations([i,j,k]), inds_jkl ∈ permutations([j,k,l])
        Aijk = permute(Aijk,inds_ijk...)
        Ajkl = permute(Ajkl,inds_jkl...)
        C = Ajkl*Aijk
        CArray = reshape(Array(permute(Ajkl,l,j,k)),dim(l),dim(j)*dim(k))*reshape(Array(permute(Aijk,j,k,i)),dim(j)*dim(k),dim(i))
        @test CArray≈Array(permute(C,l,i))
      end
    end
    @testset "Test contract ITensors (4-Tensor*3-Tensor -> 1-Tensor)" begin
      for inds_ijkl ∈ permutations([i,j,k,l]), inds_jkl ∈ permutations([j,k,l])
        Aijkl = permute(Aijkl,inds_ijkl...)
        Ajkl = permute(Ajkl,inds_jkl...) 
        C = Ajkl*Aijkl
        CArray = reshape(Array(permute(Ajkl,j,k,l)),1,dim(j)*dim(k)*dim(l))*reshape(Array(permute(Aijkl,j,k,l,i)),dim(j)*dim(k)*dim(l),dim(i))
        @test vec(CArray)≈Array(permute(C,i))
      end
    end
    @testset "Test contract ITensors (4-Tensor*3-Tensor -> 3-Tensor)" begin
      for inds_ijkl ∈ permutations([i,j,k,l]), inds_klα ∈ permutations([k,l,α])
        Aijkl = permute(Aijkl,inds_ijkl...)
        Aklα = permute(Aklα,inds_klα...)
        C = Aklα*Aijkl
        CArray = reshape(reshape(Array(permute(Aklα,α,k,l)),dim(α),dim(k)*dim(l))*reshape(Array(permute(Aijkl,k,l,i,j)),dim(k)*dim(l),dim(i)*dim(j)),dim(α),dim(i),dim(j))
        @test CArray≈Array(permute(C,α,i,j))
      end
    end
  end # End contraction testset
end

