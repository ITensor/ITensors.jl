using ITensors,
      LinearAlgebra, # For tr()
      Combinatorics, # For permutations()
      Test

digits(::Type{T},i,j,k) where {T} = T(i*10^2+j*10+k)


@testset "TagSet" begin
    ts = TagSet("t1,t2,t3")
    ts2 = copy(ts)
    @test ts == ts2
end

@testset "Index" begin
    @testset "Default Index" begin
        i = Index()
        @test id(i) == 0
        @test dim(i) == 1
        @test dir(i) == Neither
        @test plev(i) == 0
        @test tags(i) == TagSet("")
    end
    @testset "Index with dim" begin
        i = Index(2)
        @test id(i) != 0
        @test dim(i) == 2
        @test dir(i) == In
        @test plev(i) == 0
        @test tags(i) == TagSet("")
    end
    @testset "Index with all args" begin
        i = Index(UInt64(1), 2, In, 1, TagSet("Link"))
        @test id(i) == 1
        @test dim(i) == 2
        @test dir(i) == In
        @test plev(i) == 1 
        @test tags(i) == TagSet("Link")
        j = copy(i)
        @test id(j) == 1
        @test dim(j) == 2
        @test dir(j) == In
        @test plev(j) == 1 
        @test tags(j) == TagSet("Link")
        @test j == i
    end
    @testset "prime" begin
        i = Index(2)
        @test plev(i) == 0
        j = prime(i, 2)
        @test plev(j) == 2 
    end
    @testset "IndexVal" begin
        i = Index(2)
        @test_throws ErrorException IndexVal(i, 4)
        @test_throws ErrorException IndexVal(i, 0)
        @test i(2) == IndexVal(i, 2)
        @test val(IndexVal(i, 1)) == 1
        @test ind(IndexVal(i, 1)) == i
        @test i == IndexVal(i, 2)
        @test IndexVal(i, 2) == i
    end
end

@testset "ITensor, Dense{$T} storage" for T ∈ (Float64,ComplexF64)
  mi,mj,mk,ml = 2,3,4,5,6
  mα = 7
  i = Index(mi,"i")
  j = Index(mj,"j")
  k = Index(mk,"k")
  l = Index(ml,"l")
  α = Index(mα,"α") 

  @testset "Set and get values with IndexVals" begin
    A = ITensor(T,i,j,k)
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      A[k(kk),j(jj),i(ii)] = digits(T,ii,jj,kk)
    end
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      @test A[j(jj),k(kk),i(ii)]==digits(T,ii,jj,kk)
    end
  end

  @testset "Test permute(ITensor,Index...)" begin
    A = randomITensor(T,i,k,j)
    permA = permute(A,k,j,i)

    @test k==inds(permA)[1]
    @test j==inds(permA)[2]
    @test i==inds(permA)[3]

    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      @test A[k(kk),i(ii),j(jj)]==permA[i(ii),j(jj),k(kk)]
    end
  end

  @testset "Set and get values with Ints" begin
    A = ITensor(T,i,j,k)
    A = permute(A,k,i,j)
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      A[kk,ii,jj] = digits(T,ii,jj,kk)
    end
    A = permute(A,i,j,k)
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      @test A[ii,jj,kk]==digits(T,ii,jj,kk)
    end
  end

  @testset "Test scalar(ITensor)" begin
    x = T(34)
    A = ITensor(x)
    @test x==scalar(A)
  end

  @testset "Test norm(ITensor)" begin
    A = randomITensor(T,i,j,k)
    @test norm(A)≈sqrt(scalar(dag(A)*A))
  end
  
  @testset "Test add ITensors" begin
    A = randomITensor(T,i,j,k)
    B = randomITensor(T,k,i,j)
    C = A+B
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      @test C[i(ii),j(jj),k(kk)]==A[j(jj),i(ii),k(kk)]+B[i(ii),k(kk),j(jj)]
    end
    @test Array(permute(C,i,j,k))==Array(permute(A,i,j,k))+Array(permute(B,i,j,k))
  end

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

    #TODO: need to add back this outer product test
    #@testset "Test contract ITensors (Vector*Vectorᵀ -> Matrix)" begin
    #  C = Ai*Aj
    #  CArray = Array(Ai)*transpose(Array(Aj))
    #  @test CArray≈Array(permute(C,i,j))
    #end

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

  @testset "Test svd of ITensor" begin
    A = randomITensor(T,i,j,k,l)
    U,S,V = svd(A,j,l)
    @test A≈U*S*V
  end

end

