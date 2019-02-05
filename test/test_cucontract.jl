using ITensors,
      ITensors.CuITensors,
      LinearAlgebra, # For tr()
      Combinatorics, # For permutations()
      CuArrays,
      Test

@testset "cuITensor $T Contractions" for T ∈ (Float64,ComplexF64)
  mi,mj,mk,ml,ma = 2,3,4,5,6,7
  i = Index(mi,"i")
  j = Index(mj,"j")
  k = Index(mk,"k")
  l = Index(ml,"l")
  a = Index(ma,"a") 
  @testset "Test contract ITensors" begin
      A = cuITensor(randomITensor(T))
      B = cuITensor(randomITensor(T))
      Ai = cuITensor(randomITensor(T,i))
      Bi = cuITensor(randomITensor(T,i))
      Aj = cuITensor(randomITensor(T,j))
      Aij = cuITensor(randomITensor(T,i,j))
      Bij = cuITensor(randomITensor(T,i,j))
      Aik = cuITensor(randomITensor(T,i,k))
      Ajk = cuITensor(randomITensor(T,j,k))
      Ajl = cuITensor(randomITensor(T,j,l))
      Akl = cuITensor(randomITensor(T,k,l))
      Aijk = cuITensor(randomITensor(T,i,j,k))
      Ajkl = cuITensor(randomITensor(T,j,k,l))
      Aikl = cuITensor(randomITensor(T,i,k,l))
      Akla = cuITensor(randomITensor(T,k,l,a))
      Aijkl = cuITensor(randomITensor(T,i,j,k,l))
    @testset "Test contract cuITensor (Scalar*Scalar -> Scalar)" begin
      C = A*B
      @test scalar(C)≈scalar(A)*scalar(B)
    end
    @testset "Test contract cuITensor (Scalar*Vector -> Vector)" begin
      C = A*Ai
      @test collect(C)≈scalar(A)*collect(Ai)
    end
    @testset "Test contract cuITensor (Vector*Scalar -> Vector)" begin
      C = Aj*A
      @test collect(C)≈scalar(A)*collect(Aj)
    end
    #=@testset "Test contract ITensors (Vectorᵀ*Vector -> Scalar)" begin
      C = Ai*Bi
      Ccollect = transpose(Array(collect(Ai)))*Array(collect(Bi))
      @test Ccollect≈scalar(C)
    end=#
    #TODO: need to add back this outer product test
    #@testset "Test contract ITensors (Vector*Vectorᵀ -> Matrix)" begin
    #  C = Ai*Aj
    #  Ccollect = collect(Ai)*transpose(collect(Aj))
    #  @test Ccollect≈collect(permute(C,i,j))
    #end
    @testset "Test contract ITensors (Matrix*Scalar -> Matrix)" begin
      Aij = permute(Aij,i,j)
      C = Aij*A
      @test collect(permute(C,i,j))≈scalar(A)*collect(Aij)
    end
    @testset "Test contract ITensors (Matrix*Vector -> Vector)" begin
      Aij = permute(Aij,i,j)
      C = Aij*Aj
      Ccollect = collect(permute(Aij,i,j))*collect(Aj)
      @test Ccollect≈collect(C)
    end
    @testset "Test contract ITensors (Matrixᵀ*Vector -> Vector)" begin
      Aij = permute(Aij,j,i)
      C = Aij*Aj
      Ccollect = transpose(Array(collect(Aij)))*Array(collect(Aj))
      @test Ccollect≈Array(collect(C))
    end
    @testset "Test contract ITensors (Vector*Matrix -> Vector)" begin
      Aij = permute(Aij,i,j)
      C = Ai*Aij
      Ccollect = transpose(transpose(Array(collect(Ai)))*Array(collect(Aij)))
      @test Ccollect≈Array(collect(C))
    end
    @testset "Test contract ITensors (Vector*Matrixᵀ -> Vector)" begin
      Aij = permute(Aij,j,i)
      C = Ai*Aij
      Ccollect = transpose(transpose(Array(collect(Ai)))*transpose(Array(collect(Aij))))
      @test Ccollect≈Array(collect(C))
    end
    #=@testset "Test contract ITensors (Matrix*Matrix -> Scalar)" begin
      Aij = permute(Aij,i,j)
      Bij = permute(Bij,i,j)
      C = Aij*Bij
      Ccollect = tr(Array(collect(Aij))*transpose(Array(collect(Bij))))
      @test Ccollect≈scalar(C)
    end=#
    @testset "Test contract ITensors (Matrix*Matrix -> Matrix)" begin
      Aij = permute(Aij,i,j)
      Ajk = permute(Ajk,j,k)
      C = Aij*Ajk
      Ccollect = collect(Aij)*collect(Ajk)
      @test Ccollect≈collect(C)
    end
    @testset "Test contract ITensors (Matrixᵀ*Matrix -> Matrix)" begin
      Aij = permute(Aij,j,i)
      Ajk = permute(Ajk,j,k)
      C = Aij*Ajk
      Ccollect = transpose(Array(collect(Aij)))*Array(collect(Ajk))
      @test Ccollect≈Array(collect(C))
    end
    @testset "Test contract ITensors (Matrix*Matrixᵀ -> Matrix)" begin
      Aij = permute(Aij,i,j)
      Ajk = permute(Ajk,k,j)
      C = Aij*Ajk
      Ccollect = Array(collect(Aij))*transpose(Array(collect(Ajk)))
      @test Ccollect≈Array(collect(C))
    end
    @testset "Test contract ITensors (Matrixᵀ*Matrixᵀ -> Matrix)" begin
      Aij = permute(Aij,j,i)
      Ajk = permute(Ajk,k,j)
      C = Aij*Ajk
      Ccollect = transpose(Array(collect(Aij)))*transpose(Array(collect(Ajk)))
      @test Ccollect≈Array(collect(C))
    end
    @testset "Test contract ITensors (3-Tensor*Scalar -> 3-Tensor)" begin
      Aijk = permute(Aijk,i,j,k)
      C = Aijk*A
      @test collect(permute(C,i,j,k))≈scalar(A)*collect(Aijk)
    end
    @testset "Test contract ITensors (3-Tensor*Vector -> Matrix)" begin
      Aijk = permute(Aijk,i,j,k)
      C = Aijk*Ai
      Ccollect = reshape(reshape(Array(collect(permute(Aijk,j,k,i))),Dims((dim(j)*dim(k),dim(i))))*Array(collect(Ai)),dim(j),dim(k))
      @test Ccollect≈Array(collect(permute(C,j,k)))
    end
    @testset "Test contract ITensors (Vector*3-Tensor -> Matrix)" begin
      Aijk = permute(Aijk,i,j,k)
      C = Aj*Aijk
      Ccollect = reshape(transpose(Array(collect(Aj)))*reshape(Array(collect(permute(Aijk,j,i,k))),Dims((dim(j),dim(i)*dim(k)))),Dims((dim(i),dim(k))))
      @test Array(Ccollect)≈Array(collect(permute(C,i,k)))
    end
    @testset "Test contract ITensors (3-Tensor*Matrix -> Vector)" begin
      Aijk = permute(Aijk,i,j,k)
      Aik = permute(Aik,i,k)
      C = Aijk*Aik
      Ccollect = reshape(Array(collect(permute(Aijk,j,i,k))),dim(j),dim(i)*dim(k))*vec(Array(collect(Aik)))
      @test Ccollect≈Array(collect(C))
    end
    @testset "Test contract ITensors (3-Tensor*Matrix -> 3-Tensor)" begin
      Aijk = permute(Aijk,i,j,k)
      Ajl = permute(Ajl,j,l)
      C = Aijk*Ajl
      Ccollect = reshape(reshape(Array(collect(permute(Aijk,i,k,j))),Dims((dim(i)*dim(k),dim(j))))*Array(collect(Ajl)),dim(i),dim(k),dim(l))
      @test Ccollect≈Array(collect(permute(C,i,k,l)))
    end
    @testset "Test contract ITensors (Matrix*3-Tensor -> 3-Tensor)" begin
      Aijk = permute(Aijk,i,j,k)
      Akl = permute(Akl,k,l)
      C = Akl*Aijk
      Ccollect = reshape(Array(collect(permute(Akl,l,k)))*reshape(Array(collect(permute(Aijk,k,i,j))),dim(k),dim(i)*dim(j)),dim(l),dim(i),dim(j))
      @test Ccollect≈Array(collect(permute(C,l,i,j)))
    end
    @testset "Test contract ITensors (3-Tensor*3-Tensor -> 3-Tensor)" begin
      Aijk = permute(Aijk,i,j,k)
      Ajkl = permute(Ajkl,j,k,l)
      C = Aijk*Ajkl
      Ccollect = reshape(Array(collect(permute(Aijk,i,j,k))),dim(i),dim(j)*dim(k))*reshape(Array(collect(permute(Ajkl,j,k,l))),dim(j)*dim(k),dim(l))
      @test Ccollect≈Array(collect(permute(C,i,l)))
    end
    @testset "Test contract ITensors (3-Tensor*3-Tensor -> 3-Tensor)" begin
      for inds_ijk ∈ permutations([i,j,k]), inds_jkl ∈ permutations([j,k,l])
        Aijk = permute(Aijk,inds_ijk...)
        Ajkl = permute(Ajkl,inds_jkl...)
        C = Ajkl*Aijk
        Ccollect = reshape(Array(collect(permute(Ajkl,l,j,k))),dim(l),dim(j)*dim(k))*reshape(Array(collect(permute(Aijk,j,k,i))),dim(j)*dim(k),dim(i))
        @test Ccollect≈Array(collect(permute(C,l,i)))
      end
    end
    @testset "Test contract ITensors (4-Tensor*3-Tensor -> 1-Tensor)" begin
      for inds_ijkl ∈ permutations([i,j,k,l]), inds_jkl ∈ permutations([j,k,l])
        Aijkl = permute(Aijkl,inds_ijkl...)
        Ajkl = permute(Ajkl,inds_jkl...) 
        C = Ajkl*Aijkl
        Ccollect = reshape(Array(collect(permute(Ajkl,j,k,l))),1,dim(j)*dim(k)*dim(l))*reshape(Array(collect(permute(Aijkl,j,k,l,i))),dim(j)*dim(k)*dim(l),dim(i))
        @test vec(Ccollect)≈Array(collect(permute(C,i)))
      end
    end
    @testset "Test contract ITensors (4-Tensor*3-Tensor -> 3-Tensor)" begin
      for inds_ijkl ∈ permutations([i,j,k,l]), inds_kla ∈ permutations([k,l,a])
        Aijkl = permute(Aijkl,inds_ijkl...)
        Akla = permute(Akla,inds_kla...)
        C = Akla*Aijkl
        Ccollect = reshape(reshape(Array(collect(permute(Akla,a,k,l))),dim(a),dim(k)*dim(l))*reshape(Array(collect(permute(Aijkl,k,l,i,j))),dim(k)*dim(l),dim(i)*dim(j)),dim(a),dim(i),dim(j))
        @test Ccollect≈Array(collect(permute(C,a,i,j)))
      end
    end
  end # End contraction testset
end
