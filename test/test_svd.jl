using ITensors,
      Test,
      Random,
      LinearAlgebra

@testset "SVD Algorithms" begin

  @testset "Test Orthog" begin
    M1 = [1.0 2.0 5.0 4.0;
         1.0 1.0 1.0 1.0;
         0.0 0.5 0.5 1.0;
         0.0 1.0 1.0 2.0]
    orthog!(M1)
    @test norm(M1'*M1-Diagonal(ones(size(M1,1)))) < 1E-12

    M2 = rand(10,10)
    orthog!(M2)
    @test norm(M2'*M2-Diagonal(ones(size(M2,1)))) < 1E-12
  end

  @testset "Matrix With Zero Sing Val" begin
    M = [1.0 2.0 5.0 4.0;
         1.0 1.0 1.0 1.0;
         0.0 0.5 0.5 1.0;
         0.0 1.0 1.0 2.0]
    U,S,V = svd_recursive(M)
    @test norm(U*Diagonal(S)*V'-M) < 1E-13
  end

  @testset "Real Matrix" begin
    M = rand(10,20)
    U,S,V = svd_recursive(M)
    @test norm(U*Diagonal(S)*V'-M) < 1E-13

    M = rand(20,10)
    U,S,V = svd_recursive(M)
    @test norm(U*Diagonal(S)*V'-M) < 1E-13
  end

  @testset "Cplx Matrix" begin
    M = rand(ComplexF64,10,15)
    U,S,V = svd_recursive(M)
    @test norm(U*Diagonal(S)*V'-M) < 1E-13

    M = rand(ComplexF64,15,10)
    U,S,V = svd_recursive(M)
    @test norm(U*Diagonal(S)*V'-M) < 1E-13
  end

end
