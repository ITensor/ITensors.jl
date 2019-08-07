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
    U,S,V = recursiveSVD(M)
    @test norm(U*Diagonal(S)*V'-M) < 1E-13
  end

  @testset "Real Matrix" begin
    M = rand(10,20)
    U,S,V = recursiveSVD(M)
    @test norm(U*Diagonal(S)*V'-M) < 1E-13

    M = rand(20,10)
    U,S,V = recursiveSVD(M)
    @test norm(U*Diagonal(S)*V'-M) < 1E-13
  end

  @testset "Cplx Matrix" begin
    M = rand(ComplexF64,10,15)
    U,S,V = recursiveSVD(M)
    @test norm(U*Diagonal(S)*V'-M) < 1E-13

    M = rand(ComplexF64,15,10)
    U,S,V = recursiveSVD(M)
    @test norm(U*Diagonal(S)*V'-M) < 1E-13
  end
  @testset "set random number generator" begin
    i = Index(4)
    j = Index(4)
    A = randomITensor(i,j)
    u,s,v = svd(A,i)
    s[3,3] = 0
    s[4,4] = 0
    A = u*s*v
    rng1 = MersenneTwister(1)
    rng2 = MersenneTwister(1)
    u1,s1,v1 = svd(A,i; rng = rng1)
    u2,s2,v2 = svd(A,i; rng = rng2)
    @test u1.store.data == u2.store.data
    @test s1.store.data == s2.store.data
    @test v1.store.data == v2.store.data
  end
end
