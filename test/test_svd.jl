using ITensors,
      Test,
      Random

@testset "SVD Algorithms" begin

  @testset "Matrix With Zero Sing Val" begin
    M = [1.0 2.0 5.0 4.0;
         1.0 1.0 1.0 1.0;
         0.0 0.5 0.5 1.0;
         0.0 1.0 1.0 2.0]
    U,S,V = recursiveSVD(M)
    @test norm(U*Diagonal(S)*V'-M) < 1E-13
  end

  @testset "Real Matrix" begin
    M = rand(10,10)
    U,S,V = recursiveSVD(M)
    @test norm(U*Diagonal(S)*V'-M) < 1E-13
  end

  @testset "Cplx Matrix" begin
    M = rand(ComplexF64,10,10)
    U,S,V = recursiveSVD(M)
    @test norm(U*Diagonal(S)*V'-M) < 1E-13
  end

end
