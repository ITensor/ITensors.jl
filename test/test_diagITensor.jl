using ITensors,
      LinearAlgebra, # For tr()
      Random,        # To set a seed
      Test

Random.seed!(12345)

@testset "diagITensor" begin
  i = Index(2,"i")
  j = Index(2,"j")
  k = Index(2,"k")

  @testset "Zero constructor" begin
    d = diagITensor(i,j)

    @test eltype(d) == Float64
    @test d[i(1),j(1)] == 0.0
    @test d[i(2),j(2)] == 0.0
    @test d[i(1),j(2)] == 0.0
    @test d[i(2),j(1)] == 0.0
  end

  @testset "Zero constructor" begin
    d = diagITensor(ComplexF64,i,j)

    @test eltype(d) == ComplexF64
    @test d[i(1),j(1)] == complex(0.0)
    @test d[i(2),j(2)] == complex(0.0)
    @test d[i(1),j(2)] == complex(0.0)
    @test d[i(2),j(1)] == complex(0.0)
  end

  @testset "Diagonal constructor" begin
    d = diagITensor([1,2],i,j)

    @test eltype(d) == Float64
    @test d[i(1),j(1)] == 1.0
    @test d[i(2),j(2)] == 2.0
    @test d[i(1),j(2)] == 0.0
    @test d[i(2),j(1)] == 0.0
  end

  @testset "Diagonal constructor" begin
    d = diagITensor([1+1im,2+2im],i,j)

    @test eltype(d) == ComplexF64
    @test d[i(1),j(1)] == 1.0+1.0im
    @test d[i(2),j(2)] == 2.0+2.0im
    @test d[i(1),j(2)] == complex(0.0)
    @test d[i(2),j(1)] == complex(0.0)
  end

  @testset "Set elements" begin
    d = diagITensor(i,j)

    d[i(1),j(1)] = 1.0
    d[i(2),j(2)] = 2.0

    @test d[i(1),j(1)] == 1.0
    @test d[i(2),j(2)] == 2.0
    @test d[i(1),j(2)] == 0.0
    @test d[i(2),j(1)] == 0.0

    @test_throws ErrorException d[i(2),j(1)] = 0.0
    @test_throws ErrorException d[i(1),j(2)] = 0.0
  end

  @testset "Convert to dense" begin
    d = diagITensor([1,2],i,j,k)
    t = dense(d)
    
    @test store(t) isa Dense{Float64}
    @test t[i(1),j(1),k(1)] == 1.0
    @test t[i(2),j(2),k(2)] == 2.0
    @test t[i(1),j(2),k(1)] == 0.0
    @test t[i(2),j(1),k(1)] == 0.0
  end

  @testset "Contraction (all contracted)" begin
    D = diagITensor([1,2],i,j,k)
    A = randomITensor(i,j,k)
    
    @test D*A == dense(D)*A
  end
  @testset "Contraction (all dense contracted)" begin
    D = diagITensor([1,2],i,j,k)
    A = randomITensor(i,j)
    
    @test D*A == dense(D)*A
  end


end

