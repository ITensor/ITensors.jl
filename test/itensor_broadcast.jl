using ITensors,
      Test

@testset "ITensor broadcast syntax" begin

  i = Index(2,"i")
  A = randomITensor(i,i')
  B = randomITensor(i',i)
  α = 2
  β = 3

  @testset "Copy" begin
    Bc = copy(B)
    Bc .= A
    @test Bc[1,1] == A[1,1]
    @test Bc[2,1] == A[1,2]
    @test Bc[1,2] == A[2,1]
    @test Bc[2,2] == A[2,2]
  end

  @testset "Fill" begin
    Bc = copy(B)
    Bc .= α
    @test Bc[1,1] == α
    @test Bc[2,1] == α
    @test Bc[1,2] == α
    @test Bc[2,2] == α
  end

  @testset "Scaling" begin
    Bc = copy(B)
    Bc .*= α
    @test Bc[1,1] == α * B[1,1]
    @test Bc[2,1] == α * B[2,1]
    @test Bc[1,2] == α * B[1,2]
    @test Bc[2,2] == α * B[2,2]
  end

  @testset "Scalar multiplication (in-place)" begin
    Bc = copy(B)
    Bc .= α .* A
    @test Bc[1,1] == α * A[1,1]
    @test Bc[2,1] == α * A[1,2]
    @test Bc[1,2] == α * A[2,1]
    @test Bc[2,2] == α * A[2,2]
  end

  @testset "Scalar multiplication (out-of-place)" begin
    Bc = α .* A
    @test Bc[1,1] == α * A[1,1]
    @test Bc[2,1] == α * A[2,1]
    @test Bc[1,2] == α * A[1,2]
    @test Bc[2,2] == α * A[2,2]
  end

  @testset "Addition" begin
    Bc = copy(B)
    Bc .= A .+ Bc
    @test Bc[1,1] == A[1,1] + B[1,1]
    @test Bc[2,1] == A[1,2] + B[2,1]
    @test Bc[1,2] == A[2,1] + B[1,2]
    @test Bc[2,2] == A[2,2] + B[2,2]
  end

  @testset "Addition (with α)" begin
    Bc = copy(B)
    Bc .+= A .* α

    @test Bc[1,1] == α * A[1,1] + B[1,1]
    @test Bc[2,1] == α * A[1,2] + B[2,1]
    @test Bc[1,2] == α * A[2,1] + B[1,2]
    @test Bc[2,2] == α * A[2,2] + B[2,2]
  end

  @testset "Addition (with α and β)" begin
    Bc = copy(B)
    Bc .= α .* A .+ β .* Bc

    @test Bc[1,1] == α * A[1,1] + β * B[1,1]
    @test Bc[2,1] == α * A[1,2] + β * B[2,1]
    @test Bc[1,2] == α * A[2,1] + β * B[1,2]
    @test Bc[2,2] == α * A[2,2] + β * B[2,2]
  end

  @testset "Addition errors" begin
    C = randomITensor(i,i')
    @test_throws ErrorException C .= A .+ B
    @test_throws ErrorException C = A .+ B
    @test_throws ErrorException C .= A .* B
  end

end

