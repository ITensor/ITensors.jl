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

  @testset "General functions" begin
    absA = abs.(A)

    @test absA[1,1] == abs(A[1,1])
    @test absA[2,1] == abs(A[2,1])

    Bc = copy(B)
    Bc .= sqrt.(absA)

    @test Bc[1,1] == sqrt(absA[1,1])
    @test Bc[2,1] == sqrt(absA[1,2])

    Bc2 = copy(B)
    Bc2 .+= sqrt.(absA)

    @test Bc2[1,1] == B[1,1]+sqrt(absA[1,1])
    @test Bc2[2,1] == B[2,1]+sqrt(absA[1,2])
  end

  @testset "Some other operations" begin
    i = Index(2)
    A = randomITensor(i)
    B = randomITensor(i)

    absA = abs.(A)

    @test absA[1] == abs(A[1])
    @test absA[2] == abs(A[2])

    Bc = copy(B)
    Bc .= sqrt.(absA)

    @test Bc[1] == sqrt(absA[1])
    @test Bc[2] == sqrt(absA[2])

    Bc2 = copy(B)
    Bc2 .+= sqrt.(absA)

    @test Bc2[1] == B[1]+sqrt(absA[1])
    @test Bc2[2] == B[2]+sqrt(absA[2])

    Bc3 = copy(B)
    Bc3 .= sqrt.(absA) .+ sin.(Bc3)

    @test Bc3[1] == sin(B[1])+sqrt(absA[1])
    @test Bc3[2] == sin(B[2])+sqrt(absA[2])

    sqrtabsA = sqrt.(abs.(A))

    @test sqrtabsA[1] == sqrt(abs(A[1]))
    @test sqrtabsA[2] == sqrt(abs(A[2]))

    sqrtabsA = cos.(sin.(sqrt.(abs.(A))))

    @test sqrtabsA[1] == cos(sin(sqrt(abs(A[1]))))
    @test sqrtabsA[2] == cos(sin(sqrt(abs(A[2]))))

    Ap = A .+ 3

    @test Ap[1] == A[1] + 3
    @test Ap[2] == A[2] + 3

    Apow1 = A .^ 2.0

    @test Apow1[1] == A[1]^2
    @test Apow1[2] == A[2]^2

    Apow2 = A .^ 3

    @test Apow2[1] == A[1]^3
    @test Apow2[2] == A[2]^3
  end

end

