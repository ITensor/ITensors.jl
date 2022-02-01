using ITensors
using ITensors.NDTensors
using Test

@testset "ITensor (Empty)" begin
  @testset "ITensor set elements" begin
    i = Index(2; tags="i")

    E = ITensor(i', dag(i))

    @test conj(E) == E
    @test 1.2 * E == E

    @test hassameinds(E, (i', i))
    @test order(E) == 2
    @test E[i' => 1, i => 1] == 0

    E[i' => 1, i => 2] = 2.3

    @test E[i' => 1, i => 1] == 0
    @test E[i' => 2, i => 1] == 0
    @test E[i' => 1, i => 2] == 2.3
    @test E[i' => 2, i => 2] == 0
  end

  @testset "ITensor (Empty) convert to complex" begin
    i = Index(2; tags="i")
    E = ITensor(i', dag(i))
    @test eltype(E) == NDTensors.EmptyNumber

    Ec = complex(E)
    @test eltype(Ec) == Complex{NDTensors.EmptyNumber}
    Ec[1, 1] = 2.3
    @test eltype(Ec) == ComplexF64

    Ec = complex(E)
    @test eltype(Ec) == Complex{NDTensors.EmptyNumber}
    Ec[1, 1] = 2.3f0
    @test eltype(Ec) == ComplexF32

    E2 = copy(E)
    E2c = complex!(E2)
    @test eltype(E2c) == Complex{NDTensors.EmptyNumber}
  end

  @testset "ITensor set elements (QN)" begin
    i = Index(QN(0) => 2, QN(1) => 2; tags="i")

    E = ITensor(i', dag(i))

    @test hassameinds(E, (i', i))
    @test order(E) == 2
    @test isnothing(flux(E))
    @test E[i' => 1, i => 3] == 0

    E[i' => 3, i => 2] = 2.3

    @test flux(E) == QN(1)

    @test E[i' => 1, i => 1] == 0
    @test E[i' => 2, i => 1] == 0
    @test E[i' => 3, i => 2] == 2.3
    @test E[i' => 2, i => 3] == 0
    @test_throws ErrorException E[i' => 2, i => 3] = 3.2
  end

  @testset "ITensor()" begin
    i = Index(QN(0) => 2, QN(1) => 2; tags="i")

    E = ITensor()

    @test isnothing(flux(E))
    @test order(E) == 0
    @test_throws MethodError E[i' => 1, i => 3] = 0

    A = randomITensor(i', dag(i))
    E += A

    @test norm(E - A) < 1E-8
  end
end

nothing
