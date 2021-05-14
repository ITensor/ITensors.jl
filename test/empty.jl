using ITensors, Test

@testset "emptyITensor (Empty)" begin
  @testset "emptyITensor set elements" begin
    i = Index(2; tags="i")

    E = emptyITensor(i', dag(i))

    @test hassameinds(E, (i', i))
    @test order(E) == 2
    @test E[i' => 1, i => 1] == 0

    E[i' => 1, i => 2] = 2.3

    @test E[i' => 1, i => 1] == 0
    @test E[i' => 2, i => 1] == 0
    @test E[i' => 1, i => 2] == 2.3
    @test E[i' => 2, i => 2] == 0
  end

  @testset "emptyITensor set elements (QN)" begin
    i = Index(QN(0) => 2, QN(1) => 2; tags="i")

    E = emptyITensor(i', dag(i))

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

  @testset "emptyITensor()" begin
    i = Index(QN(0) => 2, QN(1) => 2; tags="i")

    E = emptyITensor()

    @test isnothing(flux(E))
    @test order(E) == 0
    @test_throws MethodError E[i' => 1, i => 3] = 0

    A = randomITensor(i', dag(i))
    E += A

    @test norm(E - A) < 1E-8
  end
end

nothing
