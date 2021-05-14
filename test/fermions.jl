using ITensors, Test

@testset "Fermions" begin
  @testset "parity_sign function" begin

    # Full permutations
    p1 = [1, 2, 3]
    @test ITensors.parity_sign(p1) == +1
    p2 = [2, 1, 3]
    @test ITensors.parity_sign(p2) == -1
    p3 = [2, 3, 1]
    @test ITensors.parity_sign(p3) == +1
    p4 = [3, 2, 1]
    @test ITensors.parity_sign(p4) == -1

    ## Partial permutations
    p5 = [2, 7]
    @test ITensors.parity_sign(p5) == +1
    p6 = [5, 3]
    @test ITensors.parity_sign(p6) == -1
    p7 = [1, 9, 3, 10]
    @test ITensors.parity_sign(p7) == -1
    p8 = [1, 12, 9, 3, 11]
    @test ITensors.parity_sign(p8) == +1
  end
end
