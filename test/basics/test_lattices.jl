using ITensors, Test

@test LatticeBond(1, 2) == LatticeBond(1, 2, 0.0, 0.0, 0.0, 0.0, "")
@testset "Square lattice" begin
  sL = square_lattice(3, 4)
  @test length(sL) == 17
end

@testset "Triangular lattice" begin
  tL = triangular_lattice(3, 4)
  @test length(tL) == 23
end

nothing
