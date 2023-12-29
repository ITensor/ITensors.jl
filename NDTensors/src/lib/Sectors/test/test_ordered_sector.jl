import NDTensors.Sectors: ⊗, ⊕, ×, Fib, Ising, Sector, SU, SU2, U1, Z
using Test

@testset "Test Ordered Sectors" begin
  @testset "Fusion of U1 products" begin
    p11 = U1(1) × U1(1)
    @test p11 ⊗ p11 == [U1(2) × U1(2)]

    p123 = U1(1) × U1(2) × U1(3)
    @test p123 ⊗ p123 == [U1(2) × U1(4) × U1(6)]
  end

  @testset "Enforce same number of spaces" begin
    p12 = U1(1) × U1(2)
    p123 = U1(1) × U1(2) × U1(3)
    @test_throws DimensionMismatch p12 ⊗ p123
  end

  @testset "Fusion of SU2 products" begin
    phh = SU2(1//2) × SU2(1//2)
    @test phh ⊗ phh ==
      (SU2(0) × SU2(0)) ⊕ (SU2(1) × SU2(0)) ⊕ (SU2(0) × SU2(1)) ⊕ (SU2(1) × SU2(1))
  end

  @testset "Fusion of mixed U1 and SU2 products" begin
    p2h = U1(2) × SU2(1//2)
    p1h = U1(1) × SU2(1//2)
    @test p2h ⊗ p1h == (U1(3) × SU2(0)) ⊕ (U1(3) × SU2(1))

    p1h1 = U1(1) × SU2(1//2) × Z{2}(1)
    @test p1h1 ⊗ p1h1 == (U1(2) × SU2(0) × Z{2}(0)) ⊕ (U1(2) × SU2(1) × Z{2}(0))
  end
end

nothing
