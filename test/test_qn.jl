using ITensors,
      Test
import ITensors.SmallString

@testset "QN" begin
  @testset "QNVal Basics" begin
    qv = QNVal()
    @test !isActive(qv)

    qv = QNVal(("Sz",0))
    @test name(qv) == SmallString("Sz")
    @test val(qv) == 0
    @test modulus(qv) == 1
    @test isActive(qv)

    qv = QNVal(("A",1,2))
    @test name(qv) == SmallString("A")
    @test val(qv) == 1
    @test modulus(qv) == 2
    @test !isFermionic(qv)

    qv = QNVal(("Nf",1,-1))
    @test name(qv) == SmallString("Nf")
    @test val(qv) == 1
    @test modulus(qv) == -1
    @test isFermionic(qv)
  end
end
