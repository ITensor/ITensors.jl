using ITensors,
      Test
import ITensors.SmallString

@testset "QN" begin

  @testset "QNVal Basics" begin
    qv = QNVal()
    @test !isActive(qv)

    qv = QNVal("Sz",0)
    @test name(qv) == SmallString("Sz")
    @test val(qv) == 0
    @test modulus(qv) == 1
    @test isActive(qv)

    qv = QNVal("A",1,2)
    @test name(qv) == SmallString("A")
    @test val(qv) == 1
    @test modulus(qv) == 2
    @test !isFermionic(qv)

    qv = QNVal("Nf",1,-1)
    @test name(qv) == SmallString("Nf")
    @test val(qv) == 1
    @test modulus(qv) == -1
    @test isFermionic(qv)
  end

  @testset "QN Basics" begin
    q = QN()
    @test length(sprint(show,q)) > 1

    q = QN(("Sz",1))
    @test length(sprint(show,q)) > 1
    @test isActive(q[1])
    @test val(q,"Sz") == 1

    q = QN(("A",1),("B",2))
    @test isActive(q[1])
    @test isActive(q[2])
    @test val(q,"A") == 1
    @test val(q,"B") == 2
    @test modulus(q,"A") == 1
    @test modulus(q,"B") == 1

    q = QN(("B",2),("A",1))
    @test val(q,"A") == 1
    @test val(q,"B") == 2
  end

  @testset "Comparison" begin
    @test QN() == QN()
  end

end
