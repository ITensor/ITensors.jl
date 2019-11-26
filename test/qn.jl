using ITensors,
      Test
import ITensors.SmallString

@testset "QN" begin

  @testset "QNVal Basics" begin
    qv = QNVal()
    @test !isactive(qv)

    qv = QNVal("Sz",0)
    @test name(qv) == SmallString("Sz")
    @test val(qv) == 0
    @test modulus(qv) == 1
    @test isactive(qv)

    qv = QNVal("A",1,2)
    @test name(qv) == SmallString("A")
    @test val(qv) == 1
    @test modulus(qv) == 2
    @test !isfermionic(qv)

    qv = QNVal("Nf",1,-1)
    @test name(qv) == SmallString("Nf")
    @test val(qv) == 1
    @test modulus(qv) == -1
    @test isfermionic(qv)
  end

  @testset "QN Basics" begin
    q = QN()
    @test length(sprint(show,q)) > 1

    q = QN(("Sz",1))
    @test length(sprint(show,q)) > 1
    @test isactive(q[1])
    @test val(q,"Sz") == 1

    q = QN("Sz",1)
    @test length(sprint(show,q)) > 1
    @test isactive(q[1])
    @test val(q,"Sz") == 1

    q = QN("P",1,2)
    @test length(sprint(show,q)) > 1
    @test isactive(q[1])
    @test val(q,"P") == 1
    @test modulus(q,"P") == 2

    q = QN(("A",1),("B",2))
    @test isactive(q[1])
    @test isactive(q[2])
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
    @test QN("A",1) == QN("A",1)
    @test QN(("A",1),("B",3)) == QN(("A",1),("B",3))
    @test QN(("A",1),("B",3)) == QN(("B",3),("A",1))

    # Zero value and missing sector treated the same:
    @test QN(("A",0),("B",3)) == QN("B",3)
    @test QN(("B",3),("A",0)) == QN("B",3)
  end

  @testset "Arithmetic" begin
    @test QN("Sz",1) + QN() == QN("Sz",1)
    @test QN("Sz",1) + QN("Sz",2) == QN("Sz",3)
    @test QN("Sz",1) + QN("Sz",-2) == QN("Sz",-1)

    @test QN(("A",1),("Sz",0)) + QN(("A",0),("Sz",1)) == QN(("A",1),("Sz",1))

    @test QN("P",0,2) + QN("P",1,2) == QN("P",1,2)
    @test QN("P",1,2) + QN("P",1,2) == QN("P",0,2)
  end

  @testset "Ordering" begin
    z = QN()
    qa = QN(("Sz",1),("Nf",1))
    qb = QN(("Sz",0),("Nf",2))
    qc = QN(("Sz",1),("Nf",2))
    qd = QN(("Sz",1),("Nf",2))
    qe = QN(("Sz",-1),("Nf",-2))

    @test !(z < z)
    @test !(qa < z)
    @test (z < qa)
    @test (z < qb)
    @test !(qb < z)
    @test (z < qc)
    @test !(qc < z)
    @test (z < qd)
    @test !(qd < z)
    @test !(z < qe)
    @test (qe < z)

    @test (qa > qb)
    @test !(qb > qa)
    @test !(qb == qa)
    @test (qb < qc)
    @test !(qc < qb)
    @test !(qc == qb)
    @test (qc == qd)
    @test !(qc < qd)
    @test !(qd < qc)
  end

end
