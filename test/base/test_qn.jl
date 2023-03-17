using ITensors, Test

import ITensors: nactive

@testset "QN" begin
  @testset "QNVal Basics" begin
    qv = ITensors.QNVal()
    @test !isactive(qv)

    qv = ITensors.QNVal("Sz", 0)
    @test ITensors.name(qv) == ITensors.SmallString("Sz")
    @test val(qv) == 0
    @test modulus(qv) == 1
    @test isactive(qv)

    qv = ITensors.QNVal("A", 1, 2)
    @test ITensors.name(qv) == ITensors.SmallString("A")
    @test val(qv) == 1
    @test modulus(qv) == 2
    @test !isfermionic(qv)

    qv = ITensors.QNVal("Nf", 1, -1)
    @test ITensors.name(qv) == ITensors.SmallString("Nf")
    @test val(qv) == 1
    @test modulus(qv) == -1
    @test isfermionic(qv)
  end

  @testset "QN Basics" begin
    q = QN()
    @test length(sprint(show, q)) > 1

    q = QN(("Sz", 1))
    @test length(sprint(show, q)) > 1
    @test isactive(q[1])
    @test val(q, "Sz") == 1
    @test !isfermionic(q)

    q = QN("Sz", 1)
    @test length(sprint(show, q)) > 1
    @test isactive(q[1])
    @test val(q, "Sz") == 1
    @test !isfermionic(q)

    q = QN("P", 1, 2)
    @test length(sprint(show, q)) > 1
    @test isactive(q[1])
    @test val(q, "P") == 1
    @test modulus(q, "P") == 2
    @test nactive(q) == 1

    q = QN(("A", 1), ("B", 2))
    @test isactive(q[1])
    @test isactive(q[2])
    @test val(q, "A") == 1
    @test val(q, "B") == 2
    @test modulus(q, "A") == 1
    @test modulus(q, "B") == 1

    q = QN(("B", 2), ("A", 1))
    @test val(q, "A") == 1
    @test val(q, "B") == 2
    @test nactive(q) == 2

    q = QN(("A", 1), ("B", 2), ("C", 3), ("D", 4))
    @test nactive(q) == 4

    @test_throws BoundsError begin
      q = QN(("A", 1), ("B", 2), ("C", 3), ("D", 4), ("E", 5))
    end
  end

  @testset "Comparison" begin
    @test QN() == QN()
    @test QN("A", 1) == QN("A", 1)
    @test QN(("A", 1), ("B", 3)) == QN(("A", 1), ("B", 3))
    @test QN(("A", 1), ("B", 3)) == QN(("B", 3), ("A", 1))

    # Zero value and missing sector treated the same:
    @test QN(("A", 0), ("B", 3)) == QN("B", 3)
    @test QN(("B", 3), ("A", 0)) == QN("B", 3)
  end

  @testset "Arithmetic" begin
    @test QN("Sz", 1) + QN() == QN("Sz", 1)
    @test QN("Sz", 1) + QN("Sz", 2) == QN("Sz", 3)
    @test QN("Sz", 1) + QN("Sz", -2) == QN("Sz", -1)

    @test QN(("A", 1), ("Sz", 0)) + QN(("A", 0), ("Sz", 1)) == QN(("A", 1), ("Sz", 1))

    @test QN("P", 0, 2) + QN("P", 1, 2) == QN("P", 1, 2)
    @test QN("P", 1, 2) + QN("P", 1, 2) == QN("P", 0, 2)

    # Arithmetic involving mixed-label QNs
    @test QN() - QN("Sz", 2) == QN("Sz", -2)
    @test QN("Sz", 2) - QN() == QN("Sz", 2)
    @test QN() - QN(("Sz", 2), ("N", 1)) == QN(("Sz", -2), ("N", -1))
    @test QN("N", 1) - QN("Sz", 2) == QN(("N", 1), ("Sz", -2))
  end

  @testset "Ordering" begin
    z = QN()
    qa = QN(("Sz", 1), ("Nf", 1))
    qb = QN(("Sz", 0), ("Nf", 2))
    qc = QN(("Sz", 1), ("Nf", 2))
    qd = QN(("Sz", 1), ("Nf", 2))
    qe = QN(("Sz", -1), ("Nf", -2))

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

    @test !(qa > qb)
    @test qb > qa
    @test !(qb == qa)
    @test (qb < qc)
    @test !(qc < qb)
    @test !(qc == qb)
    @test (qc == qd)
    @test !(qc < qd)
    @test !(qd < qc)
  end

  @testset "Hashing" begin
    @test hash(QN(("Sz", 0))) == hash(QN())
    @test hash(QN("Sz", 0)) == hash(QN("N", 0))
    @test hash(QN(("Sz", 1), ("N", 2))) == hash(QN(("N", 2), ("Sz", 1)))
  end

  @testset "Negative value for mod > 1" begin
    @test QN("T", -1, 3) == QN("T", 2, 3)
    @test QN("T", -2, 3) == QN("T", 1, 3)
    @test QN("T", -3, 3) == QN("T", 0, 3)
  end
end

nothing
