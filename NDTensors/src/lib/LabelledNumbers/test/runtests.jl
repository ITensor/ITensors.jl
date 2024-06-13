@eval module $(gensym())
using LinearAlgebra: norm
using NDTensors.LabelledNumbers:
  LabelledInteger, LabelledUnitRange, islabelled, label, labelled, unlabel
using Test: @test, @testset
@testset "LabelledNumbers" begin
  @testset "Labelled number ($n)" for n in (2, 2.0)
    x = labelled(2, "x")
    @test typeof(x) == LabelledInteger{Int,String}
    @test islabelled(x)
    @test x == 2
    @test label(x) == "x"
    @test unlabel(x) == 2
    @test !islabelled(unlabel(x))

    @test labelled(1, "x") < labelled(2, "x")
    @test !(labelled(2, "x") < labelled(2, "x"))
    @test !(labelled(3, "x") < labelled(2, "x"))

    @test !(labelled(1, "x") > labelled(2, "x"))
    @test !(labelled(2, "x") > labelled(2, "x"))
    @test labelled(3, "x") > labelled(2, "x")

    @test labelled(1, "x") <= labelled(2, "x")
    @test labelled(2, "x") <= labelled(2, "x")
    @test !(labelled(3, "x") <= labelled(2, "x"))

    @test !(labelled(1, "x") >= labelled(2, "x"))
    @test labelled(2, "x") >= labelled(2, "x")
    @test labelled(3, "x") >= labelled(2, "x")

    @test x * 2 == 4
    @test !islabelled(x * 2)
    @test 2 * x == 4
    @test !islabelled(2 * x)
    @test x * x == 4
    @test !islabelled(x * x)

    @test x + 3 == 5
    @test !islabelled(x + 3)
    @test 3 + x == 5
    @test !islabelled(3 + x)
    @test x + x == 4
    @test !islabelled(x + x)

    @test x - 3 == -1
    @test !islabelled(x - 3)
    @test 3 - x == 1
    @test !islabelled(3 - x)
    @test x - x == 0
    @test !islabelled(x - x)

    @test x / 2 == 1
    @test x / 2 isa AbstractFloat
    @test x / 2 isa Float64
    @test !islabelled(x / 2)
    @test x ÷ 2 == 1
    @test x ÷ 2 isa Integer
    @test x ÷ 2 isa Int
    @test !islabelled(x ÷ 2)
    @test -x == -2
    @test hash(x) == hash(2)
    @test zero(x) == false
    @test label(zero(x)) == "x"
    @test one(x) == true
    @test !islabelled(one(x))
    @test oneunit(x) == true
    @test label(oneunit(x)) == "x"
    @test islabelled(oneunit(x))
    @test one(typeof(x)) == true
    @test !islabelled(one(typeof(x)))
  end
  @testset "randn" begin
    d = labelled(2, "x")

    a = randn(Float32, d, d)
    @test eltype(a) === Float32
    @test size(a) == (2, 2)
    @test norm(a) > 0

    a = rand(Float32, d, d)
    @test eltype(a) === Float32
    @test size(a) == (2, 2)
    @test norm(a) > 0

    a = randn(d, d)
    @test eltype(a) === Float64
    @test size(a) == (2, 2)
    @test norm(a) > 0

    a = rand(d, d)
    @test eltype(a) === Float64
    @test size(a) == (2, 2)
    @test norm(a) > 0
  end
  @testset "Labelled array ($a)" for a in (collect(2:5), 2:5)
    x = labelled(a, "x")
    @test eltype(x) == LabelledInteger{Int,String}
    @test x == 2:5
    @test label(x) == "x"
    @test unlabel(x) == 2:5
    @test first(iterate(x, 3)) == 4
    @test label(first(iterate(x, 3))) == "x"
    @test collect(x) == 2:5
    @test label.(collect(x)) == fill("x", 4)
    @test x[2] == 3
    @test label(x[2]) == "x"
    @test x[2:3] == 3:4
    @test label(x[2:3]) == "x"
    @test x[[2, 4]] == [3, 5]
    @test label(x[[2, 4]]) == "x"

    if x isa AbstractUnitRange
      @test step(x) == true
      @test islabelled(step(x))
      @test label(step(x)) == "x"
    end
  end
end

using BlockArrays: Block, blockaxes, blocklength, blocklengths
@testset "LabelledNumbersBlockArraysExt" begin
  x = labelled(1:2, "x")
  @test blockaxes(x) == (Block.(1:1),)
  @test blocklength(x) == 1
  @test blocklengths(x) == [2]
  a = x[Block(1)]
  @test a == 1:2
  @test a isa LabelledUnitRange
  @test label(a) == "x"
end
end
