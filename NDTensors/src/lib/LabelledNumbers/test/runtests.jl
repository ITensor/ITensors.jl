@eval module $(gensym())
using NDTensors.LabelledNumbers: label, labelled, unlabel
using Test: @test, @testset
@testset "LabelledNumbers" begin
  @testset "Labelled number ($n)" for n in (2, 2.0)
    x = labelled(2, "x")
    @test x == 2
    @test label(x) == "x"
    @test unlabel(x) == 2
    @test unlabel(x) == 2

    @test x * 2 == 4
    @test 2 * x == 4
    @test label(x * 2) == "x"
    @test label(2 * x) == "x"
    @test x / 2 == 1
    @test label(x / 2) == "x"
    @test x ÷ 2 == 1
    @test label(x ÷ 2) == "x"
    @test -x == -2
  end
  @testset "Labelled array ($a)" for a in (collect(2:5), 2:5)
    x = labelled(a, "x")
    @test x == 2:5
    @test label(x) == "x"
    @test unlabel(x) == 2:5

    @test x[2] == 3
    @test label(x[2]) == "x"
    @test x[2:3] == 3:4
    @test label(x[2:3]) == "x"
    @test x[[2, 4]] == [3, 5]
    @test label(x[[2, 4]]) == "x"
  end
end
end
