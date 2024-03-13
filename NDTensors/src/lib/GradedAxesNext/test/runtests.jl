@eval module $(gensym())
using BlockArrays: Block, blockedrange, blockfirsts, blocklasts, blocklength, blocklengths
using NDTensors.GradedAxesNext: blocklabels, gradedrange
using NDTensors.LabelledNumbers: LabelledUnitRange, label, unlabel
using Test: @test, @test_broken, @testset
@testset "GradedAxes" begin
  a = gradedrange(["x" => 2, "y" => 3])
  @test length(a) == 5
  @test a[Block(2)] == 3:5
  @test label(a[Block(2)]) == "y"
  @test a[Block(2)] isa LabelledUnitRange
  @test a[4] == 4
  @test label(a[4]) == "y"
  @test unlabel(a[4]) == 4
  @test blocklengths(a) == [2, 3]
  @test blocklabels(a) == ["x", "y"]
  @test label.(blocklengths(a)) == ["x", "y"]
  @test blockfirsts(a) == [1, 3]
  @test label.(blockfirsts(a)) == ["x", "y"]
  @test first(a) == 1
  @test label(first(a)) == "x"
  @test blocklasts(a) == [2, 5]
  @test label.(blocklasts(a)) == ["x", "y"]
  @test last(a) == 5
  @test label(last(a)) == "y"
  @test a[Block(2)] == 3:5
  @test label(a[Block(2)]) == "y"
  @test length(a[Block(2)]) == 3
  @test label(a, Block(2)) == "y"
  @test label(a, 4) == "y"

  @test blocklengths(axes(a, 1)) == blocklengths(a)
  @test blocklabels(axes(a, 1)) == blocklabels(a)

  # Slicing operations
  x = gradedrange(["x" => 2, "y" => 3])
  a = x[2:4]
  @test length(a) == 3
  @test blocklength(a) == 2
  @test a[Block(1)] == 2:2
  @test label(a[Block(1)]) == "x"
  @test a[Block(2)] == 3:4
  @test label(a[Block(2)]) == "y"

  x = gradedrange(["x" => 2, "y" => 3])
  a = x[3:4]
  @test length(a) == 2
  @test blocklength(a) == 1
  @test a[Block(1)] == 3:4
  @test label(a[Block(1)]) == "y"

  x = gradedrange(["x" => 2, "y" => 3])
  a = x[2:4][1:2]
  @test length(a) == 2
  @test blocklength(a) == 2
  @test a[Block(1)] == 2:2
  @test label(a[Block(1)]) == "x"
  @test a[Block(2)] == 3:3
  @test label(a[Block(2)]) == "y"

  x = gradedrange(["x" => 2, "y" => 3])
  a = x[Block(2)[2:3]]
  @test a isa LabelledUnitRange
  @test length(a) == 2
  @test a == 4:5
  @test label(a) == "y"

  # a[Block(1):Block(2)]
  # a[[Block(2), Block(1)]]
end
end
