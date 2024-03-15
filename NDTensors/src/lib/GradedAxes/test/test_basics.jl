@eval module $(gensym())
using BlockArrays:
  Block, BlockVector, blockedrange, blockfirsts, blocklasts, blocklength, blocklengths
using NDTensors.GradedAxes: GradedUnitRange, blocklabels, gradedrange
using NDTensors.LabelledNumbers: LabelledUnitRange, label, labelled, unlabel
using Test: @test, @test_broken, @testset
@testset "GradedAxes basics" begin
  for a in (
    blockedrange([labelled(2, "x"), labelled(3, "y")]),
    gradedrange([labelled(2, "x"), labelled(3, "y")]),
    gradedrange(["x" => 2, "y" => 3]),
  )
    @test a isa GradedUnitRange
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
    @test blocklengths(only(axes(a))) == blocklengths(a)
    @test blocklabels(only(axes(a))) == blocklabels(a)
  end

  # Slicing operations
  x = gradedrange(["x" => 2, "y" => 3])
  a = x[2:4]
  @test a isa GradedUnitRange
  @test length(a) == 3
  @test blocklength(a) == 2
  @test a[Block(1)] == 2:2
  @test label(a[Block(1)]) == "x"
  @test a[Block(2)] == 3:4
  @test label(a[Block(2)]) == "y"
  @test isone(first(only(axes(a))))
  @test length(only(axes(a))) == length(a)
  @test blocklengths(only(axes(a))) == blocklengths(a)

  x = gradedrange(["x" => 2, "y" => 3])
  a = x[3:4]
  @test a isa GradedUnitRange
  @test length(a) == 2
  @test blocklength(a) == 1
  @test a[Block(1)] == 3:4
  @test label(a[Block(1)]) == "y"

  x = gradedrange(["x" => 2, "y" => 3])
  a = x[2:4][1:2]
  @test a isa GradedUnitRange
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

  x = gradedrange(["x" => 2, "y" => 3, "z" => 4])
  a = x[Block(2):Block(3)]
  @test a isa GradedUnitRange
  @test length(a) == 7
  @test blocklength(a) == 2
  @test blocklengths(a) == [3, 4]
  @test blocklabels(a) == ["y", "z"]
  @test a[Block(1)] == 3:5
  @test a[Block(2)] == 6:9

  x = gradedrange(["x" => 2, "y" => 3, "z" => 4])
  a = x[[Block(3), Block(2)]]
  @test a isa BlockVector
  @test length(a) == 7
  @test blocklength(a) == 2
  # TODO: `BlockArrays` doesn't define `blocklengths`
  # for `BlockVector`, should it?
  @test_broken blocklengths(a) == [4, 3]
  @test blocklabels(a) == ["z", "y"]
  @test a[Block(1)] == 6:9
  @test a[Block(2)] == 3:5

  x = gradedrange(["x" => 2, "y" => 3, "z" => 4])
  a = x[[Block(3)[2:3], Block(2)[2:3]]]
  @test a isa BlockVector
  @test length(a) == 4
  @test blocklength(a) == 2
  # TODO: `BlockArrays` doesn't define `blocklengths`
  # for `BlockVector`, should it?
  @test_broken blocklengths(a) == [2, 2]
  @test blocklabels(a) == ["z", "y"]
  @test a[Block(1)] == 7:8
  @test a[Block(2)] == 4:5
end
end
