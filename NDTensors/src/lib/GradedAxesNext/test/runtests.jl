@eval module $(gensym())
using BlockArrays: Block, blockedrange, blockfirsts, blocklasts, blocklengths
using NDTensors.LabelledNumbers: LabelledUnitRange, label, unlabel
using Test: @test, @test_broken, @testset
@testset "GradedAxes" begin
  a = blockedrange(["x" => 2, "y" => 3])
  @test a[Block(2)] == 3:5
  @test label(a[Block(2)]) == "y"
  @test a[Block(2)] isa LabelledUnitRange
  @test a[4] == 4
  @test label(a[4]) == "y"
  @test unlabel(a[4]) == 4
  @test blocklengths(a) == [2, 3]
  # TODO: Preserve labels.
  @test_broken label.(blocklengths(a)) == ["x", "y"]
  @test blockfirsts(a) == [1, 3]
  # TODO: Preserve labels.
  @test_broken label.(blockfirsts(a)) == ["x", "y"]
  @test first(a) == 1
  # TODO: Preserve labels.
  @test_broken label(first(a)) == "x"
  @test blocklasts(a) == [2, 5]
  @test label.(blocklasts(a)) == ["x", "y"]
  @test last(a) == 5
  @test label(last(a)) == "y"
  @test a[Block(2)] == 3:5
  @test label(a[Block(2)]) == "y"
  @test length(a[Block(2)]) == 3
  @test label(a, Block(2)) == "y"
  @show label(a, 4) == "y"
  # TODO: Define this.
  @test_broken blocklabels(a) == ["x", "y"]

  # Slicing operations
  # a[2:4]
  # a[Block(2)[2:3]]
  # a[[Block(2), Block(1)]]
  # a[Block(1):Block(2)]
end
end
