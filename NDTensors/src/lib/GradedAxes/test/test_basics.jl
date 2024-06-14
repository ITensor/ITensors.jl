@eval module $(gensym())
using BlockArrays:
  Block,
  BlockSlice,
  BlockVector,
  blockedrange,
  blockfirsts,
  blocklasts,
  blocklength,
  blocklengths,
  blocks
using NDTensors.BlockSparseArrays: BlockSparseVector
using NDTensors.GradedAxes: GradedOneTo, GradedUnitRange, blocklabels, gradedrange
using NDTensors.LabelledNumbers: LabelledUnitRange, islabelled, label, labelled, unlabel
using Test: @test, @test_broken, @testset
@testset "GradedAxes basics" begin
  for a in (
    blockedrange([labelled(2, "x"), labelled(3, "y")]),
    gradedrange([labelled(2, "x"), labelled(3, "y")]),
    gradedrange(["x" => 2, "y" => 3]),
  )
    @test a isa GradedOneTo
    for x in iterate(a)
      @test x == 1
      @test label(x) == "x"
    end
    for x in iterate(a, labelled(1, "x"))
      @test x == 2
      @test label(x) == "x"
    end
    for x in iterate(a, labelled(2, "x"))
      @test x == 3
      @test label(x) == "y"
    end
    for x in iterate(a, labelled(3, "y"))
      @test x == 4
      @test label(x) == "y"
    end
    for x in iterate(a, labelled(4, "y"))
      @test x == 5
      @test label(x) == "y"
    end
    @test isnothing(iterate(a, labelled(5, "y")))
    @test length(a) == 5
    @test step(a) == 1
    @test !islabelled(step(a))
    @test length(blocks(a)) == 2
    @test blocks(a)[1] == 1:2
    @test label(blocks(a)[1]) == "x"
    @test blocks(a)[2] == 3:5
    @test label(blocks(a)[2]) == "y"
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
  ax = only(axes(a))
  @test ax == 1:length(a)
  @test length(ax) == length(a)
  @test blocklengths(ax) == blocklengths(a)
  @test blocklabels(ax) == blocklabels(a)

  # Regression test for ambiguity error.
  x = gradedrange(["x" => 2, "y" => 3])
  a = x[BlockSlice(Block(1), Base.OneTo(2))]
  @test length(a) == 2
  @test a == 1:2
  @test blocklength(a) == 1
  # TODO: Should this be a `GradedUnitRange`,
  # or maybe just a `LabelledUnitRange`?
  @test a isa LabelledUnitRange
  @test length(a[Block(1)]) == 2
  @test label(a) == "x"
  @test a[Block(1)] == 1:2
  @test label(a[Block(1)]) == "x"

  x = gradedrange(["x" => 2, "y" => 3])
  a = x[3:4]
  @test a isa GradedUnitRange
  @test length(a) == 2
  @test blocklength(a) == 1
  @test a[Block(1)] == 3:4
  @test label(a[Block(1)]) == "y"
  ax = only(axes(a))
  @test ax == 1:length(a)
  @test length(ax) == length(a)
  @test blocklengths(ax) == blocklengths(a)
  @test blocklabels(ax) == blocklabels(a)

  x = gradedrange(["x" => 2, "y" => 3])
  a = x[2:4][1:2]
  @test a isa GradedUnitRange
  @test length(a) == 2
  @test blocklength(a) == 2
  @test a[Block(1)] == 2:2
  @test label(a[Block(1)]) == "x"
  @test a[Block(2)] == 3:3
  @test label(a[Block(2)]) == "y"
  ax = only(axes(a))
  @test ax == 1:length(a)
  @test length(ax) == length(a)
  @test blocklengths(ax) == blocklengths(a)
  @test blocklabels(ax) == blocklabels(a)

  x = gradedrange(["x" => 2, "y" => 3])
  a = x[Block(2)[2:3]]
  @test a isa LabelledUnitRange
  @test length(a) == 2
  @test a == 4:5
  @test label(a) == "y"
  ax = only(axes(a))
  @test ax == 1:length(a)
  @test length(ax) == length(a)
  @test label(ax) == label(a)

  x = gradedrange(["x" => 2, "y" => 3, "z" => 4])
  a = x[Block(2):Block(3)]
  @test a isa GradedUnitRange
  @test length(a) == 7
  @test blocklength(a) == 2
  @test blocklengths(a) == [3, 4]
  @test blocklabels(a) == ["y", "z"]
  @test a[Block(1)] == 3:5
  @test a[Block(2)] == 6:9
  ax = only(axes(a))
  @test ax == 1:length(a)
  @test length(ax) == length(a)
  @test blocklengths(ax) == blocklengths(a)
  @test blocklabels(ax) == blocklabels(a)

  x = gradedrange(["x" => 2, "y" => 3, "z" => 4])
  a = x[[Block(3), Block(2)]]
  # This is a BlockSparseArray since BlockArray
  # doesn't support axes with general integer
  # element types. That is being fixed in:
  # https://github.com/JuliaArrays/BlockArrays.jl/pull/405
  # TODO: Change to `BlockVector` once we update
  # `GradedAxes` once the axes of `BlockArray`
  # are generalized.
  @test a isa BlockSparseVector
  @test length(a) == 7
  @test blocklength(a) == 2
  # TODO: `BlockArrays` doesn't define `blocklengths`
  # `blocklengths(::BlockVector)`, unbrake this test
  # once it does.
  @test_broken blocklengths(a) == [4, 3]
  @test blocklabels(a) == ["z", "y"]
  @test a[Block(1)] == 6:9
  @test a[Block(2)] == 3:5
  ax = only(axes(a))
  @test ax == 1:length(a)
  @test length(ax) == length(a)
  # TODO: Change to:
  # @test blocklengths(ax) == blocklengths(a)
  # once `blocklengths(::BlockVector)` is defined.
  @test blocklengths(ax) == [4, 3]
  @test blocklabels(ax) == blocklabels(a)

  x = gradedrange(["x" => 2, "y" => 3, "z" => 4])
  a = x[[Block(3)[2:3], Block(2)[2:3]]]
  # This is a BlockSparseArray since BlockArray
  # doesn't support axes with general integer
  # element types. That is being fixed in:
  # https://github.com/JuliaArrays/BlockArrays.jl/pull/405
  # TODO: Change to `BlockVector` once we update
  # `GradedAxes` once the axes of `BlockArray`
  # are generalized.
  @test a isa BlockSparseVector
  @test length(a) == 4
  @test blocklength(a) == 2
  # TODO: `BlockArrays` doesn't define `blocklengths`
  # for `BlockVector`, should it?
  @test_broken blocklengths(a) == [2, 2]
  @test blocklabels(a) == ["z", "y"]
  @test a[Block(1)] == 7:8
  @test a[Block(2)] == 4:5
  ax = only(axes(a))
  @test ax == 1:length(a)
  @test length(ax) == length(a)
  # TODO: Change to:
  # @test blocklengths(ax) == blocklengths(a)
  # once `blocklengths(::BlockVector)` is defined.
  @test blocklengths(ax) == [2, 2]
  @test blocklabels(ax) == blocklabels(a)
end
end
