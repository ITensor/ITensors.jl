using Test
using NDTensors.SortedSets
using NDTensors.SmallVectors

@testset "Test NDTensors.SortedSets" begin
  for V in (Vector, MSmallVector{10}, SmallVector{10})
    s1 = SortedSet(V([1, 3, 5]))
    s2 = SortedSet(V([2, 3, 6]))

    # Set interface
    @test union(s1, s2) == SortedSet([1, 2, 3, 5, 6])
    @test setdiff(s1, s2) == SortedSet([1, 5])
    @test symdiff(s1, s2) == SortedSet([1, 2, 5, 6])
    @test intersect(s1, s2) == SortedSet([3])
    if SmallVectors.InsertStyle(V) isa IsInsertable
      @test insert!(copy(s1), 4) == SortedSet([1, 3, 4, 5])
      @test delete!(copy(s1), 3) == SortedSet([1, 5])
    end
  end
end
