using Test
using NDTensors.SortedSets
using NDTensors.SmallVectors

@testset "Test NDTensors.SortedSets" begin
  @testset "Basic operations" begin
    for V in (Vector, MSmallVector{10}, SmallVector{10})
      for by in (+, -)
        s1 = SortedSet(V([1, 5, 3]); by)
        s2 = SortedSet(V([2, 3, 6]); by)

        @test thaw(s1) == s1
        @test SmallVectors.insert(s1, 2) isa typeof(s1)
        @test SmallVectors.insert(s1, 2) == SortedSet([1, 2, 3, 5]; by)
        @test SmallVectors.delete(s1, 3) isa typeof(s1)
        @test SmallVectors.delete(s1, 3) == SortedSet([1, 5]; by)

        # Set interface
        @test union(s1, s2) == SortedSet([1, 2, 3, 5, 6]; by)
        @test union(s1, [3]) == s1
        @test setdiff(s1, s2) == SortedSet([1, 5]; by)
        @test symdiff(s1, s2) == SortedSet([1, 2, 5, 6]; by)
        @test intersect(s1, s2) == SortedSet([3]; by)
        if SmallVectors.InsertStyle(V) isa IsInsertable
          @test insert!(copy(s1), 4) == SortedSet([1, 3, 4, 5]; by)
          @test delete!(copy(s1), 3) == SortedSet([1, 5]; by)
        end
      end
    end
  end
  @testset "Replacement behavior" begin
    s1 = SortedSet([("a", 3), ("b", 2)]; by=first)
    s2 = SortedSet([("a", 5)]; by=first)
    s = union(s1, s2)
    @test s ≠ s1
    @test issetequal(s, s1)
    @test ("a", 5) ∈ parent(s)
    @test ("a", 3) ∉ parent(s)
  end
end
