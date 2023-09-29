using Test
using NDTensors.SortedSets

@testset "Test NDTensors.SortedSets" begin
  s1 = SortedSet([1, 3, 5])
  s2 = SortedSet([2, 3, 6])

  # Set interface
  @test union(s1, s2) == SortedSet([1, 2, 3, 5, 6])
  @test setdiff(s1, s2) == SortedSet([1, 5])
  @test symdiff(s1, s2) == SortedSet([1, 2, 5, 6])
  @test intersect(s1, s2) == SortedSet([3])
  @test insert!(copy(s1), 4) == SortedSet([1, 3, 4, 5])
  @test delete!(copy(s1), 3) == SortedSet([1, 5])

  # TagSet interface
  @test addtags()
end
