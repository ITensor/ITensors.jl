@eval module $(gensym())
using Test: @test, @testset
using NDTensors.TagSets
using NDTensors.SortedSets
using NDTensors.SmallVectors
using NDTensors.InlineStrings
using NDTensors.Dictionaries

@testset "Test NDTensors.TagSets" begin
  for data_type in (Vector,) # SmallVector{10})
    d1 = data_type{String31}(["1", "3", "5"])
    d2 = data_type{String31}(["2", "3", "6"])
    for set_type in (Indices, SortedSet)
      s1 = TagSet(set_type(d1))
      s2 = TagSet(set_type(d2))

      @test issetequal(union(s1, s2), ["1", "2", "3", "5", "6"])
      @test issetequal(setdiff(s1, s2), ["1", "5"])
      @test issetequal(symdiff(s1, s2), ["1", "2", "5", "6"])
      @test issetequal(intersect(s1, s2), ["3"])

      # TagSet interface
      @test issetequal(addtags(s1, ["4"]), ["1", "3", "4", "5"])
      @test issetequal(removetags(s1, ["3"]), ["1", "5"])
      @test issetequal(replacetags(s1, ["3"], ["6", "7"]), ["1", "5", "6", "7"])
      @test issetequal(replacetags(s1, ["3", "4"], ["6, 7"]), ["1", "3", "5"])

      # Only test if `isinsertable`. Make sure that is false
      # for `SmallVector`.
      ## @test issetequal(insert!(copy(s1), "4"), ["1", "3", "4", "5"])
      ## @test issetequal(delete!(copy(s1), "3"), ["1", "5"])
    end
  end
end
end
