using Test
using ITensors.LazyApply: LazyApply, Sum, Prod, Scaled, materialize

@testset "LazyApply general functionality" begin
  @test (materialize ∘ materialize ∘ materialize)(exp(Prod([1, 2, Sum([3, 4])]))) ==
    exp(prod([1, 2, sum([3 + 4])]))
  @test_broken materialize(2 * Sum([1, 2, Prod([3, 4])])) == 2 * sum([1, 2, prod([3, 4])])
  @test 2 * Sum([1, 2, Prod([3, 4])]) == Sum([2, 4, 2Prod([3, 4])])
  @test 2 * Sum([1, 2, Prod([3, 4])]) isa Sum
  @test_broken materialize(Sum() + 3 + 4) == sum([3, 4])
  @test_broken Sum() + 3 + 4 isa Sum
  @test materialize(Sum([1, 2, 3]) + Sum([4, 5, 6])) == sum([1, 2, 3, 4, 5, 6])
  @test Sum([1, 2, 3]) + Sum([4, 5, 6]) isa Sum
  @test materialize(2 * Prod([1, 2])) == 2 * prod([1, 2])
  @test_broken 2 * Prod([1, 2]) isa Scaled
  @test_broken 2 * Prod([1, 2]) isa Scaled{<:Prod}
  @test_broken 2 * Prod([1, 2]) isa Scaled{Prod{Int}}
  @test 2 * Prod([1, 2]) isa Prod{Int}
  @test Prod([1, 2]) + Prod([3, 4]) == Sum([Prod([1, 2]), Prod([3, 4])])
  @test Prod([1, 2]) + Prod([3, 4]) isa Sum
  @test_broken materialize(Sum(Prod([1, 2]) + Prod([3, 4]))) ==
    sum([prod([1, 2]), prod([3, 4])])
  @test Prod([1, 2]) + Prod([3, 4]) == Sum([Prod([1, 2]), Prod([3, 4])])
  @test Prod([1, 2]) + Prod([3, 4]) isa Sum
  @test_broken Prod([1, 2]) - Prod([3, 4]) isa Sum
  @test materialize(Prod(["X", "Y", "Z"])) == "XYZ"
  @test Prod(["X", "Y", "Z"]) isa Prod
  @test_broken materialize(Prod() * "X" * "Y" * "Z") == "XYZ"
  @test_broken Prod() * "X" * "Y" * "Z" == Prod(["X", "Y", "Z"])
  @test_broken Prod() * "X" * "Y" * "Z" isa Prod
  @test_broken 2Prod() * "X" * "Y" == 2Prod(["X", "Y"])
  @test_broken 2Prod() * "X" * "Y" isa Scaled{<:Prod}
  @test_broken 2Prod() * "X" * "Y" isa Scaled{Prod{String}}
  @test_broken 2Prod() * "X" * "Y" isa Scaled{Prod{String},Int}
  @test_broken 2Prod(["X"]) * 3Prod(["Y"]) == 6Prod(["X", "Y"])
end
