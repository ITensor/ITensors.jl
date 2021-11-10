using Test
using ITensors.LazyApply: LazyApply, Add, Mul, ∑, ∏, α, materialize

@testset "LazyApply general functionality" begin
  @test materialize(∏([1, 2, Add(3, 4)])) == prod([1, 2, 3 + 4])
  @test ∏([1, 2, Add(3, 4)]) isa ∏
  @test materialize(3 * ∏([1, 2, Add(3, 4)])) == 3 * prod([1, 2, 3 + 4])
  @test materialize(exp(∏([1, 2, ∑([3, 4])]))) == exp(prod([1, 2, sum([3 + 4])]))
  @test materialize(2 * ∑([1, 2, ∏([3, 4])])) == 2 * sum([1, 2, prod([3, 4])])
  @test 2 * ∑([1, 2, ∏([3, 4])]) == ∑([2, 4, 2∏([3, 4])])
  @test 2 * ∑([1, 2, ∏([3, 4])]) isa ∑
  @test 2∑(["X", "Y"]) == ∑([Mul(2, "X"), Mul(2, "Y")])
  @test materialize(∑() + 3 + 4) == sum([3, 4])
  @test ∑() + 3 + 4 isa ∑
  @test materialize(∑([1, 2, 3]) + ∑([4, 5, 6])) == sum([1, 2, 3, 4, 5, 6])
  @test ∑([1, 2, 3]) + ∑([4, 5, 6]) isa ∑
  @test materialize(Add(1, 2) + Add(3, 4)) == 1 + 2 + 3 + 4
  @test Add(1, 2) + Add(3, 4) == Add(1, 2, 3, 4)
  @test Add(1, 2) + Add(3, 4) isa Add
  @test materialize(2 * Add(1, 2)) == 2 * (1 + 2)
  @test 2 * Add(1, 2) isa Add
  @test materialize(3 + Add(1, 2)) == 3 + 1 + 2
  @test 3 + Add(1, 2) isa Add
  @test materialize(2 * ∏([1, 2])) == 2 * prod([1, 2])
  @test 2 * ∏([1, 2]) isa α
  @test 2 * ∏([1, 2]) isa α{<:∏}
  @test 2 * ∏([1, 2]) isa α{∏{Int}}
  @test ∏([1, 2]) + ∏([3, 4]) == ∑([∏([1, 2]), ∏([3, 4])])
  @test ∏([1, 2]) + ∏([3, 4]) isa ∑
  @test materialize(∑(∏([1, 2]) + ∏([3, 4]))) == sum([prod([1, 2]), prod([3, 4])])
  @test ∏([1, 2]) + ∏([3, 4]) == ∑([∏([1, 2]), ∏([3, 4])])
  @test ∏([1, 2]) + ∏([3, 4]) isa ∑
  @test ∏([1, 2]) - ∏([3, 4]) isa ∑
  @test materialize(∏(["X", "Y", "Z"])) == "XYZ"
  @test ∏(["X", "Y", "Z"]) isa ∏
  @test materialize(∏() * "X" * "Y" * "Z") == "XYZ"
  @test ∏() * "X" * "Y" * "Z" == ∏(["X", "Y", "Z"])
  @test ∏() * "X" * "Y" * "Z" isa ∏
  @test 2∏() * "X" * "Y" == 2∏(["X", "Y"])
  @test 2∏() * "X" * "Y" isa α{<:∏}
  @test 2∏() * "X" * "Y" isa α{∏{String}}
  @test 2∏() * "X" * "Y" isa α{∏{String},Int}
  @test 2∏(["X"]) * 3∏(["Y"]) == 6∏(["X", "Y"])
end
