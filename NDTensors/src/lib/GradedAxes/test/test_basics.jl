@eval module $(gensym())
using BlockArrays: Block, BlockVector, blockedrange, blocklength, blocklengths, findblock
using NDTensors.GradedAxes:
  GradedAxes,
  blockmergesortperm,
  dual,
  fuse,
  gradedrange,
  isdual,
  sector,
  sectors,
  tensor_product
using Test: @test, @testset, @test_throws

struct U1
  dim::Int
end
Base.isless(l1::U1, l2::U1) = isless(l1.dim, l2.dim)
GradedAxes.fuse(l1::U1, l2::U1) = U1(l1.dim + l2.dim)
GradedAxes.dual(l::U1) = U1(-l.dim)

@testset "Basics" begin
  a = gradedrange([U1(0), U1(1)], [2, 3])
  @test a isa GradedAxes.GradedUnitRange
  @test a == gradedrange([U1(0) => 2, U1(1) => 3])
  @test a == gradedrange([U1(0), U1(1)], blockedrange([2, 3]))
  @test length(a) == 5
  @test a == 1:5
  @test a[Block(1)] == 1:2
  @test a[Block(2)] == 3:5
  @test blocklength(a) == 2 # Number of sectors
  @test blocklengths(a) == [2, 3]
  # TODO: Maybe rename to `labels`, `label`.
  @test sectors(a) == [U1(0), U1(1)]
  @test sector(a, Block(1)) == U1(0)
  @test sector(a, Block(2)) == U1(1)
  @test findblock(a, 1) == Block(1)
  @test findblock(a, 2) == Block(1)
  @test findblock(a, 3) == Block(2)
  @test findblock(a, 4) == Block(2)
  @test findblock(a, 5) == Block(2)
  @test sector(a, 1) == U1(0)
  @test sector(a, 2) == U1(0)
  @test sector(a, 3) == U1(1)
  @test sector(a, 4) == U1(1)
  @test sector(a, 5) == U1(1)

  # test error for invalid input
  @test_throws DomainError gradedrange([U1(0), U1(1)], [2, 3, 4])
  @test_throws DomainError gradedrange([U1(0), U1(1)], blockedrange([2, 3, 4]))

  # Naive tensor product, no sorting and merging
  a = gradedrange([U1(0), U1(1)], [2, 3])
  a2 = tensor_product(a, a)
  @test a2 isa GradedAxes.GradedUnitRange
  @test a2 == gradedrange([U1(0) => 4, U1(1) => 6, U1(1) => 6, U1(2) => 9])
  @test length(a2) == 25
  @test a2 == 1:25
  @test blocklength(a2) == 4
  @test blocklengths(a2) == [4, 6, 6, 9]
  @test sectors(a2) == [U1(0), U1(1), U1(1), U1(2)]
  @test sector(a2, Block(1)) == U1(0)
  @test sector(a2, Block(2)) == U1(1)
  @test sector(a2, Block(3)) == U1(1)
  @test sector(a2, Block(4)) == U1(2)

  # Fusion tensor product, with sorting and merging
  a = gradedrange([U1(0), U1(1)], [2, 3])
  a2 = fuse(a, a)
  @test a2 isa GradedAxes.GradedUnitRange
  @test a2 == gradedrange([U1(0) => 4, U1(1) => 12, U1(2) => 9])
  @test length(a2) == 25
  @test a2 == 1:25
  @test blocklength(a2) == 3
  @test blocklengths(a2) == [4, 12, 9]
  @test sectors(a2) == [U1(0), U1(1), U1(2)]
  @test sector(a2, Block(1)) == U1(0)
  @test sector(a2, Block(2)) == U1(1)
  @test sector(a2, Block(3)) == U1(2)

  # The partitioned permutation needed to sort
  # and merge an unsorted graded space
  a = gradedrange([U1(0), U1(1)], [2, 3])
  perm_a = blockmergesortperm(tensor_product(a, a))
  @test perm_a == BlockVector([Block(1), Block(2), Block(3), Block(4)], [1, 2, 1])
  @test tensor_product(a, a)[perm_a] == fuse(a, a)

  a = gradedrange([U1(0), U1(1)], [2, 3])
  @test !isdual(a)
  a = dual(a)
  @test isdual(a)
  @test sectors(a) == [U1(0), U1(-1)]
  @test sector(a, Block(1)) == U1(0)
  @test sector(a, Block(2)) == U1(-1)

  # Permute blocks
  a = gradedrange([U1(0), U1(1), U1(2)], [2, 3, 4])
  perm_a = a[[Block(3), Block(1), Block(2)]]
  @test perm_a == gradedrange([U1(2), U1(0), U1(1)], [4, 2, 3])
  @test sectors(perm_a) == [U1(2), U1(0), U1(1)]
  @test blocklengths(perm_a) == [4, 2, 3]

  # Test fusion with dual spaces
  a = gradedrange([U1(0), U1(1)], [2, 3])
  a2 = fuse(a, a)
  @test !isdual(a2)
  @test sectors(a2) == [U1(0), U1(1), U1(2)]

  a = gradedrange([U1(0), U1(1)], [2, 3])
  a2 = fuse(a, a; isdual=true)
  @test isdual(a2)
  @test sectors(a2) == [U1(0), U1(1), U1(2)]

  a = gradedrange([U1(0), U1(1)], [2, 3])
  a2 = fuse(dual(a), dual(a))
  @test isdual(a2)
  @test sectors(a2) == [U1(-2), U1(-1), U1(0)]

  a = gradedrange([U1(0), U1(1)], [2, 3])
  a2 = fuse(dual(a), dual(a); isdual=false)
  @test !isdual(a2)
  @test sectors(a2) == [U1(-2), U1(-1), U1(0)]

  a = gradedrange([U1(0), U1(1)], [2, 3])
  a2 = fuse(a, dual(a))
  @test !isdual(a2)
  @test sectors(a2) == [U1(-1), U1(0), U1(1)]

  a = gradedrange([U1(0), U1(1)], [2, 3])
  a2 = fuse(a, dual(a); isdual=true)
  @test isdual(a2)
  @test sectors(a2) == [U1(-1), U1(0), U1(1)]

  a = gradedrange([U1(0), U1(1)], [2, 3])
  a2 = fuse(dual(a), a)
  @test !isdual(a2)
  @test sectors(a2) == [U1(-1), U1(0), U1(1)]

  a = gradedrange([U1(0), U1(1)], [2, 3])
  a2 = fuse(dual(a), a; isdual=true)
  @test isdual(a2)
  @test sectors(a2) == [U1(-1), U1(0), U1(1)]
end
end
