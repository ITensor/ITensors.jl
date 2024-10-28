@eval module $(gensym())
using BlockArrays:
  Block,
  BlockedOneTo,
  blockaxes,
  blockedrange,
  blockfirsts,
  blockisequal,
  blocklasts,
  blocklength,
  blocklengths,
  blocks,
  findblock
using NDTensors.GradedAxes:
  GradedAxes,
  GradedUnitRangeDual,
  OneToOne,
  blocklabels,
  blockmergesortperm,
  blocksortperm,
  dual,
  flip,
  space_isequal,
  gradedrange,
  isdual,
  nondual
using NDTensors.LabelledNumbers: LabelledInteger, label, labelled, labelled_isequal
using Test: @test, @testset
struct U1
  n::Int
end
GradedAxes.dual(c::U1) = U1(-c.n)
Base.isless(c1::U1, c2::U1) = c1.n < c2.n

@testset "AbstractUnitRange" begin
  a0 = OneToOne()
  @test !isdual(a0)
  @test dual(a0) isa OneToOne
  @test space_isequal(a0, a0)
  @test labelled_isequal(a0, a0)
  @test space_isequal(a0, dual(a0))

  a = 1:3
  ad = dual(a)
  @test !isdual(a)
  @test !isdual(ad)
  @test ad isa UnitRange
  @test space_isequal(ad, a)

  a = blockedrange([2, 3])
  ad = dual(a)
  @test !isdual(a)
  @test !isdual(ad)
  @test ad isa BlockedOneTo
  @test blockisequal(ad, a)
end

@testset "GradedUnitRangeDual" begin
  for a in
      [gradedrange([U1(0) => 2, U1(1) => 3]), gradedrange([U1(0) => 2, U1(1) => 3])[1:5]]
    ad = dual(a)
    @test ad isa GradedUnitRangeDual
    @test eltype(ad) == LabelledInteger{Int,U1}
    @test blocklengths(ad) isa Vector
    @test eltype(blocklengths(ad)) == eltype(blocklengths(a))

    @test space_isequal(dual(ad), a)
    @test space_isequal(nondual(ad), a)
    @test space_isequal(nondual(a), a)
    @test space_isequal(ad, ad)
    @test !space_isequal(a, ad)
    @test !space_isequal(ad, a)

    @test isdual(ad)
    @test !isdual(a)

    @test blockfirsts(ad) == [labelled(1, U1(0)), labelled(3, U1(-1))]
    @test blocklasts(ad) == [labelled(2, U1(0)), labelled(5, U1(-1))]
    @test blocklength(ad) == 2
    @test blocklengths(ad) == [2, 3]
    @test findblock(ad, 4) == Block(2)
    @test only(blockaxes(ad)) == Block(1):Block(2)
    @test blocks(ad) == [labelled(1:2, U1(0)), labelled(3:5, U1(-1))]
    @test ad[4] == 4
    @test label(ad[4]) == U1(-1)
    @test ad[2:4] == 2:4
    @test ad[2:4] isa GradedUnitRangeDual
    @test label(ad[2:4][Block(2)]) == U1(-1)
    @test ad[[2, 4]] == [2, 4]
    @test label(ad[[2, 4]][2]) == U1(-1)
    @test ad[Block(2)] == 3:5
    @test label(ad[Block(2)]) == U1(-1)
    @test ad[Block(1):Block(2)][Block(2)] == 3:5
    @test label(ad[Block(1):Block(2)][Block(2)]) == U1(-1)
    @test ad[[Block(2), Block(1)]][Block(1)] == 3:5
    @test label(ad[[Block(2), Block(1)]][Block(1)]) == U1(-1)
    @test ad[[Block(2)[1:2], Block(1)[1:2]]][Block(1)] == 3:4
    @test label(ad[[Block(2)[1:2], Block(1)[1:2]]][Block(1)]) == U1(-1)
    @test blocksortperm(a) == [Block(1), Block(2)]
    @test blocksortperm(ad) == [Block(1), Block(2)]
    @test blocklength(blockmergesortperm(a)) == 2
    @test blocklength(blockmergesortperm(ad)) == 2
    @test blockmergesortperm(a) == [Block(1), Block(2)]
    @test blockmergesortperm(ad) == [Block(1), Block(2)]
  end
end

@testset "flip" begin
  for a in
      [gradedrange([U1(0) => 2, U1(1) => 3]), gradedrange([U1(0) => 2, U1(1) => 3])[1:5]]
    ad = dual(a)
    @test space_isequal(flip(a), dual(gradedrange([U1(0) => 2, U1(-1) => 3])))
    @test space_isequal(flip(ad), gradedrange([U1(0) => 2, U1(-1) => 3]))

    @test blocklabels(a) == [U1(0), U1(1)]
    @test blocklabels(dual(a)) == [U1(0), U1(-1)]
    @test blocklabels(flip(a)) == [U1(0), U1(1)]
    @test blocklabels(flip(dual(a))) == [U1(0), U1(-1)]
    @test blocklabels(dual(flip(a))) == [U1(0), U1(-1)]

    @test blocklengths(a) == [2, 3]
    @test blocklengths(ad) == [2, 3]
    @test blocklengths(flip(a)) == [2, 3]
    @test blocklengths(flip(ad)) == [2, 3]
    @test blocklengths(dual(flip(a))) == [2, 3]

    @test !isdual(a)
    @test isdual(ad)
    @test isdual(flip(a))
    @test !isdual(flip(ad))
    @test !isdual(dual(flip(a)))
  end
end
end
