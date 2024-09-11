@eval module $(gensym())
using BlockArrays:
  Block,
  blockaxes,
  blockedrange,
  blockfirsts,
  blocklasts,
  blocklength,
  blocklengths,
  blocks,
  findblock
using NDTensors.GradedAxes:
  GradedAxes,
  GradedUnitRangeDual,
  OneToOne,
  UnitRangeDual,
  blocklabels,
  blockmergesortperm,
  blocksortperm,
  dual,
  flip,
  gradedisequal,
  gradedrange,
  isdual,
  nondual
using NDTensors.LabelledNumbers: LabelledInteger, label, labelled
using Test: @test, @test_broken, @testset
struct U1
  n::Int
end
GradedAxes.dual(c::U1) = U1(-c.n)
Base.isless(c1::U1, c2::U1) = c1.n < c2.n

@testset "UnitRangeDual" begin
  @testset "dual(OneToOne)" begin
    a = OneToOne()
    ad = dual(a)
    @test ad isa UnitRangeDual
    @test eltype(ad) == Bool
    @test nondual(ad) === a
    @test dual(ad) === a

    @test isdual(ad)
    @test !isdual(a)
    @test length(ad) == 1
  end
  @testset "dual(UnitRange)" begin
    a = 1:3
    ad = dual(a)
    @test ad isa UnitRangeDual
    @test eltype(ad) == Int
    @test nondual(ad) === a
    @test dual(ad) === a

    @test isdual(ad)
    @test !isdual(a)
    @test length(ad) == 3
  end
  @testset "dual(BlockedOneTo)" begin
    a = blockedrange([2, 3])
    ad = dual(a)
    @test ad isa UnitRangeDual
    @test eltype(ad) == Int
    @test nondual(ad) === a
    @test dual(ad) === a

    @test isdual(ad)
    @test !isdual(a)
    @test length(ad) == 5

    @test blockfirsts(ad) == [1, 3]
    @test blocklasts(ad) == [2, 5]
    @test findblock(ad, 4) == Block(2)
    @test only(blockaxes(ad)) == Block(1):Block(2)
    @test blocks(ad) == [1:2, 3:5]
    @test ad[4] == 4
    @test ad[2:4] == 2:4
    @test ad[2:4] isa UnitRangeDual
    @test ad[[2, 4]] == [2, 4]
    @test ad[Block(2)] == 3:5
    @test ad[Block(1):Block(2)][Block(2)] == 3:5
    @test ad[[Block(2), Block(1)]][Block(1)] == 3:5
    @test ad[[Block(2)[1:2], Block(1)[1:2]]][Block(1)] == 3:4
  end
end

@testset "GradedUnitRangeDual" begin
  a = gradedrange([U1(0) => 2, U1(1) => 3])
  ad = dual(a)
  @test ad isa GradedUnitRangeDual
  @test eltype(ad) == LabelledInteger{Int,U1}

  @test gradedisequal(dual(ad), a)
  @test gradedisequal(nondual(ad), a)
  @test gradedisequal(nondual(a), a)
  @test gradedisequal(ad, ad)
  @test !gradedisequal(a, ad)
  @test !gradedisequal(ad, a)

  @test isdual(ad)
  @test !isdual(a)

  @test blockfirsts(ad) == [labelled(1, U1(0)), labelled(3, U1(-1))]
  @test blocklasts(ad) == [labelled(2, U1(0)), labelled(5, U1(-1))]
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

@testset "flip" begin
  a = gradedrange([U1(0) => 2, U1(1) => 3])
  ad = dual(a)
  @test gradedisequal(flip(a), dual(gradedrange([U1(0) => 2, U1(-1) => 3])))
  @test gradedisequal(flip(ad), gradedrange([U1(0) => 2, U1(-1) => 3]))

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
