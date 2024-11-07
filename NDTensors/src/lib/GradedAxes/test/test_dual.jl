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
  findblock,
  mortar,
  combine_blockaxes
using NDTensors.GradedAxes:
  AbstractGradedUnitRange,
  GradedAxes,
  GradedUnitRangeDual,
  LabelledUnitRangeDual,
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
using NDTensors.LabelledNumbers:
  LabelledInteger, LabelledUnitRange, label, label_type, labelled, labelled_isequal, unlabel
using Test: @test, @test_broken, @testset
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

@testset "LabelledUnitRangeDual" begin
  la = labelled(1:2, U1(1))
  @test la isa LabelledUnitRange
  @test label(la) == U1(1)
  @test blocklabels(la) == [U1(1)]
  @test unlabel(la) == 1:2
  @test la == 1:2
  @test !isdual(la)
  @test labelled_isequal(la, la)
  @test space_isequal(la, la)
  @test label_type(la) == U1

  @test iterate(la) == (1, 1)
  @test iterate(la) == (1, 1)
  @test iterate(la, 1) == (2, 2)
  @test isnothing(iterate(la, 2))

  lad = dual(la)
  @test lad isa LabelledUnitRangeDual
  @test label(lad) == U1(-1)
  @test blocklabels(lad) == [U1(-1)]
  @test unlabel(lad) == 1:2
  @test lad == 1:2
  @test labelled_isequal(lad, lad)
  @test space_isequal(lad, lad)
  @test !labelled_isequal(la, lad)
  @test !space_isequal(la, lad)
  @test isdual(lad)
  @test nondual(lad) === la
  @test dual(lad) === la
  @test label_type(lad) == U1

  @test iterate(lad) == (1, 1)
  @test iterate(lad) == (1, 1)
  @test iterate(lad, 1) == (2, 2)
  @test isnothing(iterate(lad, 2))

  lad2 = lad[1:1]
  @test lad2 isa LabelledUnitRangeDual
  @test label(lad2) == U1(-1)
  @test unlabel(lad2) == 1:1

  laf = flip(la)
  @test laf isa LabelledUnitRangeDual
  @test label(laf) == U1(1)
  @test unlabel(laf) == 1:2
  @test labelled_isequal(la, laf)
  @test !space_isequal(la, laf)

  ladf = flip(dual(la))
  @test ladf isa LabelledUnitRange
  @test label(ladf) == U1(-1)
  @test unlabel(ladf) == 1:2

  lafd = dual(flip(la))
  @test lafd isa LabelledUnitRange
  @test label(lafd) == U1(-1)
  @test unlabel(lafd) == 1:2

  # check default behavior for objects without dual
  la = labelled(1:2, 'x')
  lad = dual(la)
  @test lad isa LabelledUnitRangeDual
  @test label(lad) == 'x'
  @test blocklabels(lad) == ['x']
  @test unlabel(lad) == 1:2
  @test lad == 1:2
  @test labelled_isequal(lad, lad)
  @test space_isequal(lad, lad)
  @test labelled_isequal(la, lad)
  @test !space_isequal(la, lad)
  @test isdual(lad)
  @test nondual(lad) === la
  @test dual(lad) === la

  laf = flip(la)
  @test laf isa LabelledUnitRangeDual
  @test label(laf) == 'x'
  @test unlabel(laf) == 1:2

  ladf = flip(lad)
  @test ladf isa LabelledUnitRange
  @test label(ladf) == 'x'
  @test unlabel(ladf) == 1:2
end

@testset "GradedUnitRangeDual" begin
  for a in
      [gradedrange([U1(0) => 2, U1(1) => 3]), gradedrange([U1(0) => 2, U1(1) => 3])[1:5]]
    ad = dual(a)
    @test ad isa GradedUnitRangeDual
    @test ad isa AbstractGradedUnitRange
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
    @test axes(Base.Slice(a)) isa Tuple{typeof(a)}
    @test AbstractUnitRange{Int}(ad) == 1:5
    b = combine_blockaxes(ad, ad)
    @test b isa GradedUnitRangeDual
    @test b == 1:5
    @test space_isequal(b, ad)

    for x in iterate(ad)
      @test x == 1
      @test label(x) == U1(0)
    end
    for x in iterate(ad, labelled(3, U1(-1)))
      @test x == 4
      @test label(x) == U1(-1)
    end

    @test blockfirsts(ad) == [labelled(1, U1(0)), labelled(3, U1(-1))]
    @test blocklasts(ad) == [labelled(2, U1(0)), labelled(5, U1(-1))]
    @test blocklength(ad) == 2
    @test blocklengths(ad) == [2, 3]
    @test blocklabels(ad) == [U1(0), U1(-1)]
    @test label.(blocklengths(ad)) == [U1(0), U1(-1)]
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

    @test isdual(ad[Block(1)])
    @test isdual(ad[Block(1)[1:1]])
    @test ad[Block(1)] isa LabelledUnitRangeDual
    @test ad[Block(1)[1:1]] isa LabelledUnitRangeDual
    @test label(ad[Block(2)]) == U1(-1)
    @test label(ad[Block(2)[1:1]]) == U1(-1)

    I = mortar([Block(2)[1:1]])
    g = ad[I]
    @test length(g) == 1
    @test label(first(g)) == U1(-1)
    @test isdual(g[Block(1)])

    @test isdual(axes(ad[[Block(1)]], 1))  # used in view(::BlockSparseVector, [Block(1)])
    @test isdual(axes(ad[mortar([Block(1)[1:1]])], 1))  # used in view(::BlockSparseVector, [Block(1)[1:1]])
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
