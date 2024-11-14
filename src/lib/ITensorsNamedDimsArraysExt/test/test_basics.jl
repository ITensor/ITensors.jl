@eval module $(gensym())
using BlockArrays: blocklengths
using ITensors: ITensor, Index, QN, dag, inds, plev, random_itensor
using ITensors.ITensorsNamedDimsArraysExt: to_nameddimsarray
using NDTensors: tensor
using NDTensors.BlockSparseArrays: BlockSparseArray, block_stored_length
using NDTensors.GradedAxes: isdual
using NDTensors.LabelledNumbers: label
using NDTensors.NamedDimsArrays: NamedDimsArray, unname
using Test: @test, @testset
@testset "to_nameddimsarray" begin
  i = Index([QN(0) => 2, QN(1) => 3])
  a = random_itensor(i', dag(i))
  b = to_nameddimsarray(a)
  @test b isa ITensor
  @test plev(inds(b)[1]) == 1
  @test plev(inds(b)[2]) == 0
  @test inds(b)[1] == i'
  @test inds(b)[2] == dag(i)
  nb = tensor(b)
  @test nb isa NamedDimsArray{Float64}
  bb = unname(nb)
  @test bb isa BlockSparseArray{Float64}
  @test !isdual(axes(bb, 1))
  @test isdual(axes(bb, 2))
  @test blocklengths(axes(bb, 1)) == [2, 3]
  @test blocklengths(axes(bb, 2)) == [2, 3]
  @test label.(blocklengths(axes(bb, 1))) == [QN(0), QN(1)]
  @test label.(blocklengths(axes(bb, 2))) == [QN(0), QN(-1)]
  @test block_stored_length(bb) == 2
  @test b' * b ≈ to_nameddimsarray(a' * a)
end
end
