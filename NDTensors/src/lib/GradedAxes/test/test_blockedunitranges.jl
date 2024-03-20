@eval module $(gensym())
using BlockArrays: blockfirts, blocklasts, blocklength, blocklengths
using NDTensors.GradedAxes.BlockedUnitRanges: blockedrange
using Test: @test, @testset
@testset "NDTensors.GradedAxes.BlockedUnitRanges" begin
  @testset "eltype=$elt" for elt in (UInt32, Int)
    a = blockedrange(elt[2, 3, 4])
    @test eltype(a) === elt
    @test length(a) == 9
    @test blockfirsts(a) == [1, 3, 6]
    @test eltype(blockfirsts(a)) === elt
    @test blocklength(a) == 3
    @test blocklengths(a) == [2, 3, 4]
    @test eltype(blocklengths(a)) === Int
  end
end
end
