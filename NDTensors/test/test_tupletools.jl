@eval module $(gensym())
using Test: @testset, @test
using NDTensors: NDTensors

@testset "Test non-exported tuple tools" begin
    @test NDTensors.diff((1, 3, 6, 4)) == (2, 3, -2)
    @test NDTensors.diff((1, 2, 3)) == (1, 1)
end

@testset "Test deleteat" begin
    t = (1, 2, 3, 4)
    t = NDTensors.deleteat(t, 2)
    @test t == (1, 3, 4)

    # deleteat with mixed-type Tuple
    t = ('a', 2, 'c', 4)
    t = NDTensors.deleteat(t, 2)
    @test t == ('a', 'c', 4)
    t = NDTensors.deleteat(t, 2)
    @test t == ('a', 4)
end

@testset "Test insertat" begin
    t = (1, 2)
    t = NDTensors.insertat(t, (3, 4), 2)
    @test t == (1, 3, 4)

    # insertat with mixed-type Tuple
    t = (1, 'b')
    t = NDTensors.insertat(t, ('c'), 2)
    @test t == (1, 'c')
end

end
