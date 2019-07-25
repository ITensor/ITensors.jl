using ITensors,
      Test

import ITensors.SmallString, ITensors.IntChar, ITensors.push
@testset "ctors" begin
    s = SmallString()
    @test length(s) == 0
    @test ITensors.isNull(s)
end

@testset "setindex" begin
    s = SmallString()
    @test ITensors.isNull(s)
    t = Base.setindex(s, IntChar(1), 1)
    @test length(t) == 1
    @test !ITensors.isNull(t)
end

@testset "push" begin
    s = SmallString()
    @test ITensors.isNull(s)
    t = push(s, IntChar(1))
    @test !ITensors.isNull(t)
end

@testset "comparison" begin
    s = SmallString()
    u = push(s, IntChar(1))
    t = push(s, IntChar(1))
    @test u == t
    t = push(s, IntChar(2))
    @test u < t
end
