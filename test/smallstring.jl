using ITensors
using Test

import ITensors: SmallString, Tag, isint, isnull, IntChar

@testset "SmallString" begin
  @testset "ctors" begin
    s = SmallString()
    @test isnull(s)
  end

  @testset "setindex" begin
    s = SmallString()
    @test isnull(s)
    t = setindex(s, IntChar(1), 1)
    @test !isnull(t)
  end

  @testset "comparison" begin
    u = SmallString("1")
    t = SmallString("1")
    @test u == t
    t = SmallString("2")
    @test u < t
  end

  @testset "Convert to String" begin
    s = SmallString("abc")
    @test typeof(s) == SmallString

    sg = String(s)
    for n in 1:length(sg)
      @test sg[n] == convert(Char, s[n])
    end

    s = SmallString("")
    sg = String(s)
    @test sg == ""
  end

  @testset "isint" begin
    i = SmallString("123")
    @test isint(i) == true

    s = SmallString("abc")
    @test isint(s) == false

    # Test maximum length
    s = SmallString("12345678")
    @test isint(s) == true
  end

  @testset "isless" begin
    s1 = SmallString("ab")
    s2 = SmallString("xy")
    @test isless(s1, s2) == true
    @test isless(s2, s1) == false
    @test isless(s1, s1) == false
    @test isless(s2, s2) == false
  end

  @testset "show" begin
    t = Tag("")
    @test sprint(show, t) == ""

    t = Tag("Red")
    @test sprint(show, t) == "Red"

    # Make sure to test maximum length tag
    t = Tag("Electron")
    @test sprint(show, t) == "Electron"
  end
end

nothing
