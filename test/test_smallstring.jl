using ITensors,
      Test

import ITensors.SmallString, ITensors.IntChar, ITensors.push

@testset "SmallString" begin
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
      
  @testset "convert to integer" begin
      s = SmallString()
      s = push(s, '2')
      s = push(s, '6')
      s = push(s, '7')
      i = parse(Int, s)
      @test i == 267
  end
      
  @testset "Convert to String" begin
    s = SmallString()
    s = push(s, 'a')
    s = push(s, 'b')
    s = push(s, 'c')
    sg = Base.String(s)
    for n=1:length(s)
      @test convert(Char,s[n]) == sg[n]
    end
  end

end

