using ITensors,
      Test

import ITensors.SmallString, 
       ITensors.IntChar, 
       ITensors.push,
       ITensors.isint

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

  @testset "isint" begin
    i = SmallString()
    i = push(i, '1')
    i = push(i, '2')
    i = push(i, '3')
    @test isint(i) == true

    s = SmallString()
    s = push(s, 'a')
    s = push(s, 'b')
    s = push(s, 'c')
    @test isint(s) == false
  end

  @testset "isless" begin
    s1 = SmallString() 
    s1=push(s1,'a') 
    s1=push(s1,'b')

    s2 = SmallString() 
    s2=push(s2,'x') 
    s2=push(s2,'y')

    @test isless(s1,s2) == true
    @test isless(s2,s1) == false
    @test isless(s1,s1) == false
    @test isless(s2,s2) == false
  end

end

