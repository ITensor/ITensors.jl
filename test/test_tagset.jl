using ITensors,
      Test

@testset "TagSet" begin
  ts = TagSet("t3,t2,t1")
  ts2 = copy(ts)
  @test ts == ts2
  @test hastags(ts,"t1")
  @test hastags(ts,"t2")
  @test hastags(ts,"t3")
  @test hastags(ts,"t3,t1")
  @test !hastags(ts,"t4")

  t1 = TagSet("t1")
  t2 = TagSet("t2")
  t3 = TagSet("t3")
  @test ts[1] == t1[1]
  @test ts[2] == t2[1]
  @test ts[3] == t3[1]

  @testset "Ignore Whitespace" begin
    ts = TagSet(" aaa , bb bb  , ccc    ")
    @test hastags(ts," aaa ")
    @test hastags(ts,"aaa")
    @test hastags(ts," aa a ")
    @test hastags(ts,"bbbb")
  end
  @testset "Remove tags" begin
    ts1 = TagSet("x,y,z")
    ts2 = TagSet("x,z")
    @test removetags(ts1,"y") == ts2
  end
  @testset "Tag too long" begin
    @test_throws ErrorException TagSet("ijklmnopq")
    @test_throws ErrorException TagSet("abcd,ijklmnopq")
    @test_throws ErrorException TagSet("ijklmnopqr,abcd")
  end
end

