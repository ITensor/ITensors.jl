using ITensors, Test

@testset "TagSet" begin
  ts = TagSet("t3,t2,t1")
  ts2 = copy(ts)
  @test ts == ts2
  @test hastags(ts, "t1")
  @test hastags(ts, "t2")
  @test hastags(ts, "t3")
  @test hastags(ts, "t3,t1")
  @test !hastags(ts, "t4")
  @test TagSet(ts) === ts

  @test ITensors.commontags() == ts""
  @test ITensors.commontags(ts"a,b", ts"a,c") == ts"a"
  @test ITensors.commontags(Index(2, "a,b,x"), Index(3, "x,a,c"), Index(4, "x,a,z,w")) ==
    ts"a,x"

  t1 = TagSet("t1")
  t2 = TagSet("t2")
  t3 = TagSet("t3")
  @test ts[1] == t1[1]
  @test ts[2] == t2[1]
  @test ts[3] == t3[1]

  @testset "Empty TagSet" begin
    ts1 = TagSet()
    @test length(ts1) == 0

    ts2 = TagSet("")
    @test ts2 == ts1
    @test length(ts2) == 0
  end

  @testset "Ignore Whitespace" begin
    ts = TagSet(" aaa , bb bb  , ccc    ")
    @test hastags(ts, " aaa ")
    @test hastags(ts, "aaa")
    @test hastags(ts, " aa a ")
    @test hastags(ts, "bbbb")
  end

  @testset "Remove tags" begin
    ts1 = TagSet("x,y,z")
    ts2 = TagSet("x,z")
    @test removetags(ts1, "y") == ts2
  end

  @testset "Unicode tags" begin
    ts = TagSet("α")
    @test length(ts) == 1
    @test hastags(ts, "α")
    @test ts[1] == ITensors.SmallString("α")

    ts = TagSet("α,β")
    @test length(ts) == 2
    @test hastags(ts, "β")
    @test hastags(ts, "α")
    @test ts[1] == ITensors.SmallString("α")
    @test ts[2] == ITensors.SmallString("β")

    ts = TagSet("αβγδϵζηθ,ijklmnop,qrstuvwx,ΑΒΓΔΕΖΗΘ")
    @test length(ts) == 4
    @test hastags(ts, "αβγδϵζηθ")
    @test hastags(ts, "ijklmnop")
    @test hastags(ts, "qrstuvwx")
    @test hastags(ts, "ΑΒΓΔΕΖΗΘ")
    @test ts[1] == ITensors.SmallString("ijklmnop")
    @test ts[2] == ITensors.SmallString("qrstuvwx")
    @test ts[3] == ITensors.SmallString("ΑΒΓΔΕΖΗΘ")
    @test ts[4] == ITensors.SmallString("αβγδϵζηθ")
  end

  @testset "Tag long" begin
    ts = TagSet("abcdefghijklmnop,ijklmnopqabcdefg")
    @test length(ts) == 2
    @test hastags(ts, "abcdefghijklmnop")
    @test hastags(ts, "ijklmnopqabcdefg")
  end

  @testset "Tag too long" begin
    @test !ITensors.using_strict_tags()
    @test TagSet("ijklmnopqabcdefgh") == TagSet("ijklmnopqabcdefg")
    @test TagSet("abcd,ijklmnopqabcdefgh") == TagSet("abcd,ijklmnopqabcdefg")
    @test TagSet("ijklmnopqabcdefgh,abcd") == TagSet("abcd,ijklmnopqabcdefg")
    ITensors.set_strict_tags!(true)
    @test ITensors.using_strict_tags()
    @test_throws ErrorException TagSet("ijklmnopqabcdefgh")
    ITensors.set_strict_tags!(false)
  end

  @testset "Too many tags" begin
    @test !ITensors.using_strict_tags()
    @test TagSet("a,b,c,d,e,f") == TagSet("a,b,c,d")
    @test addtags(TagSet("a,b,c,d"), "e") == TagSet("a,b,c,d")
    @test replacetags(TagSet("a,b,c,d"), "d", "e,f") == TagSet("a,b,c,e")
    ITensors.set_strict_tags!(true)
    @test ITensors.using_strict_tags()
    @test_throws ErrorException TagSet("a,b,c,d,e,f")
    @test_throws ErrorException addtags(TagSet("a,b,c,d"), "e")
    @test_throws ErrorException replacetags(TagSet("a,b,c,d"), "d", "e,f")
    ITensors.set_strict_tags!(false)
  end

  @testset "Integer Tags" begin
    ts = TagSet("123")
    @test length(ts) == 1
    @test hastags(ts, "123")
  end

  @testset "Show TagSet" begin
    ts = TagSet("Site,n=2")
    @test length(sprint(show, ts)) > 1
  end

  @testset "Iterate Tagset" begin
    ts = TagSet("Site, n=2")
    @test [tag for tag in ts] == [ts[1], ts[2]]
  end

  @testset "addtags" begin
    ts = TagSet("Blue")
    @test hastags(ts, "Blue")

    ts = addtags(ts, "Red")
    @test hastags(ts, "Blue")
    @test hastags(ts, "Red")

    ts = addtags(ts, "Green")
    @test hastags(ts, "Blue")
    @test hastags(ts, "Red")
    @test hastags(ts, "Green")

    ts = addtags(ts, "Yellow")
    @test hastags(ts, "Blue")
    @test hastags(ts, "Red")
    @test hastags(ts, "Green")
    @test hastags(ts, "Yellow")

    @test addtags(ts, "Orange") == ts
    ITensors.set_strict_tags!(true)
    @test_throws ErrorException addtags(ts, "Orange")
    ITensors.set_strict_tags!(false)
  end
end

nothing
