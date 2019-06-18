using ITensors,
      Test

@testset "TagSet" begin
  ts = TagSet("t3,t2,t1")
  ts2 = copy(ts)
  @test ts == ts2
  @test "t1" ∈ ts
  @test "t2" ∈ ts
  @test "t3" ∈ ts
  @test "t4" ∉ ts
  @test issorted(ts.tags)

  @testset "Ignore Whitespace" begin
    ts = TagSet(" aaa , bb bb  , ccc    ")
    @test " aaa " ∈ ts
    @test "aaa" ∈ ts
    @test " aa a " ∈ ts
    @test "bbbb" ∈ ts
  end
end

