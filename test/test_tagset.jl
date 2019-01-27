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
end

