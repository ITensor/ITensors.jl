using SafeTestsets: @safetestset

@safetestset "BaseExtensions" begin
  using NDTensors.BaseExtensions: BaseExtensions
  using Test: @test, @testset
  @testset "replace $(typeof(collection))" for collection in
                                               (["a", "b", "c"], ("a", "b", "c"))
    r1 = BaseExtensions.replace(collection, "b" => "d")
    @test r1 == typeof(collection)(["a", "d", "c"])
    @test typeof(r1) === typeof(collection)
    r2 = BaseExtensions.replace(collection, "b" => "d", "a" => "e")
    @test r2 == typeof(collection)(["e", "d", "c"])
    @test typeof(r2) === typeof(collection)
  end
end
