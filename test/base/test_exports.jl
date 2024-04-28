@eval module $(gensym())
using ITensors: ITensors
using Test: @test, @testset
include("utils/TestITensorsExportedNames/TestITensorsExportedNames.jl")
using .TestITensorsExportedNames: ITENSORS_EXPORTED_NAMES
@testset "Test exports of ITensors" begin
  @test issetequal(names(ITensors), ITENSORS_EXPORTED_NAMES)
end
end
