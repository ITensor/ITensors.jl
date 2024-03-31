@eval module $(gensym())
using ITensors: ITensors
using Test: @test, @testset
include("utils/TestITensorsExportedNames/TestITensorsExportedNames.jl")
using .TestITensorsExportedNames: ITENSORS_EXPORTED_NAMES
@testset "Test $name is exported" for name in ITENSORS_EXPORTED_NAMES
  @test Base.isexported(ITensors, name)
end
end
