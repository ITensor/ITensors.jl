using Test: @testset

@testset "NamedDimsArrays $(@__FILE__)" begin
  include("../ext/NamedDimsArraysTensorAlgebraExt/test/runtests.jl")
end
