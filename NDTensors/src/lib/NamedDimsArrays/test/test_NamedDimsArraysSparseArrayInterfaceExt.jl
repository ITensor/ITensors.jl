using Test: @testset

@testset "NamedDimsArrays $(@__FILE__)" begin
  include("../ext/NamedDimsArraysSparseArrayInterfaceExt/test/runtests.jl")
end
