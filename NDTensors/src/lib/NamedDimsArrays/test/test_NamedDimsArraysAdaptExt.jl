using Test: @testset

@testset "NamedDimsArrays $(@__FILE__)" begin
  include("../ext/NamedDimsArraysAdaptExt/test/runtests.jl")
end
