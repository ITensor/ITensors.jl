using Test: @testset

@testset "NamedDimsArrays $(@__FILE__)" begin
  include("../ext/NamedDimsArraysSparseArraysBaseExt/test/runtests.jl")
end
