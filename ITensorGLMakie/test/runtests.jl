using ITensors
using ITensorGLMakie
using Test

starts_and_ends_with(file, st, en) = startswith(file, st) && endswith(file, en)
starts_and_ends_with(st, en) = file -> starts_and_ends_with(file, st, en)

test_path = joinpath(@__DIR__)
test_files = filter(starts_and_ends_with("test_", ".jl"), readdir(test_path))
@testset "ITensorGLMakie.jl" for file in test_files
  file_path = joinpath(test_path, file)
  println("Running test $(file_path)")
  include(file_path)
end
