using Test

@testset "Examples" begin
  examples_path = joinpath(@__DIR__, "..", "examples")
  example_files = filter(starts_and_ends_with("ex_", ".jl"), readdir(examples_path))
  for file in example_files
    file_path = joinpath(examples_path, file)
    println("Testing file $(file_path)")
    empty!(ARGS)
    push!(ARGS, "false")
    res = include(file_path)
    @test isnothing(res) || all(isnothing, res)
    empty!(ARGS)
  end
end
