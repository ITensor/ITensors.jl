using Test

@testset "Examples" begin
  examples_path = joinpath(@__DIR__, "..", "examples")
  example_files = filter(starts_and_ends_with("ex_", ".jl"), readdir(examples_path))
  for file in example_files
    file_path = joinpath(examples_path, file)
    println("Testing file $(file_path)")
    empty!(ARGS)
    push!(ARGS, "false")
    @test !isnothing(include(file_path))
    empty!(ARGS)
  end
end
