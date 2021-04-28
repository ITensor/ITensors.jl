using ITensors
using Test

@testset "Test argsdict function" begin
  empty!(ARGS)
  push!(
    ARGS,
    "x",
    "n = 1",
    "nf :: Float64 = 2",
    "ni :: AutoType = 2",
    "ns :: String = 2",
    "nc :: ComplexF64 = 3",
    "x = 2e-1, 2e-3, 0.1",
    "2::AutoType",
    "N = 1e-3",
    "vc = ComplexF64[1 + 2im, 3]",
    "3",
    "--autotype",
    "vf = [1.0, 3]",
    "y = true",
    "1+2im",
    "s = \"use_qr\"",
    "--stringtype",
    "y",
  )
  args = argsdict()
  empty!(ARGS)

  @test args["_arg1"] == "x"
  @test args["nf"] == 2.0
  @test args["nc"] == 3.0 + 0.0im
  @test args["ns"] == "2"
  @test args["ni"] == 2
  @test args["y"] == true
  @test args["N"] == 0.001
  @test args["x"] == (0.2, 0.002, 0.1)
  @test args["_arg2"] == 2
  @test args["vc"] == Complex{Float64}[1.0 + 2.0im, 3.0 + 0.0im]
  @test args["_arg3"] == "3"
  @test args["vf"] == [1.0, 3.0]
  @test args["n"] == 1
  @test args["_arg4"] == 1 + 2im
  @test args["s"] == "use_qr"
  @test args["_arg5"] == "y"

  push!(
    ARGS,
    "x",
    "n -> 1",
    "nf :: Float64 -> 2",
    "ni :: AutoType -> 2",
    "ns :: String -> 2",
    "nc :: ComplexF64 -> 3",
    "x -> 2e-1, 2e-3, 0.1",
    "2",
    "N -> 1e-3",
    "vc -> ComplexF64[1 + 2im, 3]",
    "3",
    "--autotype",
    "vf -> [1.0, 3]",
    "y -> true",
    "1+2im",
    "s -> \"use_qr\"",
    "--stringtype",
    "y",
  )
  args = argsdict(;
    first_arg=2,
    delim="->",
    as_symbols=true,
    default_named_type=String,
    default_positional_type=ITensors.AutoType,
    prefix="test",
  )
  empty!(ARGS)

  @test args[:nf] == 2.0
  @test args[:ni] == 2
  @test args[:ns] == "2"
  @test args[:nc] == 3.0 + 0.0im
  @test args[:y] == true
  @test args[:N] == "1e-3"
  @test args[:x] == "2e-1, 2e-3, 0.1"
  @test args[:test1] == 2
  @test args[:vc] == "ComplexF64[1 + 2im, 3]"
  @test args[:test2] == 3
  @test args[:vf] == [1.0, 3.0]
  @test args[:n] == "1"
  @test args[:test3] == 1 + 2im
  @test args[:s] == "use_qr"
  @test args[:test4] == "y"

  #
  # Check for some syntax errors
  #

  push!(ARGS, "x y=2")
  @test_throws ErrorException argsdict()
  empty!(ARGS)

  push!(ARGS, "x=y=2")
  @test_throws ErrorException argsdict()
  empty!(ARGS)

  push!(ARGS, "x::MyType = 2")
  @test_throws UndefVarError argsdict()
  empty!(ARGS)

  push!(ARGS, "x = y")
  @test_throws UndefVarError argsdict()
  empty!(ARGS)
end
