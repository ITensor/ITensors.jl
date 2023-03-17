using ITensors
using Test

@testset "Sweeps constructor" begin

  #Sweeps
  #1 cutoff=1.0E-12, maxdim=50, mindim=10, noise=1.0E-07
  #2 cutoff=1.0E-12, maxdim=100, mindim=20, noise=1.0E-08
  #3 cutoff=1.0E-12, maxdim=200, mindim=20, noise=1.0E-10
  #4 cutoff=1.0E-12, maxdim=400, mindim=20, noise=0.0E+00
  #5 cutoff=1.0E-12, maxdim=800, mindim=20, noise=1.0E-11
  #6 cutoff=1.0E-12, maxdim=800, mindim=20, noise=0.0E+00

  sweep_args = [
    "maxdim" "mindim" "cutoff" "noise"
    50 10 1e-12 1e-7
    100 20 1e-12 1e-8
    200 20 1e-12 1e-10
    400 20 1e-12 0
    800 20 1e-12 1e-11
    800 20 1e-12 0
  ]

  @testset "Don't specify nsweep" begin
    nsw = size(sweep_args, 1) - 1
    sw = Sweeps(sweep_args)

    @test nsweep(sw) == nsw

    @test maxdim(sw, 1) == 50
    @test maxdim(sw, 2) == 100
    @test maxdim(sw, 3) == 200
    @test maxdim(sw, 4) == 400
    for n in 5:nsw
      @test maxdim(sw, n) == 800
    end

    @test mindim(sw, 1) == 10
    for n in 2:nsw
      @test mindim(sw, n) == 20
    end

    for n in 1:nsw
      @test cutoff(sw, n) == 1e-12
    end

    @test noise(sw, 1) == 1e-7
    @test noise(sw, 2) == 1e-8
    @test noise(sw, 3) == 1e-10
    @test noise(sw, 4) == 0
    @test noise(sw, 5) == 1e-11
    @test noise(sw, 6) == 0
  end

  @testset "Specify nsweep, more than data" begin
    nsw = 7
    sw = Sweeps(nsw, sweep_args)

    @test nsweep(sw) == nsw

    @test maxdim(sw, 1) == 50
    @test maxdim(sw, 2) == 100
    @test maxdim(sw, 3) == 200
    @test maxdim(sw, 4) == 400
    for n in 5:nsw
      @test maxdim(sw, n) == 800
    end

    @test mindim(sw, 1) == 10
    for n in 2:nsw
      @test mindim(sw, n) == 20
    end

    for n in 1:nsw
      @test cutoff(sw, n) == 1e-12
    end

    @test noise(sw, 1) == 1e-7
    @test noise(sw, 2) == 1e-8
    @test noise(sw, 3) == 1e-10
    @test noise(sw, 4) == 0
    @test noise(sw, 5) == 1e-11
    @test noise(sw, 6) == 0
    @test noise(sw, 7) == 0
  end

  @testset "Specify nsweep, less than data" begin
    nsw = 5
    sw = Sweeps(nsw, sweep_args)

    @test nsweep(sw) == nsw

    @test maxdim(sw, 1) == 50
    @test maxdim(sw, 2) == 100
    @test maxdim(sw, 3) == 200
    @test maxdim(sw, 4) == 400
    for n in 5:nsw
      @test maxdim(sw, n) == 800
    end

    @test mindim(sw, 1) == 10
    for n in 2:nsw
      @test mindim(sw, n) == 20
    end

    for n in 1:nsw
      @test cutoff(sw, n) == 1e-12
    end

    @test noise(sw, 1) == 1e-7
    @test noise(sw, 2) == 1e-8
    @test noise(sw, 3) == 1e-10
    @test noise(sw, 4) == 0
    @test noise(sw, 5) == 1e-11
  end

  @testset "Variable types of input" begin
    sw = Sweeps(5)
    setnoise!(sw, 1E-8, 0)
    @test noise(sw, 1) ≈ 1E-8
    @test noise(sw, 2) ≈ 0.0
    @test noise(sw, 3) ≈ 0.0
    setcutoff!(sw, 0, 1E-8, 0, 1E-12)
    @test cutoff(sw, 1) ≈ 0.0
    @test cutoff(sw, 2) ≈ 1E-8
    @test cutoff(sw, 3) ≈ 0.0
    @test cutoff(sw, 4) ≈ 1E-12
  end

  @testset "Keyword args to constructor" begin
    sw = Sweeps(5; maxdim=[4, 8, 16], mindim=1, cutoff=[1E-5, 1E-8])
    @test maxdim(sw, 1) == 4
    @test maxdim(sw, 2) == 8
    @test maxdim(sw, 3) == 16
    @test maxdim(sw, 4) == 16
    @test maxdim(sw, 5) == 16

    @test mindim(sw, 1) == 1
    @test mindim(sw, 5) == 1

    @test cutoff(sw, 1) ≈ 1E-5
    @test cutoff(sw, 2) ≈ 1E-8
    @test cutoff(sw, 3) ≈ 1E-8
    @test cutoff(sw, 4) ≈ 1E-8
    @test cutoff(sw, 5) ≈ 1E-8

    sw = Sweeps(5; cutoff=1E-8)
    @test maxdim(sw, 1) == typemax(Int)
    @test maxdim(sw, 5) == typemax(Int)
  end
end

nothing
