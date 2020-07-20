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

  sw = Sweeps([
         "maxdim" "mindim" "cutoff" "noise"
          50       10       1e-12    1e-7
          100      20       1e-12    1e-8
          200      20       1e-12    1e-10
          400      20       1e-12    0
          800      20       1e-12    1e-11
          800      20       1e-12    0
         ])

  @test nsweep(sw) == 6

  @test maxdim(sw, 1) == 50
  @test maxdim(sw, 2) == 100
  @test maxdim(sw, 3) == 200
  @test maxdim(sw, 4) == 400
  @test maxdim(sw, 5) == 800
  @test maxdim(sw, 6) == 800

  @test mindim(sw, 1) == 10
  for n in 2:6
    @test mindim(sw, n) == 20
  end

  for n in 1:6
    @test cutoff(sw, n) == 1e-12
  end

  @test noise(sw, 1) == 1e-7
  @test noise(sw, 2) == 1e-8
  @test noise(sw, 3) == 1e-10
  @test noise(sw, 4) == 0
  @test noise(sw, 5) == 1e-11
  @test noise(sw, 6) == 0

end

nothing
