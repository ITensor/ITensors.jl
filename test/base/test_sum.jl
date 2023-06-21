using ITensors
@testset "directsum test" begin
    @testset "Sum of MPS" begin
     sites = siteinds("Boson",5,dim=5)
     psis = [randomMPS(sites) for i in 1:10]
     directsum = sum(psis,alg="directsum")
      @test typeof(directsum) == MPS
    end
    @testset "Sum of MPO" begin
        sites = siteinds("Boson",5,dim=5)
        H_s = [randomMPO(sites) for i in 1:10]
        directsum = sum(H_s,alg="directsum")
         @test typeof(directsum) == MPO
       end
end