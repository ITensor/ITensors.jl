using ITensors, Test

@testset "ITensorNetwork" begin
    i, j, k, l = Index(4), Index(5), Index(6), Index(7)
    x, y, z = randomITensor(i, j), randomITensor(j, k), randomITensor(k, l)
    it = ITensorNetwork([[x, y] ,z])
    @test evaluate(it) ≈ x * y * z
    @test ITensors.flatten(it) == [x, y, z]
end

@testset "contraction order" begin
    i, j, k, l = Index(4), Index(5), Index(6), Index(7)
    x, y, z = randomITensor(i, j), randomITensor(j, k), randomITensor(k, l)
    lt = [x, y, z]
    optcode = optimize_code(lt, TreeSA())
    tc, sc, rw = timespacereadwrite_complexity(optcode)
    @test tc ≈ 8.169925001442312
    @test flop(optcode) ≈ 288
    @test label_elimination_order(optcode) == [j, k]
    @test peak_memory(optcode) == 94
    @test evaluate(optcode) ≈ x * y * z
end

@testset "square lattice contraction" begin
    # square lattice test
    n = 6
    vindices = [Index(2) for i=1:n, j=1:n]
    hindices = [Index(2) for i=1:n, j=1:n]
    tensors = vec([randomITensor(hindices[i, j], hindices[mod1(i+1, n), j], vindices[i,j], vindices[i, mod1(j+1,n)]) for i=1:n, j=1:n])
    opt = optimize_code(tensors, TreeSA(ntrials=1))
    @test timespacereadwrite_complexity(opt)[2] <= 2n+1

    r1 = foldl(*, tensors)
    r2 = evaluate(opt)
    @test r1 ≈ r2
end
