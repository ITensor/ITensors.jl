using ITensors, Test
using Combinatorics: permutations

@testset "Combiner" begin

i = Index(2,"i")
j = Index(3,"j")
k = Index(4,"k")
l = Index(5,"l")

A = randomITensor(i, j, k, l)

@testset "Two index combiner" begin
    for inds_ij ∈ permutations([i,j])
        C,c = combiner(inds_ij...)
        B = A*C
        @test hasinds(B, l, k, c)
        @test c == commonindex(B, C)
        D = B*C
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
    end
    for inds_il ∈ permutations([i,l])
        C,c = combiner(inds_il...)
        B = A*C
        @test hasinds(B, j, k)
        @test c == commonindex(B, C)
        D = B*C
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
    end
    for inds_ik ∈ permutations([i,k])
        C,c = combiner(inds_ik...)
        B = A*C
        @test hasinds(B, j, l)
        @test c == commonindex(B, C)
        D = B*C
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
    end
    for inds_jk ∈ permutations([j,k])
        C,c = combiner(inds_jk...)
        B = A*C
        @test hasinds(B, i, l)
        @test c == commonindex(B, C)
        D = B*C
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
        D = C*B
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
        B = C*A
        @test hasinds(B, i, l)
        @test c == commonindex(B, C)
        D = B*C
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
        D = C*B
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
    end
    for inds_jl ∈ permutations([j,l])
        C,c = combiner(inds_jl...)
        B = A*C
        @test hasinds(B, i, k)
        @test c == commonindex(B, C)
        D = B*C
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
        D = C*B
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
        B = C*A
        @test hasinds(B, i, k)
        @test c == commonindex(B, C)
        D = B*C
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
        D = C*B
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
    end
    for inds_kl ∈ permutations([k,l])
        C,c = combiner(inds_kl...)
        B = A*C
        @test hasinds(B, i, j)
        @test c == commonindex(B, C)
        D = B*C
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
        D = C*B
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
        B = C*A
        @test hasinds(B, i, j)
        @test c == commonindex(B, C)
        D = B*C
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
        D = C*B
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
    end
end

@testset "Three index combiner" begin
    for inds_ijl ∈ permutations([i,j,l])
        C,c = combiner(inds_ijl...)
        B = A*C
        @test hasindex(B, k)
        @test c == commonindex(B, C)
        D = B*C
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
        D = C*B
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
        B = C*A
        @test hasindex(B, k)
        @test c == commonindex(B, C)
        D = B*C
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
        D = C*B
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
    end
    for inds_ijk ∈ permutations([i,j,k])
        C,c = combiner(inds_ijk...)
        B = A*C
        @test hasindex(B, l)
        @test c == commonindex(B, C)
        D = B*C
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
        D = C*B
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
        B = C*A
        @test hasindex(B, l)
        @test c == commonindex(B, C)
        D = B*C
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
        D = C*B
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
    end
    for inds_jkl ∈ permutations([j,k,l])
        C,c = combiner(inds_jkl...)
        B = A*C
        @test hasindex(B, i)
        @test c == commonindex(B, C)
        D = B*C
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
        D = C*B
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
        B = C*A
        @test hasindex(B, i)
        @test c == commonindex(B, C)
        D = B*C
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
        D = C*B
        @test hasinds(D, i, j, k, l)
        @test D ≈ A
    end
end

@testset "SVD/Combiner should play nice" begin
    cmb, ci = combiner(i, j, k)
    Ac = A*cmb
    U,S,V,spec,u,v = svd(Ac, ci)
    Uc = cmb*U
    Ua,Sa,Va,spec,ua,va = svd(A, i, j, k)
    replaceindex!(Ua, ua, u)
    @test A ≈ cmb*Ac 
    @test A ≈ Ac*cmb
    @test Ua*cmb ≈ U
    @test cmb*Ua ≈ U
    @test Ua ≈ Uc
    @test Uc*S*V ≈ A
    @test (cmb*Ua)*S*V ≈ Ac
    cmb, ci = combiner(i, j)
    Ac = A*cmb
    U,S,V,spec,u,v = svd(Ac, ci)
    Uc = U*cmb
    Ua,Sa,Va,spec,ua,va = svd(A, i, j)
    replaceindex!(Ua, ua, u)
    @test Ua ≈ Uc
    @test Ua*cmb ≈ U
    @test cmb*Ua ≈ U
    @test Uc*S*V ≈ A
    @test (cmb*Ua)*S*V ≈ Ac
end

@testset "mult/Combiner should play nice" begin
    cmb, ci = combiner(i, j, k)
    Ac = A*cmb
    B = randomITensor(l)
    C = Ac*B
    @test C*cmb ≈ A*B
end

@testset "Replace index combiner" begin
    C,nl = combiner(l, tags="nl")
    B = A*C
    replaceindex!(B, nl, l)
    @test B == A 
end

end
