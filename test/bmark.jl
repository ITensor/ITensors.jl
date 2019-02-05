using ITensors,
      LinearAlgebra, # For tr()
      Combinatorics, # For permutations()
      CuArrays,
      BenchmarkTools

digits(::Type{T},i,j,k) where {T} = T(i*10^2+j*10+k)
mi,mj,mk,ml,mα= 32,32,32,32,32
i = Index(mi,"i")
j = Index(mj,"j")
k = Index(mk,"k")
l = Index(ml,"l")
α = Index(mα,"α")
for inds_ijkl ∈ permutations([i,j,k,l]), inds_klα ∈ permutations([k,l,α])
    Aijkl = randomITensor(Float64,i,j,k,l)
    Aklα = randomITensor(Float64,k,l,α)
    println("transfer time:")
    @btime randomCuITensor(Float64,i,j,k,l)
    @btime randomCuITensor(Float64,k,l,α)
    cAijkl = randomCuITensor(Float64,i,j,k,l)
    cAklα = randomCuITensor(Float64,k,l,α)
    Aijkl = permute(Aijkl,inds_ijkl...)
    Aklα = permute(Aklα,inds_klα...)
    cAijkl = permute(cAijkl,inds_ijkl...)
    cAklα = permute(cAklα,inds_klα...)
    println("CPU only time:")
    @btime $Aklα*$Aijkl
    println("GPU only time:")
    @btime $cAklα*$cAijkl
    println()
end
