using ITensors,
      LinearAlgebra, # For tr()
      Random,        # To set a seed
      Test

Random.seed!(12345)

digits(::Type{T},i,j,k) where {T} = T(i*10^2+j*10+k)

@testset "ITensor constructors" begin
  i = Index(2,"i")
  j = Index(2,"j")

  @testset "Default" begin
    A = ITensor()
    @test store(A) isa Dense{Nothing}
    @test isNull(A)
  end

  @testset "Undef with index" begin
    A = ITensor(undef, i)
    @test store(A) isa Dense{Float64}
    @test !isNull(A)
  end

  @testset "Default with indices" begin
    A = ITensor(i,j)
    @test store(A) isa Dense{Float64}
    @test !isNull(A)
  end

  @testset "Random" begin
    A = randomITensor(i,j)
    @test store(A) isa Dense{Float64}
    @test !isNull(A)
  end

  @testset "From matrix" begin
    M = [1 2; 3 4]
    A = ITensor(M,i,j)
    @test store(A) isa Dense{Float64}
    M = [1 2 3; 4 5 6]
    @test_throws DimensionMismatch ITensor(M,i,j)
  end

  @testset "Complex" begin
    A = ITensor(Complex,i,j)
    @test store(A) isa Dense{ComplexF64}
  end

  @testset "Random complex" begin
    A = randomITensor(Complex,i,j)
    @test store(A) isa Dense{ComplexF64}
  end

  @testset "From complex matrix" begin
    M = [1+2im 2; 3 4]
    A = ITensor(M,i,j)
    @test store(A) isa Dense{ComplexF64}
  end

end

@testset "Convert to complex" begin
  i = Index(2,"i")
  j = Index(2,"j")
  A = randomITensor(i,j)
  B = complex(A)
  for ii ∈ dim(i), jj ∈ dim(j)
    @test complex(A[i(ii),j(jj)]) == B[i(ii),j(jj)]
  end
end

@testset "similar" begin
  i = Index(2,"i")
  j = Index(2,"j")
  A = randomITensor(i,j)
  B = similar(A)
  @test inds(B) == inds(A)
  @test_throws ErrorException similar(A, ComplexF32)
end

@testset "fill!" begin
  i = Index(2,"i")
  j = Index(2,"j")
  A = randomITensor(i,j)
  fill!(A, 1.0)
  @test all(data(store(A)) .== 1.0)
end

@testset "copyto!" begin
  i = Index(2,"i")
  j = Index(2,"j")
  M = [1 2; 3 4]
  A = ITensor(M,i,j)
  N = 2*M 
  B = ITensor(N,i,j)
  copyto!(A, B)
  @test A == B
  @test data(store(A)) == vec(N)
  A = ITensor(M,i,j)
  B = ITensor(N,j,i)
  copyto!(A, B)
  @test A == B
  @test data(store(A)) == vec(transpose(N))
end

@testset "Unary -" begin
  i = Index(2,"i")
  j = Index(2,"j")
  M = [1 2; 3 4]
  A = ITensor(M,i,j)
  @test -A == ITensor(-M, i, j) 
end

@testset "dot" begin
  i = Index(2,"i")
  a = [1.0; 2.0]
  b = [3.0; 4.0]
  A = ITensor(a,i)
  B = ITensor(b,i)
  @test dot(A, B) == 11.0
end

@testset "add and axpy" begin
  i = Index(2,"i")
  a = [1.0; 2.0]
  b = [3.0; 4.0]
  A = ITensor(a,i)
  B = ITensor(b,i)
  c = [5.0; 8.0]
  @test A + B == ITensor([4.0; 6.0], i)
  @test axpy!(2.0, A, B) == ITensor(c, i) 
  a = [1.0; 2.0]
  b = [3.0; 4.0]
  A = ITensor(a,i)
  B = ITensor(b,i)
  c = [8.0; 12.0]
  @test add!(A, 2.0, 2.0, B) == ITensor(c, i) 
  
end

@testset "mul! and rmul!" begin
  i = Index(2,"i")
  a = [1.0; 2.0]
  b = [2.0; 4.0]
  A = ITensor(a,i)
  A2 = copy(A)
  B = ITensor(b,i)
  @test mul!(A2, A, 2.0) == B
  @test rmul!(A, 2.0) == B
  i = Index(2,"i")
  j = Index(2,"j")
  M = [1 2; 3 4]
  A = ITensor(M,i,j)
  N = 2*M 
  B = ITensor(N,j,i)
  @test data(store(mul!(B, A, 2.0))) == 2.0*vec(transpose(M))

end

@testset "show" begin
  i = Index(2,"i")
  a = [1.0; 2.0]
  A = ITensor(a,i)
  s = split(sprint(show, A), '\n')
  @test s[1] == "ITensor ord=1 " * sprint(show, i)
  @test s[2] == "Dense{Float64}"
  @test s[3] == " 1.0"
  @test s[4] == " 2.0"
end

@testset "Test isapprox for ITensors" begin
        m,n = rand(0:20,2)
        i = Index(m)
        j = Index(n)
        realData = rand(m,n)
        complexData = realData+ zeros(m,n)*1im
        A = ITensor(realData, i,j)
        B = ITensor(complexData, i,j)
        @test A≈B
        @test B≈A
        realDataT = Array(transpose(realData))
        A = ITensor(realDataT, j,i)
        @test A≈B
        @test B≈A
    end

@testset "ITensor tagging and priming" begin
  s1 = Index(2,"Site,s=1")
  s2 = Index(2,"Site,s=2")
  l = Index(3,"Link")
  A1 = randomITensor(s1,l,l')
  A2 = randomITensor(s2,l',l'')
  @testset "findindex(::ITensor,::String)" begin
    @test s1==findindex(A1,"Site")
    @test s1==findindex(A1,"s=1")
    @test s1==findindex(A1,"s=1,Site")
    @test l==findindex(A1,"Link,0")
    @test l'==findindex(A1,"1")
    @test l'==findindex(A1,"Link,1")
    @test s2==findindex(A2,"Site")
    @test s2==findindex(A2,"s=2")
    @test s2==findindex(A2,"Site")
    @test s2==findindex(A2,"0")
    @test s2==findindex(A2,"s=2,0")
    @test s2==findindex(A2,"Site,0")
    @test s2==findindex(A2,"s=2,Site,0")
    @test l'==findindex(A2,"1")
    @test l'==findindex(A2,"Link,1")
    @test l''==findindex(A2,"2")
    @test l''==findindex(A2,"Link,2")
  end
  @testset "addtags(::ITensor,::String,::String)" begin
    s1u = addtags(s1,"u")
    lu = addtags(l,"u")

    A1u = addtags(A1,"u")
    @test hasinds(A1u,s1u,lu,lu')

    A1u = addtags(A1,"u","Link")
    @test hasinds(A1u,s1,lu,lu')

    A1u = addtags(A1,"u","0")
    @test hasinds(A1u,s1u,lu,l')

    A1u = addtags(A1,"u","Link,0")
    @test hasinds(A1u,s1,lu,l')

    A1u = addtags(A1,"u","Link,1")
    @test hasinds(A1u,s1,l,lu')
  end
  @testset "removetags(::ITensor,::String,::String)" begin
    A2r = removetags(A2,"Site")
    @test hasinds(A2r,removetags(s2,"Site"),l',l'')

    A2r = removetags(A2,"Link","1")
    @test hasinds(A2r,s2,removetags(l,"Link")',l'')
  end
  @testset "replacetags(::ITensor,::String,::String)" begin
    s2tmp = replacetags(s2,"Site","Temp")
    ltmp = replacetags(l,"Link","Temp")

    A2r = replacetags(A2,"Site","Temp")
    @test hasinds(A2r,s2tmp,l',l'')

    A2r = replacetags(A2,"Link","Temp")
    @test hasinds(A2r,s2,ltmp',ltmp'')

    A2r = replacetags(A2,"Link","Temp","1")
    @test hasinds(A2r,s2,ltmp',l'')

    A2r = replacetags(A2,"Link,2","Temp,3")
    @test hasinds(A2r,s2,l',ltmp''')

    A2r = replacetags(A2,"1","5")
    @test hasinds(A2r,s2,prime(l,5),l'')
  end
  @testset "prime(::ITensor,::String)" begin
    A2p = prime(A2)
    @test A2p==A2'
    @test hasinds(A2p,s2',l'',l''')
    
    A2p = prime(A2,2)
    A2p = A2''
    @test hasinds(A2p,s2'',l''',l'''')

    A2p = prime(A2,"s=2")
    @test hasinds(A2p,s2',l',l'')
  end

  @testset "mapprime" begin
    @test hasinds(mapprime(A2,1,7),s2,l^7,l'')
    @test hasinds(mapprime(A2,0,1),s2',l',l'')
  end
end

@testset "ITensor, Dense{$SType} storage" for SType ∈ (Float64,ComplexF64)
  mi,mj,mk,ml,mα = 2,3,4,5,6,7
  i = Index(mi,"i")
  j = Index(mj,"j")
  k = Index(mk,"k")
  l = Index(ml,"l")
  α = Index(mα,"alpha")
  @testset "Set and get values with IndexVals" begin
    A = ITensor(SType,i,j,k)
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      A[k(kk),j(jj),i(ii)] = digits(SType,ii,jj,kk)
    end
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      @test A[j(jj),k(kk),i(ii)]==digits(SType,ii,jj,kk)
    end
    @test_throws ErrorException A[1]
  end
  @testset "Test permute(ITensor,Index...)" begin
    A = randomITensor(SType,i,k,j)
    permA = permute(A,k,j,i)
    @test k==inds(permA)[1]
    @test j==inds(permA)[2]
    @test i==inds(permA)[3]
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      @test A[k(kk),i(ii),j(jj)]==permA[i(ii),j(jj),k(kk)]
    end
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      @test A[k(kk),i(ii),j(jj)]==permA[i(ii),j(jj),k(kk)]
    end
    @testset "getindex and setindex with vector of IndexVals" begin
        k_inds = [k(kk) for kk ∈ 1:dim(k)]
        for ii ∈ 1:dim(i), jj ∈ 1:dim(j)
          @test A[k_inds,i(ii),j(jj)]==permA[i(ii),j(jj),k_inds]
        end
        for ii ∈ 1:dim(i), jj ∈ 1:dim(j)
            A[k_inds,i(ii),j(jj)]=collect(1:length(k_inds))
        end
        permA = permute(A,k,j,i)
        for ii ∈ 1:dim(i), jj ∈ 1:dim(j)
          @test A[k_inds,i(ii),j(jj)]==permA[i(ii),j(jj),k_inds]
        end
    end
  end
  @testset "Set and get values with Ints" begin
    A = ITensor(SType,i,j,k)
    A = permute(A,k,i,j)
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      A[kk,ii,jj] = digits(SType,ii,jj,kk)
    end
    A = permute(A,i,j,k)
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      @test A[ii,jj,kk]==digits(SType,ii,jj,kk)
    end
  end
  @testset "Test scalar(ITensor)" begin
    x = SType(34)
    A = ITensor(x)
    @test x==scalar(A)
    A = ITensor(SType,i,j,k)
    @test_throws ArgumentError scalar(A)
    # test the storage_scalar error throw
    ds = Dense{Float64}(rand(10))
    @test_throws ErrorException ITensor.storage_scalar(ds)
  end
  @testset "Test norm(ITensor)" begin
    A = randomITensor(SType,i,j,k)
    @test norm(A)≈sqrt(scalar(dag(A)*A))
  end
  @testset "Test add ITensors" begin
    A = randomITensor(SType,i,j,k)
    B = randomITensor(SType,k,i,j)
    C = A+B
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      @test C[i(ii),j(jj),k(kk)]==A[j(jj),i(ii),k(kk)]+B[i(ii),k(kk),j(jj)]
    end
    @test Array(permute(C,i,j,k))==Array(permute(A,i,j,k))+Array(permute(B,i,j,k))
  end

  @testset "Test factorizations of an ITensor" begin

    A = randomITensor(SType,i,j,k,l)

    @testset "Test SVD of an ITensor" begin
      U,S,Vh = svd(A,(j,l))
      u = commonindex(U,S)
      v = commonindex(S,Vh)
      @test A≈U*S*Vh
      @test U*dag(prime(U,u))≈δ(SType,u,u') atol=1e-14
      @test Vh*dag(prime(Vh,v))≈δ(SType,v,v') atol=1e-14
    end

    @testset "Test SVD truncation" begin 
        M = randn(4,4) + randn(4,4)*1.0im
        (U,s,Vh) = svd(M)
        ii = Index(4)
        jj = Index(4)
        S = Diagonal(s)
        T = ITensor(IndexSet(ii,jj),Dense{ComplexF64}(vec(U*S*Vh)))
        (U,S,Vh) = svd(T,ii;maxdim=2)
        @test norm(U*S*Vh-T)≈sqrt(s[3]^2+s[4]^2)
    end 

    @testset "Test QR decomposition of an ITensor" begin
      Q,R = qr(A,(i,l))
      q = commonindex(Q,R)
      @test A≈Q*R
      @test Q*dag(prime(Q,q))≈δ(SType,q,q') atol=1e-14
    end

    @testset "Test polar decomposition of an ITensor" begin
      U,P = polar(A,(k,l))
      @test A≈U*P
      #Note: this is only satisfied when left dimensions 
      #are greater than right dimensions
      uinds = commoninds(U,P)
      UUᵀ =  U*dag(prime(U,uinds))
      for ii ∈ dim(uinds[1]), jj ∈ dim(uinds[2])
        @test UUᵀ[uinds[1](ii),uinds[2](jj),prime(uinds[1])(ii),prime(uinds[2])(jj)]≈one(SType) atol=1e-14
      end
    end
  end # End ITensor factorization testset
end # End Dense storage test

@testset "Converting Real and Complex Storage" begin

  @testset "Add Real and Complex" begin
    i = Index(2,"i")
    j = Index(2,"j")
    TC = randomITensor(ComplexF64,i,j)
    TR = randomITensor(Float64,i,j)

    S1 = TC+TR
    S2 = TR+TC
    @test typeof(S1.store) == Dense{ComplexF64}
    @test typeof(S2.store) == Dense{ComplexF64}
    for ii=1:dim(i),jj=1:dim(j)
      @test S1[i(ii),j(jj)] ≈ TC[i(ii),j(jj)]+TR[i(ii),j(jj)]
      @test S2[i(ii),j(jj)] ≈ TC[i(ii),j(jj)]+TR[i(ii),j(jj)]
    end
  end

end


