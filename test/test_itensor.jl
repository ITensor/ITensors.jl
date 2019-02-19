using ITensors,
      LinearAlgebra, # For tr()
      Combinatorics, # For permutations()
      Random,        # To set a seed
      Test

Random.seed!(12345)

digits(::Type{T},i,j,k) where {T} = T(i*10^2+j*10+k)

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
    A1u = addtags(A1,"u")
    @test hasindex(A1u,s1("Site,s=1,u"))
    @test hasindex(A1u,s1("Site,s=1,u,0"))
    @test hasindex(A1u,l("Link,u"))
    @test hasindex(A1u,l("Link,u,0"))
    @test hasindex(A1u,l("Link,u")')
    @test hasindex(A1u,l("Link,u,1"))

    A1u = addtags(A1,"u","Link")
    @test hasindex(A1u,s1)
    @test hasindex(A1u,s1("Site,s=1"))
    @test hasindex(A1u,s1("Site,s=1,0"))
    @test hasindex(A1u,l("Link,u"))
    @test hasindex(A1u,l("Link,u,0"))
    @test hasindex(A1u,l("Link,u")')
    @test hasindex(A1u,l("Link,u,1"))

    A1u = addtags(A1,"u","0")
    @test hasindex(A1u,s1("Site,s=1,u"))
    @test hasindex(A1u,s1("Site,s=1,0,u"))
    @test hasindex(A1u,l("Link,u"))
    @test hasindex(A1u,l("Link,u,0"))
    @test hasindex(A1u,l')
    @test hasindex(A1u,l("Link")')
    @test hasindex(A1u,l("1,Link"))

    A1u = addtags(A1,"u","Link,0")
    @test hasindex(A1u,s1)
    @test hasindex(A1u,s1("Site,s=1"))
    @test hasindex(A1u,s1("Site,s=1,0"))
    @test hasindex(A1u,l("Link,u"))
    @test hasindex(A1u,l("Link,u,0"))
    @test hasindex(A1u,l')
    @test hasindex(A1u,l("Link")')
    @test hasindex(A1u,l("Link,1"))

    A1u = addtags(A1,"u","Link,1")
    @test hasindex(A1u,s1)
    @test hasindex(A1u,s1("Site,s=1"))
    @test hasindex(A1u,s1("Site,s=1,0"))
    @test hasindex(A1u,l)
    @test hasindex(A1u,l("Link"))
    @test hasindex(A1u,l("Link,0"))
    @test hasindex(A1u,l("Link,u")')
    @test hasindex(A1u,l("Link,u,1"))
  end
  @testset "removetags(::ITensor,::String,::String)" begin
    A2r = removetags(A2,"Site")
    @test hasindex(A2r,s2("s=2"))
    @test hasindex(A2r,l')
    @test hasindex(A2r,l'')

    A2r = removetags(A2,"Link","1")
    @test hasindex(A2r,s2)
    @test hasindex(A2r,l("")')
    @test hasindex(A2r,l("1"))
    @test hasindex(A2r,l'')
  end
  @testset "replacetags(::ITensor,::String,::String)" begin
    A2r = replacetags(A2,"Site","Temp")
    @test hasindex(A2r,s2("Temp,s=2"))
    @test hasindex(A2r,l')
    @test hasindex(A2r,l'')

    A2r = replacetags(A2,"Link","Temp")
    @test hasindex(A2r,s2)
    @test hasindex(A2r,l("Temp")')
    @test hasindex(A2r,l("Temp")'')

    A2r = replacetags(A2,"Link","Temp","1")
    @test hasindex(A2r,s2)
    @test hasindex(A2r,l("Temp")')
    @test hasindex(A2r,l'')

    A2r = replacetags(A2,"Link,2","Temp,3")
    @test hasindex(A2r,s2)
    @test hasindex(A2r,l')
    @test hasindex(A2r,l("Temp")''')
    @test hasindex(A2r,l("Temp,3"))

    A2r = replacetags(A2,"1","5")
    @test hasindex(A2r,s2)
    @test hasindex(A2r,prime(l,5))
    @test hasindex(A2r,l("Link,5"))
    @test hasindex(A2r,l'')
    @test hasindex(A2r,l("Link,2"))
  end
  @testset "prime(::ITensor,::String)" begin
    A2p = prime(A2)
    @test A2p==A2'
    @test hasindex(A2p,s2')
    @test hasindex(A2p,l'')
    @test hasindex(A2p,l''')
    
    A2p = prime(A2,2)
    A2p = A2''
    @test hasindex(A2p,s2'')
    @test hasindex(A2p,l''')
    @test hasindex(A2p,l'''')

    A2p = prime(A2,"s=2")
    @test hasindex(A2p,s2')
    @test hasindex(A2p,l')
    @test hasindex(A2p,l'')
  end
end

@testset "ITensor, Dense{$SType} storage" for SType ∈ (Float64,)#,ComplexF64)
  mi,mj,mk,ml,mα = 2,3,4,5,6,7
  i = Index(mi,"i")
  j = Index(mj,"j")
  k = Index(mk,"k")
  l = Index(ml,"l")
  α = Index(mα,"α") 
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
      U,S,V = svd(A,j,l)
      u = commonindex(U,S)
      v = commonindex(S,V)
      @test A≈U*S*V
      @test U*dag(prime(U,u))≈δ(SType,u,u') atol=1e-14
      @test V*dag(prime(V,v))≈δ(SType,v,v') atol=1e-14
    end

    @testset "Test SVD truncation" begin 
        M = randn(4,4)
        (U,s,V) = svd(M)
        ii = Index(4)
        jj = Index(4)
        S = Diagonal(s)
        T = ITensor(IndexSet(ii,jj),Dense{Float64}(vec(U*S*V')))
        (U,S,V) = svd(T,ii;maxm=2)
        @test norm(U*S*V-T)≈sqrt(s[3]^2+s[4]^2)
    end 

    @testset "Test QR decomposition of an ITensor" begin
      Q,R = qr(A,i,l)
      q = commonindex(Q,R)
      @test A≈Q*R
      @test Q*dag(prime(Q,q))≈δ(SType,q,q') atol=1e-14
    end

    @testset "Test polar decomposition of an ITensor" begin
      U,P = polar(A,k,l)
      @test A≈U*P
      #Note: this is only satisfied when left dimensions 
      #are greater than right dimensions
      uinds = commoninds(U,P)
      UUᵀ =  U*dag(prime(U,"u"))
      for ii ∈ dim(uinds[1]), jj ∈ dim(uinds[2])
        @test UUᵀ[uinds[1](ii),uinds[2](jj),prime(uinds[1])(ii),prime(uinds[2])(jj)]≈one(SType) atol=1e-14
      end
    end

  end # End ITensor factorization testset
end # End Dense storage test
