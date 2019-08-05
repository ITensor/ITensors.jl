using ITensors,
      LinearAlgebra, # For tr()
      Random,        # To set a seed
      Test

Random.seed!(12345)

@testset "diagITensor" begin
  d = 3
  i = Index(d,"i")
  j = Index(d,"j")
  k = Index(d,"k")
  l = Index(d,"l")
  m = Index(d,"m")
  n = Index(d,"n")
  o = Index(d,"o")
  p = Index(d,"p")
  q = Index(d,"q")

  v = [1:d...]
  vr = randn(d)

  @testset "Zero constructor (order 2)" begin
    D = diagITensor(i,j)

    @test eltype(D) == Float64
    for ii = 1:d, jj = 1:d
      if ii == jj
        @test D[i(ii),j(jj)] == 0.0
      else
        @test D[i(ii),j(jj)] == 0.0
      end
    end
  end

  @testset "Zero constructor (order 3)" begin
    D = diagITensor(i,j,k)

    @test eltype(D) == Float64
    for ii = 1:d, jj = 1:d, kk = 1:d
      if ii == jj == kk
        @test D[i(ii),j(jj),k(kk)] == 0.0
      else
        @test D[i(ii),j(jj),k(kk)] == 0.0
      end
    end
  end

  @testset "Zero constructor (complex)" begin
    D = diagITensor(ComplexF64,i,j)

    @test eltype(D) == ComplexF64
    for ii = 1:d, jj = 1:d
      if ii == jj
        @test D[i(ii),j(jj)] == complex(0.0)
      else
        @test D[i(ii),j(jj)] == complex(0.0)
      end
    end
  end

  @testset "Diagonal constructor (order 2)" begin
    D = diagITensor(v,i,j)

    @test eltype(D) == Float64
    for ii = 1:d, jj = 1:d
      if ii == jj
        @test D[i(ii),j(jj)] == v[ii]
      else
        @test D[i(ii),j(jj)] == 0.0
      end
    end
  end

  @testset "Diagonal constructor (order 3)" begin
    D = diagITensor(v,i,j,k)

    @test eltype(D) == Float64
    for ii = 1:d, jj = 1:d, kk = 1:d
      if ii == jj == kk
        @test D[i(ii),j(jj),k(kk)] == v[ii]
      else
        @test D[i(ii),j(jj),k(kk)] == 0.0
      end
    end
  end

  @testset "Diagonal constructor (complex)" begin
    vc = v+im*v
    D = diagITensor(vc,i,j,k)

    @test eltype(D) == ComplexF64
    for ii = 1:d, jj = 1:d, kk = 1:d
      if ii == jj == kk
        @test D[i(ii),j(jj),k(kk)] == vc[ii]
      else
        @test D[i(ii),j(jj),k(kk)] == complex(0.0)
      end
    end
  end

  @testset "Set elements" begin
    D = diagITensor(i,j,k)

    for ii = 1:d
      D[i(ii),j(ii),k(ii)] = ii
    end

    @test eltype(D) == Float64
    for ii = 1:d, jj = 1:d, kk = 1:d
      if ii == jj == kk
        @test D[i(ii),j(jj),k(kk)] == ii
      else
        @test D[i(ii),j(jj),k(kk)] == 0.0
      end
    end

    # Can't set off-diagonal elements
    @test_throws ErrorException D[i(2),j(1),k(1)] = 0.0
    @test_throws ErrorException D[i(1),j(2),k(1)] = 0.0
  end

  @testset "Convert to dense" begin
    D = diagITensor(v,i,j,k)
    T = dense(D)
    
    @test store(T) isa Dense{Float64}
    for ii = 1:d, jj = 1:d, kk = 1:d
      if ii == jj == kk
        @test T[ii,ii,ii] == ii
      else
        @test T[i(ii),j(jj),k(kk)] == 0.0
      end
    end
  end

  @testset "Add (Diag+Diag)" begin
    v1 = randn(d)
    v2 = randn(d)
    D1 = diagITensor(v1,i,j,k)
    D2 = diagITensor(v2,i,j,k)

    v3 = v1 + v2
    D3 = D1 + D2

    @test D3 ≈ dense(D1) + dense(D2) 
    for ii = 1:d
      @test D3[ii,ii,ii] == v3[ii]
    end
  end

  @testset "Add (Diag+Dense)" begin
    D = diagITensor(vr,i,j,k)
    A = randomITensor(k,j,i)

    R = D + A

    @test R ≈ dense(D) + A
    for ii = 1:d
      @test R[ii,ii,ii] ≈ D[ii,ii,ii] + A[ii,ii,ii]
    end
  end

  @testset "Add (Dense+Diag)" begin
    D = diagITensor(vr,i,j,k)
    A = randomITensor(i,k,j)

    R = A + D

    @test R ≈ dense(D) + A
    for ii = 1:d
      @test R[ii,ii,ii] ≈ D[ii,ii,ii] + A[ii,ii,ii]
    end
  end

  @testset "Contraction (all contracted)" begin
    D = diagITensor(v,i,j,k)
    A = randomITensor(j,k,i)
    
    @test D*A ≈ dense(D)*A
    @test A*D ≈ dense(D)*A
  end

  @testset "Contraction (all dense contracted)" begin
    D = diagITensor(v,j,k,i)
    A = randomITensor(i,j)
    
    @test D*A ≈ dense(D)*A
    @test A*D ≈ dense(D)*A
  end

  @testset "Contraction Diag*Dense (general)" begin
    D = diagITensor(v,l,i,k,j)
    A = randomITensor(m,k,n,l)

    @test D*A ≈ dense(D)*A
    @test A*D ≈ dense(D)*A
  end

  @testset "Contraction Diag*Diag (all contracted)" begin
    D1 = diagITensor(v,l,i,k,j)
    D2 = diagITensor(vr,j,l,i,k)

    @test D1*D2 ≈ dense(D1)*dense(D2)
    @test D2*D1 ≈ dense(D1)*dense(D2)
  end

  @testset "Contraction Diag*Diag (all contracted)" begin
    D1 = diagITensor(v,l,i,k,j)
    D2 = diagITensor(vr,m,k,n,l)

    @test D1*D2 ≈ dense(D1)*dense(D2)
    @test D2*D1 ≈ dense(D1)*dense(D2)
  end

end

