#
# Optimizations:
#  - replace leftmap, rightmap with Dicts
# 

export SiteOp,
       MPOTerm,
       AutoMPO,
       terms,
       add!,
       toMPO

import LinearAlgebra.svd

###########################
# SiteOp                  # 
###########################

struct SiteOp
  name::String
  site::Int
end
name(s::SiteOp) = s.name
site(s::SiteOp) = s.site
Base.show(io::IO,s::SiteOp) = print(io,"\"$(name(s))\"($(site(s)))")

###########################
# OpTerm                  # 
###########################

const OpTerm = Vector{SiteOp}
mult(t1::OpTerm,t2::OpTerm) = isempty(t2) ? t1 : vcat(t1,t2)

###########################
# MPOTerm                 # 
###########################

struct MPOTerm
  coef::ComplexF64
  ops::OpTerm
end
coef(op::MPOTerm) = op.coef
ops(op::MPOTerm) = op.ops
length(op::MPOTerm) = length(op.ops)

function MPOTerm(c::Number,
                 op1::String,
                 i1::Int) 
  return MPOTerm(convert(ComplexF64,c),[SiteOp(op1,i1)])
end

function MPOTerm(c::Number,
                 op1::String,i1::Int,
                 op2::String,i2::Int)
  return MPOTerm(convert(ComplexF64,c),[SiteOp(op1,i1),SiteOp(op2,i2)])
end

function Base.show(io::IO,
              op::MPOTerm) 
  c = coef(op)
  if c != 1.0+0.0im
    if imag(c) == 0.0
      print(io,"$(real(c)) ")
    elseif real(c) == 0.0
      print(io,"$(imag(c))im ")
    else
      print(io,"($c) ")
    end
  end
  for o in ops(op)
    print(io,"\"$(name(o))\"($(site(o)))")
  end
end

############################
## AutoMPO                 #
############################

struct AutoMPO
  sites::SiteSet
  terms::Vector{MPOTerm}
  AutoMPO(s::SiteSet) = new(s,Vector{MPOTerm}())
end
sites(ampo::AutoMPO) = ampo.sites
terms(ampo::AutoMPO) = ampo.terms

function add!(ampo::AutoMPO,
              op::String, i::Int)
  push!(terms(ampo),MPOTerm(1.0,op,i))
end

function add!(ampo::AutoMPO,
              coef::Number,
              op::String, i::Int)
  push!(terms(ampo),MPOTerm(coef,op,i))
end

function add!(ampo::AutoMPO,
              op1::String, i1::Int,
              op2::String, i2::Int)
  push!(terms(ampo),MPOTerm(1.0,op1,i1,op2,i2))
end

function add!(ampo::AutoMPO,
              coef::Number,
              op1::String, i1::Int,
              op2::String, i2::Int)
  push!(terms(ampo),MPOTerm(coef,op1,i1,op2,i2))
end

function show(io::IO,
              ampo::AutoMPO) 
  println(io,"AutoMPO:")
  for term in terms(ampo)
    println(io,"  $term")
  end
end

##################################
# MatElem (simple sparse matrix) #
##################################

struct MatElem{T}
  row::Int
  col::Int
  val::T
end

function Base.show(io::IO,m::MatElem)
  print(io,"($(m.row),$(m.col),$(m.val))")
end

function toMatrix(els::Vector{MatElem{T}})::Matrix{T} where {T}
  nr = 0
  nc = 0
  for el in els
    nr = max(nr,el.row)
    nc = max(nr,el.col)
  end
  M = zeros(T,nr,nc)
  for el in els
    M[el.row,el.col] = el.val
  end
  return M
end

function posInLink!(linkmap::Vector{OpTerm},
                    op::OpTerm)::Int
  isempty(op) && return -1
  for n=1:length(linkmap)
    (linkmap[n]==op) && return n
  end
  push!(linkmap,op)
  return length(linkmap)
end

function determineValType(terms::Vector{MPOTerm})
  for t in terms
    (!isreal(coef(t))) && return ComplexF64
  end
  return Float64
end

function partitionHTerms(sites::SiteSet,
                         terms::Vector{MPOTerm}
                         ; kwargs...)
  N = length(sites)

  ValType = determineValType(terms)

  bond_coefs = [MatElem{ValType}[] for n=1:N]
  tempMPO = [Set(MatElem{MPOTerm}[]) for n=1:N]

  crosses_bond(t::MPOTerm,n::Int) = (ops(t)[1].site <= n <= ops(t)[end].site)

  rightmap = OpTerm[]
  next_rightmap = OpTerm[]
  
  for n=1:N

    leftmap = OpTerm[]
    for term in terms 
      crosses_bond(term,n) || continue

      left::OpTerm   = filter(t->(t.site < n),ops(term))
      onsite::OpTerm = filter(t->(t.site == n),ops(term))
      right::OpTerm  = filter(t->(t.site > n),ops(term))

      bond_row = -1
      bond_col = -1
      if !isempty(left)
        bond_row = posInLink!(leftmap,left)
        bond_col = posInLink!(rightmap,mult(onsite,right))
        bond_coef = convert(ValType,coef(term))
        push!(bond_coefs[n-1],MatElem(bond_row,bond_col,bond_coef))
      end

      A_row = bond_col
      A_col = posInLink!(next_rightmap,right)
      site_coef = 1.0+0.0im
      if A_row == -1
        site_coef = coef(term)
      end
      el = MatElem(A_row,A_col,MPOTerm(site_coef,onsite))
      push!(tempMPO[n],el)
    end
    rightmap = next_rightmap
    next_rightmap = OpTerm[]

  end

  return bond_coefs,tempMPO
end

function multSiteOps(A::ITensor,
                     B::ITensor)::ITensor
  R = copy(A)
  prime!(R,"Site")
  R *= B
  return mapprime(R,2,1)
end

function computeProd(sites::SiteSet,
                     ops::OpTerm)::ITensor
  i = ops[1].site
  T = op(sites,ops[1].name,i)
  for j=2:length(ops)
    (ops[j].i != i) && error("Mismatch of site number in computeProd")
    opj = op(sites,ops[j].name,i)
    T = multSiteOps(T,opj)
  end
  return T
end

function compressMPO(sites::SiteSet,
                     bond_coefs::Vector{Vector{MatElem{ValType}}},
                     tempMPO::Vector{Set{MatElem{MPOTerm}}}
                     ; kwargs...) where {ValType}

  mindim::Int = get(kwargs,:mindim,1)
  maxdim::Int = get(kwargs,:maxdim,10000)
  cutoff::Float64 = get(kwargs,:cutoff,1E-13)

  N = length(sites)

  links = Dict{Int,Index}()
  links[0] = Index(2,"Link,n=0")

  V_n = Matrix{ValType}(undef,1,1)
  V_npp = Matrix{ValType}(undef,1,1)

  # Constants which define MPO start/end scheme
  rowShift = 2
  colShift = 2
  startState = 2
  endState = 1

  H = MPO(sites)

  for n=1:N
    tdim = 0
    if !isempty(bond_coefs[n])
      M = toMatrix(bond_coefs[n])
      U,S,V = svd(M)
      P = S.^2
      truncate!(P;maxdim=maxdim,cutoff=cutoff,mindim=mindim)
      tdim = length(P)
      nc = size(M,2)
      V_npp = Matrix{ValType}(V[1:nc,1:tdim])
    end
    links[n] = Index(2+tdim,"Link,n=$n")

    finalMPO = Dict{OpTerm,Matrix{ValType}}()

    ll = links[n-1]
    rl = links[n]

    idTerm = [SiteOp("Id",n)]
    finalMPO[idTerm] = zeros(ValType,dim(ll),dim(rl))
    idM = finalMPO[idTerm]
    idM[1,1] = 1.0
    idM[2,2] = 1.0

    for el in tempMPO[n]
      A_row = el.row
      A_col = el.col
      t = el.val
      (abs(coef(t)) < eps()) && continue

      if !haskey(finalMPO,ops(t)) 
        finalMPO[ops(t)] = zeros(ValType,dim(ll),dim(rl))
      end
      M = finalMPO[ops(t)]

      ct = convert(ValType,coef(t))
      if A_row==-1 && A_col==-1 #onsite term
        M[startState,endState] += ct
      elseif A_row==-1 #term starting on site n
        for c=1:size(V_npp,2)
          z = ct*V_npp[A_col,c]
          M[startState,colShift+c] += z
        end
      elseif A_col==-1 #term ending on site n
        for r=1:size(V_n,2)
          z = ct*conj(V_n[A_row,r])
          M[rowShift+r,endState] += z
        end
      else
        for r=1:size(V_n,2),c=1:size(V_npp,2)
          z = ct*conj(V_n[A_row,r])*V_npp[A_col,c]
          M[rowShift+r,colShift+c] += z
        end
      end
    end

    H[n] = ITensor(dag(sites[n]),sites[n]',ll,rl)
    for (op,M) in finalMPO
      T = ITensor(M,ll,rl)
      H[n] += T*computeProd(sites,op)
    end

    V_n = V_npp
  end

  L = ITensor(links[0])
  L[startState] = 1.0

  R = ITensor(links[N])
  R[endState] = 1.0

  H[1] *= L
  H[N] *= R

  #println("\n\nfinalMPO:\n")
  #for n=1:length(H)
  #  @show H[n]
  #end
  
  return H
end

function svdMPO(ampo::AutoMPO; 
                kwargs...)
  bond_coefs,tempMPO = partitionHTerms(sites(ampo),terms(ampo);kwargs...)
  return compressMPO(sites(ampo),bond_coefs,tempMPO;kwargs...)
end

function toMPO(ampo::AutoMPO; 
               kwargs...)::MPO
  return svdMPO(ampo;kwargs...)
end

