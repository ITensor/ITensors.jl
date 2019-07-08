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

##################################
# MPOBlock                       #
##################################

mutable struct MPOBlock{T}
  leftmap::Vector{OpTerm}
  rightmap::Vector{OpTerm}
  mat_els::Vector{MatElem{T}}

  function MPOBlock{T}() where {T}
    return new{T}(Vector{OpTerm}(),
                  Vector{OpTerm}(),
                  Vector{MatElem{T}}())
  end
end

function Base.show(io::IO,block::MPOBlock)
  print(io,"\nleftmap: ")
  println(io,block.leftmap)
  print(io,"rightmap: ")
  println(io,block.rightmap)
  print(io,"mat_els: [")
  for el in block.mat_els
    print(io,el)
  end
  print(io,"]")
end

function posInLink!(linkmap::Vector{OpTerm},
                    op::OpTerm)::Int
  for n=1:length(linkmap)
    (linkmap[n]==op) && return n
  end
  push!(linkmap,op)
  return length(linkmap)
end

#
# Improvement ideas:
# - take leftmap & rightmap out of blocks
#   maybe just a single "bondmap", no left/right distinction
# - do a "dry run" making left/right/bondmap to determine h_{ab}
#   (blocks) and tempMPO matrix sizes, then use dense matrices for these
# - rename "blocks" to something like "coefMats"
#   (it is just the h^n_{a,b} matrix from Nakatani)
# - make actual MPO straightaway in compressMPO? or still have
#   separate constructMPOTensors function?
#

function partitionHTerms(sites::SiteSet,
                         terms::Vector{MPOTerm},
                         val_type::Union{Type{Float64},Type{ComplexF64}}
                         ; kwargs...)
  N = length(sites)

  blocks = [MPOBlock{val_type}() for n=1:N]
  tempMPO = [Set(MatElem{MPOTerm}[]) for n=1:N]

  for term in terms 
    for n=ops(term)[1].site:ops(term)[end].site
      println("n = $n")
      left::OpTerm   = filter(t->(t.site < n),ops(term))
      onsite::OpTerm = filter(t->(t.site == n),ops(term))
      right::OpTerm  = filter(t->(t.site > n),ops(term))

      b_row = -1
      b_col = -1

      rightblock = blocks[n]

      if isempty(left)
        if !isempty(right)
          b_col = posInLink!(rightblock.rightmap,right)
        end
      else
        leftblock = blocks[n-1]
        if isempty(right)
          b_row = posInLink!(leftblock.rightmap,onsite)
        else
          @show mult(onsite,right)
          b_row = posInLink!(leftblock.rightmap,mult(onsite,right))
          b_col = posInLink!(rightblock.rightmap,right)
        end
        l = posInLink!(leftblock.leftmap,left)
        c = convert(val_type,coef(term))
        push!(leftblock.mat_els,MatElem(l,b_row,c))
      end

      c::ComplexF64 = (b_row == -1) ? coef(term) : ComplexF64(1.,0.)
      el = MatElem(b_row,b_col,MPOTerm(c,onsite))
      push!(tempMPO[n],el)
    end
  end
  for n=1:N
    println("n = $n")
    @show blocks[n]
    println()
    @show tempMPO[n]
    println()
  end
  return blocks,tempMPO
end

function compressMPO(sites::SiteSet,
                     qbs::Vector{MPOBlock{val_type}},
                     tempMPO::Vector{Set{MatElem{MPOTerm}}}
                     ; kwargs...) where {val_type}

  mindim::Int = get(kwargs,:mindim,1)
  maxdim::Int = get(kwargs,:maxdim,10000)
  cutoff::Float64 = get(kwargs,:cutoff,1E-13)

  N = length(sites)

  finalMPO = [Dict{OpTerm,Matrix{val_type}}() for n=1:N]
  links = Dict{Int,Index}()
  links[0] = Index(2,"Link,n=0")

  V_n = Matrix{val_type}(undef,1,1)
  V_npp = Matrix{val_type}(undef,1,1)

  # Constants which define MPO start/end scheme
  rowShift = 2
  colShift = 2
  startState = 2
  endState = 1

  for n=1:N
    println("\n\nn = $n\n~~~~~~~~~~~~~~~~\n\n")
    @show qbs[n]
    @show qbs[n].mat_els
    tdim = 0
    if !isempty(qbs[n].mat_els)
      M = toMatrix(qbs[n].mat_els)
      @show size(M)
      U,S,V = svd(M)
      P = S.^2
      truncate!(P;maxdim=maxdim,cutoff=cutoff,mindim=mindim)
      tdim = length(P)
      nc = size(M,2)
      V_npp = Matrix{val_type}(V[1:nc,1:tdim])
    end
    links[n] = Index(2+tdim,"Link,n=$n")

    fm = finalMPO[n]
    ll = links[n-1]
    rl = links[n]

    idTerm = [SiteOp("Id",n)]
    fm[idTerm] = zeros(val_type,dim(ll),dim(rl))
    Mid = fm[idTerm]
    Mid[1,1] = 1.0
    Mid[2,2] = 1.0

    for el in tempMPO[n]
      b_row = el.row
      b_col = el.col
      t = el.val
      (abs(coef(t)) < eps()) && continue
      if !haskey(fm,ops(t)) 
        fm[ops(t)] = zeros(val_type,dim(ll),dim(rl))
      end
      M = fm[ops(t)]
      ct = convert(val_type,coef(t))
      @show t
      @show b_row
      @show b_col
      @show size(V_n)
      @show size(V_npp)
      if b_row==-1 && b_col==-1 #onsite term
        M[startState,endState] += ct
      elseif b_row==-1 #term starting on site n
        for c=1:size(V_npp,2)
          z = ct*V_npp[b_col,c]
          M[startState,colShift+c] += z
        end
      elseif b_col==-1 #term ending on site n
        for r=1:size(V_n,2)
          z = ct*conj(V_n[b_row,r])
          M[rowShift+r,endState] += z
        end
      else
        for r=1:size(V_n,2),c=1:size(V_npp,2)
          z = ct*conj(V_n[b_row,r])*V_npp[b_col,c]
          M[rowShift+r,colShift+c] += z
        end
      end
    end

    V_n = V_npp
  end

  println("\n\nfinalMPO:\n")
  for (n,fm) in enumerate(finalMPO)
    @show n
    @show fm
  end
  
  return finalMPO,links
end

 
#function constructMPOTensors(sites::SiteSet,
#                             finalMPO::Vector{MPOPiece},
#                             links::IndexSet
#                             ; kwargs...)::MPO
#end

function determineValType(ampo::AutoMPO)
  for t in terms(ampo) 
    (!isreal(coef(t))) && return ComplexF64
  end
  return Float64
end

function svdMPO(ampo::AutoMPO; kwargs...)
  val_type = determineValType(ampo)
  blocks,tempMPO = partitionHTerms(sites(ampo),terms(ampo),val_type;kwargs...)
  finalMPO,links = compressMPO(sites(ampo),blocks,tempMPO;kwargs...)
  #mpo = constructMPOTensors(sites(ampo),finalMPO,links;kwargs...)
  #return mpo
  return MPO()
end

function toMPO(ampo::AutoMPO; kwargs...)::MPO
  return svdMPO(ampo;kwargs...)
end

