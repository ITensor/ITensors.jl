export SiteOp,
       OpTerm,
       MPOTerm,
       AutoMPO,
       add!,
       toMPO

###########################
# SiteOp                  # 
###########################

struct SiteOp
  name::String
  site::Int
end
name(s::SiteOp) = s.name
site(s::SiteOp) = s.site
show(io::IO,s::SiteOp) = print(io,"\"$(name(s))\"($(site(s)))")

###########################
# OpTerm                  # 
###########################

const OpTerm = Vector{SiteOp}
#struct OpTerm
#  ops::Vector{SiteOp}
#end
#ops(t::OpTerm) = t.ops
#
#function ==(t1::OpTerm,
#            t2::OpTerm)::Bool
#  t1l = length(t1)
#  (t1l != length(t2)) && return false
#  for n=1:t1l
#    (t1[n] != t2[n]) && return false
#  end
#  return true
#end

###########################
# MPOTerm                 # 
###########################

const Coef = ComplexF64

struct MPOTerm
  coef::Coef
  ops::Vector{SiteOp}
end
coef(op::MPOTerm) = op.coef
ops(op::MPOTerm) = op.ops
length(op::MPOTerm) = length(op.ops)

MPOTerm(c::Number,op1::String,i1::Int) = MPOTerm(convert(Coef,c),[SiteOp(op,i)])

function MPOTerm(c::Number,
                op1::String,i1::Int,
                op2::String,i2::Int)
  return MPOTerm(convert(Coef,c),[SiteOp(op1,i1),SiteOp(op2,i2)])
end

function show(io::IO,
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
    print(io,"\"$(name(o))\"($(site(o))) ")
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

###############
###############

struct MatElem
  row::Int
  col::Int
  val::ComplexF64
end

mutable struct MPOBlock
  leftmap::Vector{OpTerm}
  rightmap::Vector{OpTerm}
  matels::Vector{MatElem}
end
MPOBlock() = MPOBlock(Vector{OpTerm}(),Vector{OpTerm}(),Vector{MatElem}())


function posInLink!(linkmap::Vector{OpTerm},
                    op::OpTerm)::Int
  ll = length(linkmap)
  for n=1:ll
    (linkmap[n]==op) && return n
  end
  push!(linkmap,op)
  return ll
end

struct MPOMatElem
  row::Int
  col::Int
  val::MPOTerm
end

function pushUnique!(vec::Vector{T},
                     x::T) where T
  for n=1:length(vec)
    (vec[n]==x) && return
  end
  push!(vec,x)
end


function partitionHTerms(sites::SiteSet,
                         terms::Vector{MPOTerm},
                         ; kwargs...)
  N = length(sites)

  blocks = fill(MPOBlock(),N)
  tempMPO = fill(Vector{MPOMatElem}(),N)

  for term in terms 
    #@show term
    for n=ops(term)[1].site:ops(term)[end].site
      left::OpTerm   = filter(t->(t.site < n),ops(term))
      onsite::OpTerm = filter(t->(t.site == n),ops(term))
      right::OpTerm  = filter(t->(t.site > n),ops(term))

      b_row = -1
      b_col = -1

      #println("  n = $n")
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
          b_row = posInLink!(leftblock.rightmap,mult(onsite,right))
          b_col = posInLink!(rightblock.rightmap,right)
        end
        l = posInLink!(leftblock.leftmap,left)
        push!(leftblock.matels,MatElem(l,b_row,coef(term)))
      end

      el = MPOMatElem(b_row,b_col,MPOTerm(coef(term),onsite))
      pushUnique!(tempMPO[n],el)
    end
  end

  return blocks,tempMPO
end

#function compressMPO(sites::SiteSet,
#                     qbs::Vector{QNBlock},
#                     tempMPO::Vector{IQMatEls}
#                     ; kwargs...)
#                     #::Tuple{Vector{MPOPiece},IndexSet}
#end
#
#function constructMPOTensors(sites::SiteSet,
#                             finalMPO::Vector{MPOPiece},
#                             links::IndexSet
#                             ; kwargs...)::MPO
#end


function svdMPO(am::AutoMPO; kwargs...)
  blocks,tempMPO = partitionHTerms(sites(am),terms(am);kwargs...)
  #finalMPO,links = compressMPO(sites(am),blocks,tempMPO;kwargs...)
  #mpo = constructMPOTensors(sites(am),finalMPO,links;kwargs...)
  #return mpo
  return MPO()
end

function toMPO(am::AutoMPO; kwargs...)::MPO
  return svdMPO(am;kwargs...)
end

