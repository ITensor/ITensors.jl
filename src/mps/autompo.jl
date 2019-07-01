export SiteOp,
       OpProd,
       AutoMPO,
       add!

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
# OpProd                  # 
###########################

const Coef = ComplexF64

struct OpProd
  coef::Coef
  ops::Vector{SiteOp}
end
coef(op::OpProd) = op.coef
ops(op::OpProd) = op.ops

OpProd(c::Number,op1::String,i1::Int) = OpProd(convert(Coef,c),[SiteOp(op,i)])

function OpProd(c::Number,
                op1::String,i1::Int,
                op2::String,i2::Int)
  return OpProd(convert(Coef,c),[SiteOp(op1,i1),SiteOp(op2,i2)])
end

function show(io::IO,
              op::OpProd) 
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
  terms::Vector{OpProd}
  AutoMPO(s::SiteSet) = new(s,Vector{OpProd}())
end
sites(ampo::AutoMPO) = ampo.sites
terms(ampo::AutoMPO) = ampo.terms

function add!(ampo::AutoMPO,
              op::String, i::Int)
  push!(terms(ampo),OpProd(1.0,op,i))
end

function add!(ampo::AutoMPO,
              coef::Number,
              op::String, i::Int)
  push!(terms(ampo),OpProd(coef,op,i))
end

function add!(ampo::AutoMPO,
              op1::String, i1::Int,
              op2::String, i2::Int)
  push!(terms(ampo),OpProd(1.0,op1,i1,op2,i2))
end

function add!(ampo::AutoMPO,
              coef::Number,
              op1::String, i1::Int,
              op2::String, i2::Int)
  push!(terms(ampo),OpProd(coef,op1,i1,op2,i2))
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

struct MatElem{T}
  val::T
  ind::Tuple{Int,Int}
end


function partitionHTerms(sites::SiteSet,
                         terms::Vector{OpProd},
                         ; kwargs...)
                         #::Tuple{Vector{QNBlock},Vector{IQMatEls}}

  qbs = Vector{QNBlock}()
  tempMPO = Vector{IQMatEls}()
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


function svdMPO(am::AutoMPO; kwargs...)::MPO
  #qbs,tempMPO = partitionHTerms(sites(am),terms(am);kwargs...)
  #finalMPO,links = compressMPO(sites(am),qbs,tempMPO;kwargs...)
  #mpo = constructMPOTensors(sites(am),finalMPO,links;kwargs...)
  #return mpo
end




