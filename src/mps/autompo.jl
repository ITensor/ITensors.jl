
###########################
# SiteTerm                # 
###########################

struct SiteOp
  name::String
  i::Int
end

const SiteOpProd = Vector{SiteOp}

###########################
# HTerm                   # 
###########################

const Coef = ComplexF64

struct HTerm
  coef::Coef
  ops::SiteOpProd
end

coef(ht::HTerm) = ht.coef
ops(ht::HTerm) = ht.ops

HTerm(c::Number,op::String,i::Int) = HTerm(convert(Coef,c),[SiteOp(op,i)])

function HTerm(c::Number,
               op1::String,i1::Int,
               op2::String,i2::Int)
  return HTerm(convert(Coef,c),[SiteOp(op1,i1),SiteOp(op2,i2)])
end

function show(io::IO,
              ht::HTerm) 
  c = coef(ht)
  if c != 1.0+0.0im
    if imag(c) == 0.0
      print(io,"$(real(c))*")
    elseif real(c) == 0.0
      print(io,"$(imag(c))im*")
    else
      print(io,"($c)*")
    end
  end
  for op in ops(ht)
    print(io,"\"$(op.name)\"($(op.i)) ")
  end
end

###########################
# AutoMPO                 #
###########################

struct AutoMPO
  terms::Vector{HTerm}
  AutoMPO() = new(Vector{HTerm}())
end

terms(ampo::AutoMPO) = ampo.terms

function add!(ampo::AutoMPO,
              op::String, i::Int)
  push!(terms(ampo),HTerm(1.0,op,i))
end

function add!(ampo::AutoMPO,
              coef::Number,
              op::String, i::Int)
  push!(terms(ampo),HTerm(coef,op,i))
end

function add!(ampo::AutoMPO,
              op1::String, i1::Int,
              op2::String, i2::Int)
  push!(terms(ampo),HTerm(1.0,op1,i1,op2,i2))
end

function add!(ampo::AutoMPO,
              coef::Number,
              op1::String, i1::Int,
              op2::String, i2::Int)
  push!(terms(ampo),HTerm(coef,op1,i1,op2,i2))
end

function show(io::IO,
              ampo::AutoMPO) 
  println(io,"AutoMPO:")
  for term in terms(ampo)
    println(io,"  $term")
  end
end
