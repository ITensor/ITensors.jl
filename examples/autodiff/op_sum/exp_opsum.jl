using ITensors
using ChainRulesCore

using ITensors: ExpAlgorithm, sites, MPOTerm, OpTerm, SiteOp, coef, ops, sites
using ITensors.LazyApply: Applied, Exp

import Base: exp, *, /, ^
import ITensors: op
import ITensors.LazyApply: Exp

Exp(arg) = Applied(exp, (arg,))

exp(o::MPOTerm) = Exp(o)

function (o::Vector{Exp{MPOTerm}} ^ n::Int)
  for _ in 1:(n - 1)
    o = [o; o]
  end
  return o
end
function exp_one_step(trotter::Trotter{1}, o::OpSum)
  exp_o = [exp(oₙ) for oₙ in o]
  return exp_o
end

function exp_one_step(trotter::Trotter{N}, o::OpSum) where {N}
  exp_o_order_1 = exp_one_step(Trotter{Int(N / 2)}(1), o / 2)
  exp_o = exp_o_order_1 * reverse(exp_o_order_1)
  return exp_o
end

function exp(trotter::Trotter, o::OpSum)
  expδo = exp_one_step(one(trotter), o / trotter.nsteps)
  return expδo^trotter.nsteps
end

function exp(o::OpSum; alg::ExpAlgorithm=Trotter{1}(1))
  return exp(alg, o)
end

function op(o::Exp{MPOTerm}, s::Vector{<:Index})
  return exp(op(o.args[1], s))
end

function op(o::MPOTerm, s::Vector{<:Index})
  return o.coef * op(o.ops, s)
end

function _replace_site(s::Int, r::Vector{Pair{Int,Int}})
  return Dict(r)[s]
end

function replace_sites(o::SiteOp, r::Dict{Int,Int})
  new_site = map(s -> r[s], o.site)
  return SiteOp(o.name, new_site, o.params)
end

function support(o::Vector{SiteOp})
  return sort(unique(Iterators.flatten(sites.(o))))
end

function op(o::Vector{<:SiteOp}, s::Vector{<:Index})
  support_o = support(o)
  new_support = collect(1:length(support_o))
  new_s = s[support_o]
  r = @ignore_derivatives Dict(support_o .=> new_support)
  new_o = [replace_sites(oₙ, r) for oₙ in o]
  new_os = OpSum([MPOTerm(1.0, new_o)])
  return prod(MPO(new_os, new_s))
end
