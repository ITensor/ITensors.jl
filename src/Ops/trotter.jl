abstract type ExpAlgorithm end

struct Exact <: ExpAlgorithm end

struct Trotter{Order} <: ExpAlgorithm
  nsteps::Int
end
one(::Trotter{Order}) where {Order} = Trotter{Order}(1)

function exp(o::Sum; alg::ExpAlgorithm=Exact())
  return exp(alg, o)
end

function exp(::Exact, o::Sum)
  return Applied(prod, [Applied(exp, o)])
end

function exp_one_step(trotter::Trotter{1}, o::Sum)
  exp_o = Applied(prod, map(exp, only(o.args)))
  return exp_o
end

function exp_one_step(trotter::Trotter{2}, o::Sum)
  exp_o_order_1 = exp_one_step(Trotter{1}(1), o / 2)
  exp_o = exp_o_order_1 * reverse(exp_o_order_1)
  return exp_o
end

function exp(trotter::Trotter, o::Sum)
  expδo = exp_one_step(one(trotter), o / trotter.nsteps)
  return expδo^trotter.nsteps
end
