abstract type ExpAlgorithm end

struct Exact <: ExpAlgorithm end

struct Trotter{Order} <: ExpAlgorithm
  nsteps::Int
end
one(::Trotter{Order}) where {Order} = Trotter{Order}(1)

function exp(o::∑; alg::ExpAlgorithm=Exact())
  return exp(alg, o)
end

function exp(::Exact, o::∑)
  return ∏([Applied(exp, o)])
end

function exp_one_step(trotter::Trotter{1}, o::∑)
  # TODO: Customize broadcast of `∏`.
  exp_o = ∏(exp.(o))
  return exp_o
end

function exp_one_step(trotter::Trotter{N}, o::∑) where {N}
  exp_o_order_1 = exp_one_step(Trotter{Int(N / 2)}(1), o / 2)
  exp_o = exp_o_order_1 * reverse(exp_o_order_1)
  return exp_o
end

function exp(trotter::Trotter, o::∑)
  expδo = exp_one_step(one(trotter), o / trotter.nsteps)
  return expδo^trotter.nsteps
end
