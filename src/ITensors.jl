module ITensors

export ITensor, Index

using TensorAlgebra: contract
using ITensorBase:
  ITensor,
  Index,
  commonind,
  commoninds,
  gettag,
  inds,
  mapinds,
  noprime,
  plev,
  prime,
  replaceinds,
  settag,
  sim,
  tags,
  uniqueind,
  uniqueinds,
  unsettag

# Quirks, decide where or if to define.
using ITensorBase: dag, dim, factorize, hasqns, onehot

# TODO: Used in `ITensorMPS.jl`, define in `BackendSelection.jl` or `AlgorithmSelection.jl`.
struct Algorithm{algname} end
macro Algorithm_str(algname)
  return :(Algorithm{$(Expr(:quote, Symbol(algname)))})
end

# TODO: Used in `ITensorMPS.jl`, decide where or if to define it.
# Probably define in `ITensorBase.jl` as a shorthand for
# constructing a set of tags.
macro ts_str(tags) end

# TODO: Used in `ITensorMPS.jl`, define in `ITensorBase.jl`.
function replaceprime end
function setprime end
function swapprime end

# TODO: Update tag functions for tagdict.
function removetags end
function replacetags end
function settags end

end
