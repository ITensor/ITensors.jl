module ITensors

export ITensor, Index

# TODO: Delete this and require loading instead?
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

# TODO: Used in `ITensorMPS.jl`, define in `ITensorBase.jl`.
function replaceprime end
function setprime end
function swapprime end

# TODO: Update tag functions for tagdict.
function removetags end
function replacetags end
function settags end

end
