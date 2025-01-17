module ITensors

export ITensor, Index

using TensorAlgebra: contract
using ITensorBase:
  ITensor,
  Index,
  addtags,
  commonind,
  commoninds,
  inds,
  plev,
  prime,
  tags,
  uniqueind,
  uniqueinds

# Quirks, decide where or if to define.
using ITensorBase: dag, dim, factorize, hasqns, itensor, onehot

include("SiteTypes/SiteTypes.jl")
using .SiteTypes: SiteTypes
include("LazyApply/LazyApply.jl")
using .LazyApply: LazyApply
include("Ops/Ops.jl")
using .Ops: Ops

# TODO: Used in `ITensorMPS.jl`, define in `BackendSelection.jl`.
struct Algorithm{algname} end
macro Algorithm_str(algname)
  return :(Algorithm{$(Expr(:quote, Symbol(algname)))})
end

# TODO: Used in `ITensorMPS.jl`, decide where or if to define it.
# Maybe define in `TensorAlgebra.jl`.
function outer end

# TODO: Used in `ITensorMPS.jl`, decide where or if to define it.
struct Apply{Args}
  args::Args
end

# TODO: Used in `ITensorMPS.jl`, decide where or if to define it.
# Probably define in `ITensorBase.jl` as a shorthand for
# constructing a set of tags.
macro ts_str(tags) end

# TODO: Used in `ITensorMPS.jl`, decide where or if to define it.
struct OneITensor end

# TODO: Used in `ITensors.SiteTypes`, need to define.
function product end

# TODO: Used in `ITensorMPS.jl`, define in `ITensorBase.jl`.
function noprime end
function removetags end
function replaceprime end
function replacetags end
function setprime end
function settags end
function sim end
function swapprime end

# TODO: Delete these in-place versions, only define
# them in `ITensorMPS.jl`.
function addtags! end
function noprime! end
function prime! end
function removetags! end
function replaceprime! end
function replacetags! end
function setprime! end
function settags! end
function swapprime! end

end
