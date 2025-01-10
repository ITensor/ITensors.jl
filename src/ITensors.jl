module ITensors

using TensorAlgebra: contract
using ITensorBase: ITensor, Index
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
function outer end

# TODO: Used in `ITensorMPS.jl`, decide where or if to define it.
struct Apply{Args}
  args::Args
end

# TODO: Used in `ITensorMPS.jl`, decide where or if to define it.
macro ts_str(tags) end

# TODO: Used in `ITensorMPS.jl`, decide where or if to define it.
struct OneITensor end

# TODO: Used in `ITensorMPS.jl`, define in `ITensorBase.jl`.
function addtags end
function addtags! end
function commonind end
function commoninds end
function dag end
function noprime end
function noprime! end
function prime end
function prime! end
function removetags end
function removetags! end
function replaceprime end
function replaceprime! end
function replacetags end
function replacetags! end
function setprime end
function setprime! end
function settags end
function settags! end
function sim end
function swapprime end
function swapprime! end
function uniqueind end
function uniqueinds end

end
