"""
    TBLIS <: ContractAlgorithm

Algorithm tag for the TBLIS dense contraction backend. Selected via
[`with_tblis`](@ref). The `contract!` implementation and the positive
`is_applicable` overload live in `NDTensorsTBLISExt`, gated on
`TBLIS.jl` being loaded; the baseline reject-everything overload below
ensures that without the extension loaded, `with_tblis` scopes fall
through to the default rather than picking `TBLIS()` and then hitting a
`MethodError` for the absent `contract!(::TBLIS, …)` method.

TBLIS has bad/slow complex support, so the extension's `is_applicable`
restricts this algorithm to dense inputs with matching `Float32` /
`Float64` element types; other contractions inside a `with_tblis` scope
fall through to the default.
"""
struct TBLIS <: ContractAlgorithm end

# Reject everything by default. The positive overload accepting
# matching real-BlasReal `DenseTensor` × `DenseTensor` lives in
# `NDTensorsTBLISExt`, so a `with_tblis` scope without `TBLIS.jl`
# loaded falls through here to `default_contract_algorithm`.
is_applicable(::TBLIS, ::Type, ::Type) = false

"""
    with_tblis(f, enable::Bool = true)

Run `f()` with the TBLIS dense contraction backend preferred for any
contractions inside the block. When `enable = false`, runs `f()`
unchanged.

Equivalent to `with_contract_algorithm(f, TBLIS())`. Requires
`NDTensorsTBLISExt` to be loaded (i.e. `using TBLIS`); without the
extension loaded the scope is inert (no `TBLIS` `is_applicable` case
returns `true`, so all contractions inside fall through to the
default). Applies only to real-eltype `DenseTensor` × `DenseTensor`
contractions where both inputs share the same `Float32` or `Float64`
eltype; other contractions inside the scope fall through to the
default.
"""
function with_tblis(f, enable::Bool = true)
    return enable ? with_contract_algorithm(f, TBLIS()) : f()
end
