
const default_warn_itensor_order = 14

const warn_itensor_order =
  Ref{Union{Int, Nothing}}(default_warn_itensor_order)

"""
    get_warn_itensor_order()

Return the threshold for the order of an ITensor above which 
ITensors will emit a warning.

You can set the threshold with the function `set_warn_itensor_order!(N::Int)`.
"""
get_warn_itensor_order() = warn_itensor_order[]

"""
    set_warn_itensor_order!(N::Int)

After this is called, ITensor will warn about ITensor contractions
that result in ITensors above the order `N`.

This function returns the initial warning threshold (what it was
set to before this function was called).

You can get the current threshold with the function `get_warn_itensor_order(N::Int)`. You can reset to the default value with
`reset_warn_itensor_order!()`.
"""
function set_warn_itensor_order!(N::Union{Int, Nothing})
  N_init = get_warn_itensor_order()
  warn_itensor_order[] = N
  return N_init
end

"""
    reset_warn_itensor_order!()

After this is called, ITensor will warn about ITensor contractions
that result in ITensors above the default order 
$default_warn_itensor_order.

This function returns the initial warning threshold (what it was
set to before this function was called).
"""
reset_warn_itensor_order!() =
  set_warn_itensor_order!(default_warn_itensor_order)

"""
    disable_warn_itensor_order!()

After this is called, ITensor will not warn about ITensor
contractions that result in large ITensor orders.

This function returns the initial warning threshold (what it was
set to before this function was called).
"""
disable_warn_itensor_order!() =
  set_warn_itensor_order!(nothing)

const GLOBAL_TIMER = TimerOutput()

