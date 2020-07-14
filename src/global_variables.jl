
const default_warn_order = 14

const warn_order =
  Ref{Union{Int, Nothing}}(default_warn_order)

"""
    get_warn_order()

Return the threshold for the order of an ITensor above which 
ITensors will emit a warning.

You can set the threshold with the function `set_warn_order!(N::Int)`.
"""
get_warn_order() = warn_order[]

"""
    set_warn_order!(N::Int)

After this is called, ITensor will warn about ITensor contractions
that result in ITensors above the order `N`.

This function returns the initial warning threshold (what it was
set to before this function was called).

You can get the current threshold with the function `get_warn_order(N::Int)`. You can reset to the default value with
`reset_warn_order!()`.
"""
function set_warn_order!(N::Union{Int, Nothing})
  N_init = get_warn_order()
  warn_order[] = N
  return N_init
end

"""
    reset_warn_order!()

After this is called, ITensor will warn about ITensor contractions
that result in ITensors above the default order 
$default_warn_order.

This function returns the initial warning threshold (what it was
set to before this function was called).
"""
reset_warn_order!() =
  set_warn_order!(default_warn_order)

"""
    disable_warn_order!()

After this is called, ITensor will not warn about ITensor
contractions that result in large ITensor orders.

This function returns the initial warning threshold (what it was
set to before this function was called).
"""
disable_warn_order!() =
  set_warn_order!(nothing)

"""
    @disable_warn_order

Disable warning about the ITensor order in a block of code.

# Examples
```julia
@disable_warn_order begin
  A * B
end
```
"""
macro disable_warn_order(block)
  quote
    old_order = disable_warn_order!()
    r = $(esc(block))
    set_warn_order!(old_order)
    return r
  end
end

"""
    @set_warn_order

Temporarily set the order threshold for warning about the ITensor
order in a block of code.

# Examples
```julia
@set_warn_order 12 begin
  A * B
end
```
"""
macro set_warn_order(new_order, block)
  quote
    old_order = set_warn_order!($(esc(new_order)))
    r = $(esc(block))
    set_warn_order!(old_order)
    return r
  end
end

"""
    @reset_warn_order

Temporarily reset the order threshold for warning about the ITensor
order in a block of code to the default value $default_warn_order.

# Examples
```julia
@reset_warn_order begin
  A * B
end
```
"""
macro reset_warn_order(block)
  quote
    old_order = reset_warn_order!()
    r = $(esc(block))
    set_warn_order!(old_order)
    return r
  end
end

const GLOBAL_TIMER = TimerOutput()

