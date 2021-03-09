
#
# Warn about the order of the ITensor after contractions
#

const default_warn_order = 14

const warn_order =
  Ref{Union{Int, Nothing}}(default_warn_order)

"""
    ITensors.get_warn_order()

Return the threshold for the order of an ITensor above which 
ITensors will emit a warning.

You can set the threshold with the function `set_warn_order!(N::Int)`.
"""
get_warn_order() = warn_order[]

"""
    ITensors.set_warn_order(N::Int)

After this is called, ITensor will warn about ITensor contractions
that result in ITensors above the order `N`.

This function returns the initial warning threshold (what it was
set to before this function was called).

You can get the current threshold with the function `ITensors.get_warn_order(N::Int)`. You can reset to the default value with
`ITensors.reset_warn_order()`.
"""
function set_warn_order(N::Union{Int, Nothing})
  N_init = get_warn_order()
  warn_order[] = N
  return N_init
end

"""
    ITensors.reset_warn_order()

After this is called, ITensor will warn about ITensor contractions
that result in ITensors above the default order 
$default_warn_order.

This function returns the initial warning threshold (what it was
set to before this function was called).
"""
reset_warn_order() = set_warn_order(default_warn_order)

"""
    ITensors.disable_warn_order()

After this is called, ITensor will not warn about ITensor
contractions that result in large ITensor orders.

This function returns the initial warning threshold (what it was
set to before this function was called).
"""
disable_warn_order() = set_warn_order(nothing)

"""
    @disable_warn_order

Disable warning about the ITensor order in a block of code.

# Examples
```julia
A = ITensor(IndexSet(_ -> Index(1), Order(8)))
B = ITensor(IndexSet(_ -> Index(1), Order(8)))
A * B
@disable_warn_order A * B
@reset_warn_order A * B
@set_warn_order 17 A * B
@set_warn_order 12 A * B
```
"""
macro disable_warn_order(block)
  quote
    local old_order = disable_warn_order()
    r = $(esc(block))
    set_warn_order(old_order)
    r
  end
end

"""
    @set_warn_order

Temporarily set the order threshold for warning about the ITensor
order in a block of code.

# Examples
```julia
@set_warn_order 12 A * B

@set_warn_order 15 begin
  C = A * B
  E = C * D
end
```
"""
macro set_warn_order(new_order, block)
  quote
    local old_order = set_warn_order($(esc(new_order)))
    r = $(esc(block))
    set_warn_order(old_order)
    r
  end
end

"""
    @reset_warn_order

Temporarily sets the order threshold for warning about the ITensor
order in a block of code to the default value $default_warn_order.

# Examples
```julia
@reset_warn_order A * B
```
"""
macro reset_warn_order(block)
  quote
    local old_order = reset_warn_order()
    r = $(esc(block))
    set_warn_order(old_order)
    r
  end
end

#
# A global timer used with TimerOutputs.jl
#

using NDTensors: timer

#
# Get the current number of BLAS threads
# For VERSION >= v"1.6" this will become
# using LinearAlgebra; BLAS.get_num_threads()
#

using NDTensors: blas_get_num_threads

#
# Block sparse multithreading
#

"""
$(NDTensors.enable_threaded_blocksparse_docstring(@__MODULE__))
"""
using_threaded_blocksparse() = NDTensors._using_threaded_blocksparse[]

"""
$(NDTensors.enable_threaded_blocksparse_docstring(@__MODULE__))
"""
enable_threaded_blocksparse() = NDTensors._enable_threaded_blocksparse()

"""
$(NDTensors.enable_threaded_blocksparse_docstring(@__MODULE__))
"""
disable_threaded_blocksparse() = NDTensors._disable_threaded_blocksparse()

#
# Turn enable or disable combining QN ITensors before contracting
#

const _using_combine_contract = Ref(false)

using_combine_contract() = _using_combine_contract[]

function enable_combine_contract()
  _using_combine_contract[] = true
  return nothing
end

function disable_combine_contract()
  _using_combine_contract[] = false
  return nothing
end

#
# Turn debug checks on and off
#

using_debug_checks() = false

macro debug_check(ex)
  quote
    if using_debug_checks()
      $(esc(ex))
    end
  end
end

function enable_debug_checks()
  if !getfield(@__MODULE__, :using_debug_checks)()
    Core.eval(@__MODULE__, :(using_debug_checks() = true))
  end
end

function disable_debug_checks()
  if getfield(@__MODULE__, :using_debug_checks)()
    Core.eval(@__MODULE__, :(using_debug_checks() = false))
  end
end

#
# Turn contraction sequence optimizations on and off
#

const _using_contraction_sequence_optimization = Ref(false)

using_contraction_sequence_optimization() = _using_contraction_sequence_optimization[]

function enable_contraction_sequence_optimization()
  _using_contraction_sequence_optimization[] = true
  return nothing
end

function disable_contraction_sequence_optimization()
  _using_contraction_sequence_optimization[] = false
  return nothing
end

