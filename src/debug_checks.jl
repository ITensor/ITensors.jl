
# XXX: rename use_debug_checks
use_debug_checks() = false

macro debug_check(ex)
  quote
    if use_debug_checks()
      $(esc(ex))
    end
  end
end

function enable_debug_checks()
  if !getfield(@__MODULE__, :use_debug_checks)()
    Core.eval(@__MODULE__, :(use_debug_checks() = true))
  end
end

function disable_debug_checks()
  if getfield(@__MODULE__, :use_debug_checks)()
    Core.eval(@__MODULE__, :(use_debug_checks() = false))
  end
end

