
struct AutoType end

"""
    auto_parse(::Union{Type,AutoType}, val)

Automatically parse the value into a Julia value.
"""
auto_parse(ValType::Type, val) = parse(ValType, val)

auto_parse(::Type{AutoType}, val) = eval(Meta.parse(val))

auto_parse(::Type{String}, val) = String(strip(val))

"""
    parse_type(valtype, default_type::Type = AutoType)

Parse the type of `valtype`. If `valtype` has a type declaration,
like `parse_type("2::ComplexF64")`, it gets parsed as that type
declared, and returns `(ComplexF64, 2)`.

If `val` doesn't have a type declaration, it gets parsed into
`default_type`, which defaults to `AutoType`, so `parse_type("2")`
returns `(AutoType, 2)`
"""
function parse_type(valtype, default_type::Type=AutoType)
  # Check for a type decleration
  valtype_vec = split(valtype, "::"; limit=2)
  ValType = default_type
  if length(valtype_vec) > 1
    # Type declaration
    ValType = eval(Meta.parse(valtype_vec[2]))
  end
  val = valtype_vec[1]
  return ValType, val
end

"""
    argsdict([args_list::Vector];
             first_arg::Int = 1,
             delim = '=',
             as_symbols::Bool = false,
             default_named_type::Type = ITensors.AutoType,
             save_positional::Bool = true,
             default_positional_type::Type = String,
             prefix::String = "_arg")

Parse the command line arguments such as `julia N=2 X=1e-12`
and put them in a dictionary, where the keys are before
the delimiter and the values are after, so
`Dict("N" => "2", "X" => "1e-12")`.
"""
function argsdict(
  args_list::Vector;
  first_arg::Int=1,
  delim='=',
  as_symbols::Bool=false,
  default_named_type::Type=AutoType,
  save_positional::Bool=true,
  default_positional_type::Type=String,
  prefix::String="_arg",
)
  KeyType = as_symbols ? Symbol : String
  parsed = Dict{KeyType,Any}()
  narg = 1
  for n in first_arg:length(args_list)
    a = args_list[n]

    # Check if it is a command line flag
    if startswith(a, "--")
      flag = a[3:end]
      if flag == "autotype" || flag == "a"
        default_positional_type = AutoType
        default_named_type = AutoType
      elseif flag == "stringtype" || flag == "s"
        default_positional_type = String
        default_named_type = String
      end
      continue
    end

    optval = split(a, delim)
    if length(optval) == 1
      if save_positional
        val = only(optval)
        parsed[KeyType("$prefix$narg")] = auto_parse(
          parse_type(val, default_positional_type)...
        )
        narg += 1
      else
        @warn "Ignoring argument $a since it does not have the delimiter \"$delim\"."
      end
      continue
    elseif length(optval) == 2
      opt, val = optval
    else
      error(
        "Argument $a has more than one delimiter \"$delim\", which is not well defined."
      )
    end
    ValType, key = parse_type(opt, default_named_type)
    key = strip(key)
    ' ' in key && error("Option \"$key\" contains spaces, which is not well defined")
    typedkey = KeyType(key)
    typedval = auto_parse(ValType, val)
    parsed[typedkey] = typedval
  end
  return parsed
end

argsdict(; kwargs...) = argsdict(ARGS; kwargs...)
