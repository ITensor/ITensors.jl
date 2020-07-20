
parse_args(; first_arg::Int = 1,
             delimiter = '=') =
  parse_args(ARGS; first_arg = first_arg, delimiter = delimiter)

"""
    parse_args([args_list::Vector];
               first_arg::Int = 1, delimiter = '=')

Parse the command line arguments such as `julia N=2 X=1e-12`
and put the in a dictionary, where the keys are before
the delimiter and the values are after, so `Dict(:N => "2", :X => "1e-12")`.
"""
function parse_args(args_list::Vector; first_arg::Int = 1,
                                       delimiter = '=')
  parsed = Dict{Symbol, String}()
  for n in first_arg:length(args_list)
    a = args_list[n]
    opt, arg = split(a, delimiter)
    parsed[Symbol(opt)] = arg
  end
  return parsed
end

