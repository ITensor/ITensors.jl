using Preferences

function allow_ndtensorcuda(allow::Bool)
  return @set_preferences!("allow_ndtensorcuda" => allow)
end

const allow_ndtensorcuda() = @load_preference("allow_ndtensorcuda", false)
