struct Expose{Unwraped,Object}
  object::Object
end

expose(object) = Expose{unwrap_type(object),typeof(object)}(object)
