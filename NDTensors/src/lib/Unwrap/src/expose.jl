struct Exposed{Unwrapped,Object}
  object::Object
end

expose(object) = Exposed{unwrap_array_type(object),typeof(object)}(object)

unexpose(E::Exposed) = E.object
