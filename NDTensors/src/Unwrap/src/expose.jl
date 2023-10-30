struct Exposed{Unwraped,Object}
  object::Object
end

expose(object) = Exposed{unwrap_type(object),typeof(object)}(object)

unexpose(E::Exposed) = E.object
