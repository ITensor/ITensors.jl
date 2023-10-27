function qr(E::Exposed)
  return qr(unexpose(E))
end

function eigen(E::Exposed)
  return eigen(expose(E))
end

function svd(E::Exposed)
  return svd(expose(E))
end
