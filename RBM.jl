hid = 3
vis = 6
W = rand(hid, vis)
sigm(x) = 1./(1 + exp(-x))
function vhv(vvec, W, hbias, vbias)
  hvec = rand(size(hbias)) .< sigm(vvec'*W' + hbias)
  vv = rand(size(vvec)) .< sigm(W'*hvec' + vbias)
  return vv
end

function vh(vvec, W, hbias, vbias)
  hvec = rand(size(hbias)) .< sigm(vvec'*W' + hbias)
  return hvec
end
#main loop
