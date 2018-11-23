using ForwardDiff
function exlin(x)
  return x >= 0 ? x : exp(x)-1
end
function logsig(x)
  return 1 / (1 + exp(-x))
end

function fwprop(inputs, trans, t)
  IH, HH, HO, Hinit, Hbias, Obias = trans
  n = size(inputs)[2]
  outs = Array{Number}((size(Obias)[1], n + t))
  hids = Array{Number}((size(Hbias)[1], n + t))

  hids[:, 1] = exlin.(HH*Hinit + IH*inputs[:, 1] + Hbias)
  outs[:, 1] = exlin.(HO*hids[:, 1] + Obias)
  for i in 2:n #i=1 gets unrolled
    hids[:, i] = exlin.(HH*hids[:, i-1] + IH*inputs[:, i] + Hbias)
    outs[:, i] = exlin.(HO*hids[:, i] + Obias)
  end
  for i in n+1:n+t
    hids[:, i] = exlin.(HH*hids[:, i-1] + Hbias)
    outs[:, i] = exlin.(HO*hids[:, i] + Obias)
  end
  return outs
end

inp = 2
hid = 10
out = 1
#init
IH = randn(hid, inp)/sqrt(hid*inp + hid*hid)
HH = randn(hid, hid)/sqrt(hid*hid + hid*hid)
#HH = HH/max(abs(eig(HH)[1])...)
HO = randn(out, hid)/sqrt(hid*hid)
#init states
I = zeros(inp) # obviously this has to come from data
H = zeros(hid) # these are gonna be learned.
Hbias = zeros(hid)
Obias = zeros(out) #These biases as well...
Hinit = zeros(hid)
O = zeros(out)
state0 = [I, H, O]
coefs = [IH, HH, HO, Hinit, Hbias, Obias]
inps = []
outs = []


#println(inps)
#println(outs)
#inps = [[0 0 1; 0 0 0],[1 1 0; 0 1 0], [0 1 1; 1 0 1], [0 0 1; 1 1 1], [1 1 1 1 1; 1 0 0 0 0], [1 0 0 0 0 0; 0 1 0 0 0 0]]
#outs = [[0 0 1 0], [1 0 1 0], [1 1 0 1], [1 1 0 1], [0 0 0 0 0 1], [1 1 0 0 0 0 0]]
scrunch(A, B, C, D, E, F) = [A[:];B[:];C[:];D[:];E[:];F[:]]
unscrunch(A) = reshape(A[1:hid*inp], size(IH)), reshape(A[hid*inp + 1:hid*inp + hid*hid], size(HH)), reshape(A[hid*inp + hid*hid + 1:hid*inp + hid*hid + hid*out], size(HO)), A[hid*inp + hid*hid + hid*out + 1:hid*inp + hid*hid + hid*out + hid], A[hid*inp + hid*hid + hid*out + hid + 1:hid*inp + hid*hid + hid*out + hid + hid], A[hid*inp + hid*hid + hid*out + hid + hid + 1: hid*inp + hid*hid + hid*out + hid + hid + out]
#totalcost(A, B, C, D, E, F) = sum((fwprop(inps[1], [A, B, C, D, E, F], 1) - outs[1]).^2)
#println(totalcost(IH, HH, HO, Hinit, Hbias, Obias))
#A -> totalcost(reshape(A, size(IH)), HH, HO, Hinit, Hbias, Obias)
#B -> totalcost(IH, reshape(B, size(HH)), HO, Hinit, Hbias, Obias)
#C -> totalcost(IH, HH, reshape(C, size(HO)), Hinit, Hbias, Obias)
#D -> totalcost(IH, HH, HO, D, Hbias, Obias)
#E -> totalcost(IH, HH, HO, Hinit, E, Obias)
#F -> totalcost(IH, HH, HO, Hinit, Hbias, F)
#for i in 1:0
#delIH = reshape(ForwardDiff.gradient(A -> totalcost(reshape(A, size(IH)), HH, HO, Hinit, Hbias, Obias), IH[:]), size(IH))
#delHH = reshape(ForwardDiff.gradient(B -> totalcost(IH, reshape(B, size(HH)), HO, Hinit, Hbias, Obias) , HH[:]), size(HH))
#delHO = reshape(ForwardDiff.gradient(C -> totalcost(IH, HH, reshape(C, size(HO)), Hinit, Hbias, Obias), HO[:]), size(HO))
#delHinit = ForwardDiff.gradient(D -> totalcost(IH, HH, HO, D, Hbias, Obias), Hinit)
#delHbias = ForwardDiff.gradient(E -> totalcost(IH, HH, HO, Hinit, E, Obias), Hbias)
#delObias = ForwardDiff.gradient(F -> totalcost(IH, HH, HO, Hinit, Hbias, F), Obias)
#IH -= lam*delIH
#HH -= lam*delHH
#HO -= lam*delHO
#Hinit -= lam*delHinit
#Hbias -= lam*delHbias
#Obias -= lam*delObias
#end
#println(totalcost(IH,HH,HO,Hinit,Hbias,Obias))
coefss = scrunch(IH, HH, HO, Hinit, Hbias, Obias)
#lam = 0.08
#for k in 1:0
#for i in 1:100
#delCoefss = ForwardDiff.gradient(tcurl, coefss)
#coefss -= lam*delCoefss
#end
#println(tcurl(coefss))
#end
for i in 0:3, j in 0:3
push!(inps,[digits(i,2,2) digits(j,2,2)]')
push!(outs,digits(i+j, 2, 3)')
end
for i in 0:7, j in 0:7
push!(inps,[digits(i,2,3) digits(j,2,3)]')
push!(outs,digits(i+j, 2, 4)')
end
#for i in 0:15, j in 0:15
#push!(inps,[digits(i,2,4) digits(j,2,4)]')
#push!(outs,digits(i+j, 2, 5)')
#end
#for i in 0:127, j in 0:127
#push!(inps,[digits(i,2,7) digits(j,2,7)]')
#push!(outs,digits(i+j, 2, 8)')
#end
for i in 0:31, j in 0:31
push!(inps,[digits(i,2,5) digits(j,2,5)]')
push!(outs,digits(i+j, 2, 6)')
end
for i in 0:63, j in 0:63
push!(inps,[digits(i,2,6) digits(j,2,6)]')
push!(outs,digits(i+j, 2,7)')
end
totalcost(inds, A, B, C, D, E, F) = sum([sum((fwprop(inps[i], [A, B, C, D, E, F], 1) - outs[i]).^2) for i in inds])
lam = 0.007
mu = 0
gam = 0.1
velCoefss = zeros(coefss)
delRMS = zeros(coefss)
coefssupd = zeros(coefss)
for k in 1:500
inds = []
for i in 1:10

inds = rand(1:length(outs), 15)
tcurl = A -> totalcost(inds, unscrunch(A)...)
delCoefss = ForwardDiff.gradient(tcurl, coefss)
delRMS = (1 - gam)*delRMS + gam*(delCoefss.^2)
delRMSupd = sqrt.(delRMS)
velCoefss = mu*velCoefss + (1 - mu)*delCoefss
velCoefssupd = velCoefss/(1-mu)
coefssupd = lam*velCoefssupd./delRMSupd
coefss -= coefssupd
end
println(totalcost(inds, unscrunch(coefss)...))
println(sum(coefssupd[:].^2))


IH, HH, HO, Hinit, Hbias, Obias = unscrunch(coefss)
inpss = [digits(9, 2, 5)'; digits(12,2,5)']
println(digits(9 + 12, 2, 6)')
println(fwprop(inpss, [IH, HH, HO, Hinit, Hbias, Obias], 1))

inpss = [digits(1 + 2 + 4 + 8, 2, 12)'; digits(1 + 2 + 4 + 8 + 16 + 32,2,12)']
println(digits(78, 2, 13)')
println(fwprop(inpss, [IH, HH, HO, Hinit, Hbias, Obias], 1))

inpss = [digits(256 + 128 + 64 + 32 + 16 + 8, 2, 9)'; digits(1 + 2 + 4 + 8,2,9)']
println(digits(519, 2, 10)')
println(fwprop(inpss, [IH, HH, HO, Hinit, Hbias, Obias], 1))
end
