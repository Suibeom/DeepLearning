using ForwardDiff
function exlin(x)
  return x >= 0 ? x : exp(x)-1
end
function logsig(x)
  return 1 / (1 + exp(-x))
end



function fwprop(inputs, coefs)
  IH, HHbias, HHfact, HO, Hinit, Hbias, Obias = coefs
  #HHfact is hid + hid + in by k
  t = 0
  n = size(inputs)[2]
  hid = size(IH)[1]
  ins = size(IH)[2]
  outs = Array{Number}((size(Obias)[1], n + t))
  hids = Array{Number}((size(Hbias)[1], n + t))
  #println(size(HHfact[2*hid+1:2*hid+inp, 1]'))
  #println(size(HHfact[2*hid+1:2*hid+inp, 1]'*inputs[:, 1]))
  #println(size(HHfact[2*hid+1:2*hid+inp, 1]'*inputs[:, 1] .* (HHfact[hid+1:2*hid, 1]*HHfact[1:hid, 1]')))
  HH = HHbias + sum([HHfact[2*hid+1:2*hid+inp, k]'*inputs[:, 1] .* (HHfact[hid+1:2*hid, k]*HHfact[1:hid, k]') for k in 1:facts])
  hids[:, 1] = logsig.(HH*Hinit + IH*inputs[:, 1] + Hbias)
  outs[:, 1] = logsig.(HO*hids[:, 1] + Obias)
  for i in 2:n #i=1 gets unrolled
    HH = HHbias + sum([HHfact[2*hid+1:2*hid+inp, k]'*inputs[:, i] .* (HHfact[hid+1:2*hid, k]*HHfact[1:hid, k]') for k in 1:facts])
    hids[:, i] = logsig.(HH*hids[:, i-1] + IH*inputs[:, i] + Hbias)
    outs[:, i] = logsig.(HO*hids[:, i] + Obias)
  end
  #for i in n+1:n+t
  #  HH = HHbias + sum([HHfact[1:hid, k] * HHfact[hid+1:2hid, k]' * HHfact[2hid+1:2hid+inp, k]'*inputs[:, i], for k in 1:facts])
  #  hids[:, i] = logsig.(HH*hids[:, i-1] + Hbias)
  #  outs[:, i] = logsig.(HO*hids[:, i] + Obias)
  #end
  return outs
end

function jabber(coefs, t)
  IH, HHbias, HHfact, HO, Hinit, Hbias, Obias = coefs
  st = "abcdefghiklmnoprst"[rand(1:18)]
  start = ind(UInt8(st))
  jab = string(st)
  hid = size(IH)[1]
  ins = size(IH)[2]
  outs = Array{Number}((size(Obias)[1], t))
  hids = Array{Number}((size(Hbias)[1], t))
  HH = HHbias + sum([HHfact[2*hid+1:2*hid+inp, k]'*start.* (HHfact[hid+1:2*hid, k]*HHfact[1:hid, k]') for k in 1:facts])
  hids[:, 1] = logsig.(HH*Hinit + IH*start + Hbias)
  A = logsig.(HO*hids[:, 1] + Obias)
  outs[:, 1] = zeros(size(Obias))
  a = max(A...)
  outs[rand(find(x -> x==a,A)), 1] = 1
  jab = jab*string(Char(findfirst(outs[:,1])))
  for i in 2:t #i=1 gets unrolled
    HH = HHbias + sum([HHfact[2*hid+1:2*hid+inp, k]'*outs[:, i-1] .* (HHfact[hid+1:2*hid, k]*HHfact[1:hid, k]') for k in 1:facts])
    hids[:, i] = logsig.(HH*hids[:, i-1] + IH*outs[:, i-1] + Hbias)
    A = logsig.(HO*hids[:, i] + Obias)
    outs[:, i] = zeros(size(Obias))
    a = max(A...)
    outs[rand(find(x -> x==a,A)), i] = 1
    jab = jab*string(Char(findfirst(outs[:,i])))
  end
  return jab
end

inp = 128
hid = 10
facts = 10
out = 128

function ind(x)
A = zeros(128)
A[x] = 1
return A
end
function indn(x)
A = randn(128).^2/100
A[x] = 1 - randn()^2/100
return A
end

#init
IH = randn(hid, inp)/sqrt(hid*inp + hid*hid)
HHfact = randn(2hid + inp, facts)/sqrt(sqrt(hid*hid + hid*hid))
HHbias = randn(hid, hid)/sqrt(hid*hid + hid*hid)
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
coefss = [IH, HHbias, HHfact, HO, Hinit, Hbias, Obias]
inps = []
outs = []

f = open("voltaire-candide-193.txt")
k = rand(50:500)
A = Array{UInt8}(k)
read!(f, A)
for i in 1:1000
  k = rand(6:15)
  A = Array{UInt8}(k)
  read!(f, A)
  B = [ind(A[j])[i]  for i in 1:128, j in 1:k-1]
  C = [ind(A[j])[i]  for i in 1:128, j in 2:k]
  #println(B)
  push!(inps, B)
  push!(outs, C)
end

scrunch(A, B, C, D, E, F, G) = [A[:];B[:];C[:];D[:];E[:];F[:];G[:]]

unscrunch(A) = reshape(A[1:length(IH)], size(IH)), reshape(A[length(IH) + 1:length(IH) + length(HHbias)], size(HHbias)), reshape(A[length(IH) + length(HHbias) + 1:length(IH) + length(HHbias) + length(HHfact)], size(HHfact)), reshape(A[length(IH) + length(HHbias) + length(HHfact) + 1:length(IH) + length(HHbias) + length(HHfact) + length(HO)], size(HO)), A[length(IH) + length(HHbias) + length(HHfact) + length(HO) + 1: length(IH) + length(HHbias) + length(HHfact) + length(HO) + length(Hinit)], A[ length(IH) + length(HHbias) + length(HHfact) + length(HO) + length(Hinit) + 1:  length(IH) + length(HHbias) + length(HHfact) + length(HO) + length(Hinit) + length(Hbias)], A[  length(IH) + length(HHbias) + length(HHfact) + length(HO) + length(Hinit) + length(Hbias) + 1 :   length(IH) + length(HHbias) + length(HHfact) + length(HO) + length(Hinit) + length(Hbias) + length(Obias)]

totalcost(inds, A, B, C, D, E, F, G) = sum([sum((fwprop(inps[i], [A, B, C, D, E, F, G]) - outs[i]).^2) for i in inds])
totalcost(inds, A) = sum([sum((fwprop(inps[i], unscrunch(A))[end-3:end] - outs[i][end-3:end]).^2) for i in inds])
coefss = scrunch(coefss...)
lam = 0.05
mu = 0
gam = 0.1
velCoefss = zeros(coefss)
delRMS = ones(coefss)/10^12
coefssupd = zeros(coefss)
for k in 1:5000
Konfig = ForwardDiff.MultithreadConfig(ForwardDiff.GradientConfig(coefss))
function updates(tcurl, coefss)
  return ForwardDiff.gradient(tcurl, coefss)
end
function pupdates(tcurl, coefss)
  return ForwardDiff.gradient(tcurl, coefss, Konfig)
end

println("Starting ", k)
inds = rand(1:length(outs), 10)
tcurl = A -> totalcost(inds, A)
delCoefss = pupdates(tcurl, coefss)
println(sum(delCoefss[:].^2))
delRMS = (1 - gam)*delRMS + gam*(delCoefss.^2)
println(sum(delRMS[:].^2))
delRMSupd = sqrt.(delRMS)
println(min(delRMSupd[:]...))
velCoefss = mu*velCoefss + (1 - mu)*delCoefss
velCoefssupd = velCoefss/(1-mu)
coefssupd = lam*velCoefssupd./delRMSupd
println(totalcost(inds, coefss))
coefss -= coefssupd
println(totalcost(inds, coefss))
println(sum(coefssupd[:].^2))
println(jabber([unscrunch(coefss)...], 75))
println(jabber([unscrunch(coefss)...], 75))
println(jabber([unscrunch(coefss)...], 75))
println(jabber([unscrunch(coefss)...], 75))
println(jabber([unscrunch(coefss)...], 75))
end
