include("ESNResGen.jl")
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
  hids[:, 1] = tanh.(HH*Hinit + IH*inputs[:, 1] + Hbias)
  outs[:, 1] = tanh.(HO*hids[:, 1] + Obias)
  for i in 2:n #i=1 gets unrolled
    HH = HHbias + sum([HHfact[2*hid+1:2*hid+inp, k]'*inputs[:, i] .* (HHfact[hid+1:2*hid, k]*HHfact[1:hid, k]') for k in 1:facts])
    hids[:, i] = tanh.(HH*hids[:, i-1] + IH*inputs[:, i] + Hbias)
    outs[:, i] = tanh.(HO*hids[:, i] + Obias)
  end
  #for i in n+1:n+t
  #  HH = HHbias + sum([HHfact[1:hid, k] * HHfact[hid+1:2hid, k]' * HHfact[2hid+1:2hid+inp, k]'*inputs[:, i], for k in 1:facts])
  #  hids[:, i] = logsig.(HH*hids[:, i-1] + Hbias)
  #  outs[:, i] = logsig.(HO*hids[:, i] + Obias)
  #end
  return outs
end

function jabber(coefs, b, t)
  IH, HHbias, HHfact, HO, Hinit, Hbias, Obias = coefs
  hid = size(IH)[1]
  ins = size(IH)[2]
  outs = Array{Number}((size(Obias)[1], t))
  hids = Array{Number}((size(Hbias)[1], t))
  jab = ""
  st = "abcdefghiklmnoprst"[rand(1:18)]
  start = ind(UInt8(st))
  jab = jab*string(st)
  HH = HHbias + sum([HHfact[2*hid+1:2*hid+inp, k]'*start.* (HHfact[hid+1:2*hid, k]*HHfact[1:hid, k]') for k in 1:facts])
  hids[:, 1] = tanh.(HH*Hinit + IH*start + Hbias)
  A = tanh.(HO*hids[:, 1] + Obias)
  outs[:, 1] = zeros(size(Obias))
  a = max(A...)
  outs[rand(find(x -> x==a,A)), 1] = 1
  for i in 2:b
    st = "abcdefghiklmnoprst"[rand(1:18)]
    start = ind(UInt8(st))
    #jab = jab*string(st)
    HH = HHbias + sum([HHfact[2*hid+1:2*hid+inp, k]'*start.* (HHfact[hid+1:2*hid, k]*HHfact[1:hid, k]') for k in 1:facts])
    hids[:, i] = tanh.(HH*Hinit + IH*start + Hbias)
    A = tanh.(HO*hids[:, i] + Obias)
    outs[:, i] = zeros(size(Obias))
    a = max(A...)
    outs[rand(find(x -> x==a,A)), i] = 1
    jab = jab*string(Char(findfirst(outs[:,i])))
  end
  for i in b+1:t #i=1 gets unrolled
    HH = HHbias + sum([HHfact[2*hid+1:2*hid+inp, k]'*outs[:, i-1] .* (HHfact[hid+1:2*hid, k]*HHfact[1:hid, k]') for k in 1:facts])
    hids[:, i] = tanh.(HH*hids[:, i-1] + IH*outs[:, i-1] + Hbias)
    A = tanh.(HO*hids[:, i] + Obias)
    outs[:, i] = zeros(size(Obias))
    a = max(A...)
    outs[rand(find(x -> x==a,A)), i] = 1
    jab = jab*string(Char(findfirst(outs[:,i])))
  end
  return jab
end

inp = 128
hid = 10
hid = parse(UInt8, ARGS[2])
facts =  10
out = 128

function ind(x)
A = -.76*ones(128)
A[x] = .76
return A
end
function indn(x)
A = randn(128).^2/100 - 1
A[x] = 1 - randn()^2/100
return .76*A
end

#init
IH = randn(hid, inp)/sqrt(hid*inp + hid*hid)
HHfact = FactorTrain(hid, parse(UInt32, ARGS[3]))
HHbias = Reservoir(hid, parse(UInt32, ARGS[3]))
#HH = HH/max(abs(eig(HH)[1])...)
#Train the reservoir!
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
function VoltairData()
inps = []
outs = []
f = open("voltaire-candide-193.txt")
k = rand(500:1000)
A = Array{UInt8}(k)
read!(f, A)
for i in 1:1000
  k = rand(2:3)
  A = Array{UInt8}(k)
  read!(f, A)
  B = [ind(A[j])[i]  for i in 1:128, j in 1:k-1]
  C = [ind(A[j])[i]  for i in 1:128, j in 2:k]
  #println(B)
  push!(inps, B)
  push!(outs, C)
end
for i in 1:100
  k = rand(10:40)
  A = Array{UInt8}(k)
  read!(f, A)
  B = [ind(A[j])[i]  for i in 1:128, j in 1:k-1]
  C = [ind(A[j])[i]  for i in 1:128, j in 2:k]
  #println(B)
  push!(inps, B)
  push!(outs, C)
end
return inps, outs
end

function ABCData(n)
  inps = []
  outs = []
  abc = "abcdefghijklmnopqrstuvwxyz"[1:n]^100
  s = 0

  for i in 1:100

    k = rand(4:8)
    A = Array{UInt8}(k)
    for j in 1:k
      A[j] = abc[j+s]
    end
    B = [ind(A[j])[i]  for i in 1:128, j in 1:k-1]
    C = [ind(A[j])[i]  for i in 1:128, j in 2:k]
    #println(B)
    push!(inps, B)
    push!(outs, C)
    s = s + k

  end
  return inps, outs
end

inps, outs = VoltairData()
#inps, outs = ABCData(10)
NoHid = true
if NoHid
coefss = [IH, HO, Hinit, Hbias, Obias]
scrunch(A, D, E, F, G) = [A[:];D[:];E[:];F[:];G[:]]
unscrunch(A) = reshape(A[1:length(IH)], size(IH)), HHbias, HHfact, reshape(A[length(IH) + 1:length(IH) + length(HO)], size(HO)), A[length(IH) + length(HO) + 1: length(IH) + length(HO) + length(Hinit)], A[ length(IH) + length(HO) + length(Hinit) + 1:  length(IH) + length(HO) + length(Hinit) + length(Hbias)], A[  length(IH) + length(HO) + length(Hinit) + length(Hbias) + 1 :   length(IH) + length(HO) + length(Hinit) + length(Hbias) + length(Obias)]
totalcost(inds, A, D, E, F, G) = sum([sum((fwprop(inps[i], [A, HHbias, HHfact, D, E, F, G]) - outs[i]).^2) for i in inds])
totalcost(inds, A) = sum([sum((fwprop(inps[i], unscrunch(A)) - outs[i]).^2) for i in inds])
coefss = scrunch(coefss...)
else
scrunch(A, B, C, D, E, F, G) = [A[:];B[:];C[:];D[:];E[:];F[:];G[:]]
unscrunch(A) = reshape(A[1:length(IH)], size(IH)), reshape(A[length(IH) + 1:length(IH) + length(HHbias)], size(HHbias)), reshape(A[length(IH) + length(HHbias) + 1:length(IH) + length(HHbias) + length(HHfact)], size(HHfact)), reshape(A[length(IH) + length(HHbias) + length(HHfact) + 1:length(IH) + length(HHbias) + length(HHfact) + length(HO)], size(HO)), A[length(IH) + length(HHbias) + length(HHfact) + length(HO) + 1: length(IH) + length(HHbias) + length(HHfact) + length(HO) + length(Hinit)], A[ length(IH) + length(HHbias) + length(HHfact) + length(HO) + length(Hinit) + 1:  length(IH) + length(HHbias) + length(HHfact) + length(HO) + length(Hinit) + length(Hbias)], A[  length(IH) + length(HHbias) + length(HHfact) + length(HO) + length(Hinit) + length(Hbias) + 1 :   length(IH) + length(HHbias) + length(HHfact) + length(HO) + length(Hinit) + length(Hbias) + length(Obias)]
totalcost(inds, A, B, C, D, E, F, G) = sum([sum((fwprop(inps[i], [A, B, C, D, E, F, G]) - outs[i]).^2) for i in inds])
totalcost(inds, A) = sum([sum((fwprop(inps[i], unscrunch(A)) - outs[i]).^2) for i in inds])
coefss = scrunch(coefss...)
end
lam = 0.5
lam = float(ARGS[1])
mu = 0
gam = 0.1
velCoefss = zeros(coefss)
delRMS = ones(coefss)/10^12
coefssupd = zeros(coefss)
Konfig = ForwardDiff.MultithreadConfig(ForwardDiff.GradientConfig(coefss))
function updates(tcurl, coefss)
  return ForwardDiff.gradient(tcurl, coefss)
end
function pupdates(tcurl, coefss)
  return ForwardDiff.gradient(tcurl, coefss, Konfig)
end
for k in 1:5000
println("Starting ", k)
inds = rand(1:length(outs), 10)
println("Total length ", sum([size(inps[i])[2] for i in inds]))
tcurl = A -> totalcost(inds, A)
score = 0
letters = 0
for i in inds
#println(size(inds[i]))
  outp = fwprop(inps[i], unscrunch(coefss))
  strin = ""
  strout = ""
  for k in 1:size(outp)[2]
  a = max(outs[i][:, k]...)
  answer = string(Char(rand(find(x -> x==a,outs[i][:, k]))))
  strin = strin*answer
  #println(strin)
  a = max(outp[:, k]...)
  guess = string(Char(rand(find(x -> x==a,outp[:, k]))))
  if guess == answer
    letters += 1
  end
  strout = strout*guess
  end
  if strout == strin
    score += 1
  end
  println(strout)

end
println("Perfect guesses: ", score, " out of ", length(inds))
println("Guessed letters: ", 100* letters / sum([size(inps[i])[2] for i in inds]))
#Compute these any way you want;
delCoefss = pupdates(tcurl, coefss)
#Sync up and apply them one after another, then start again.

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


println(jabber([unscrunch(coefss)...], 10, 75))
println(jabber([unscrunch(coefss)...], 15, 75))
println(jabber([unscrunch(coefss)...], 20, 75))
println(jabber([unscrunch(coefss)...], 25, 75))
println(jabber([unscrunch(coefss)...], 30, 75))
end
