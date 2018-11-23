using ForwardDiff
#train an n-by-n tanh dynamical reservoir using two (later n?) seeds
function Step(v, M, k)
  vk = copy(v)
  for i in 1:k
    vk = tanh.(M*vk)
  end
  return vk
end
function FStep(v, M, k)
  vk = copy(v)
  l = UInt32(length(M)/2)
  for i in 1:k
    vk = tanh.(M[l+1:2*l]*M[1:l]'*vk)
  end
  return vk
end
function FactorTrain(n,e)
  v = randn(2*n)
  Konfig = ForwardDiff.MultithreadConfig(ForwardDiff.GradientConfig(v))
  lam = 0.001
  rms = ones(v)/10^12
  for i in 1:e
    nTestStates = floor(UInt32, sqrt(n))
    TestStates = randn(n, nTestStates)

    for i in 1:nTestStates
      #clamp to -1, 1 ?? Maybe try -0.76 to 0.76 in the future
      TestStates[:, i] = 0.76*TestStates[:, i]/max(abs.(TestStates[:, i])...)
    end

    CF(v) = sum([sum(TestStates[:, i].^2) - sum(FStep(TestStates[:, i], v, 50).^2) for i in 1:nTestStates].^2)
    mnorm(v) = max([sum(FStep(TestStates[:, i], v, 10).^2)./sum(TestStates[:, i].^2) for i in 1:nTestStates]...)
    dv = ForwardDiff.gradient(CF, v)
    rms = 0.9*rms + 0.1*dv.^2
    rmsup = sqrt.(rms)
    up = lam*dv./rmsup
    v -= up
    dv = nTestStates*ForwardDiff.gradient(mnorm, v, Konfig)
    rms = 0.9*rms + 0.1*dv.^2
    rmsup = sqrt.(rms)
    up = lam*dv./rmsup
    v -= up
    println(CF(v))
    println(mnorm(v))
  end
end
function Reservoir(n, e)

  Seed1 = randn(n, n)
  Seed2 = randn(n, n)
  nTestStates = UInt32(3*n)
  TestStates = randn(n, nTestStates)
  for i in 1:nTestStates
    #clamp to -1, 1 ?? Maybe try -0.76 to 0.76 in the future
    TestStates[:, i] = 0.76*TestStates[:, i]/max(abs.(TestStates[:, i])...)
  end
  v = randn(2)
  Konfig = ForwardDiff.MultithreadConfig(ForwardDiff.GradientConfig(v))
  println([sum(Step(TestStates[:, i], v[1]*Seed1 + v[2]*Seed2, 50).^2)/sum(TestStates[:, i].^2) for i in 1:nTestStates])
  lam = 0.001
  rms = ones(v)/10^12
  for i in 1:e
    nTestStates = floor(UInt32, n)
    TestStates = randn(n, nTestStates)
    for i in 1:nTestStates
      #clamp to -1, 1 ?? Maybe try -0.76 to 0.76 in the future
      TestStates[:, i] = 0.76*TestStates[:, i]/max(abs.(TestStates[:, i])...)
    end
    CF(v) = sum([sum(TestStates[:, i].^2) - sum(Step(TestStates[:, i], v[1]*Seed1 + v[2]*Seed2, 50).^2) for i in 1:nTestStates].^2)
    mnorm(v) = max([sum(Step(TestStates[:, i], v[1]*Seed1 + v[2]*Seed2, 10).^2)./sum(TestStates[:, i].^2) for i in 1:nTestStates]...)
    dv = ForwardDiff.gradient(CF, v)
    rms = 0.9*rms + 0.1*dv.^2
    rmsup = sqrt.(rms)
    up = lam*dv./rmsup
    v -= up
    dv = nTestStates*ForwardDiff.gradient(mnorm, v, Konfig)
    rms = 0.9*rms + 0.1*dv.^2
    rmsup = sqrt.(rms)
    up = lam*dv./rmsup
    v -= up
    println(CF(v))
    println(mnorm(v))
  end
  nTestStates = 4*n
  TestStates = randn(n, nTestStates)
  for i in 1:nTestStates
    #clamp to -1, 1 ?? Maybe try -0.76 to 0.76 in the future
    TestStates[:, i] = 0.76*TestStates[:, i]/max(abs.(TestStates[:, i])...)
  end
  k = 1.1
  while k > 1
    k = max([sum(Step(TestStates[:, i], v[1]*Seed1 + v[2]*Seed2, 100).^2)./sum(TestStates[:, i].^2) for i in 1:nTestStates]...)
    v = .999*v
    println(k)
  end
  return v[1]*Seed1 + v[2]*Seed2
end
#Reservoir(100)
