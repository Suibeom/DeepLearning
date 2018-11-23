# DeepLearning
I wrote these while working through Geoffry Hinton's deep learning Coursera course. At the time, there were no good deep learning frameworks for Julia and I wanted to learn the language, so I reimplemented the algorithms by hand (like you do, in a good course on algorithms..!)

I think they were written for Julia v0.4, and some aggressive refactoring has been done now that we're at v1.02 so some stuff probably needs to be tweaks to make them runnable on the latest stable Julia version.

Echo State Networks are antique technology now, but I thought they were fun. The idea was that in the early days when people were not having much success training RNNs, because they were too big and SGD optimizers were not very sophistocated, you would instead start with something that just fed forward internal states chaotically. So, the network would learn how to put things into the internal state, and how to read things out of it after it's fed forward, but it would NOT learn how to feed forward the internal states.

RBMs are also pretty antique. The idea is that you learn a Gibbs measure tied between your hidden and visible states, in such a way that states with the right visible parts are low energy, and ones with the wrong visible parts are high energy. It's basically a bizzaro-world thermodynamic version of an autoencoder, and they're similar enough to a real autoencoder that you can use them to pretrain things in the same way.
These days this stuff is too convoluted and computationally intense. Just use regular autoencoders... Fun gadget though.

Factor RNN is no longer the widely used name for this network, but it's also not in fashion anymore. The idea was to cause an RNN to change the way it passed its hidden states forward depending on the current input. Most RNN architectures do NOT change the feedforward weights dynamically; the inputs interact with the hidden states additively. 
