import numpy as np

def to_texture(data, texture_size, mean=True, variance=True):

    w = texture_size
    N = len(data)

    ret = np.array(())
    ret.shape = (0,0)
    dT = data.T
    it = 1 if dT.ndim ==  1 else dT.shape[0]

    #print 'N:', N

    n = 0
    S = np.array([0.0] * it)
    m = np.array([0.0] * it)

    #print dT.shape

    saida = np.zeros((data.shape[0], data.shape[1]*2))

    #print 'saida', saida.shape

    for x in dT[:,:w].T:
        n+=1
        n_m = m + (x-m)/n
        n_s = S + (x-m)*(x-n_m)
        m = n_m
        S = n_s

    y = np.concatenate((m, S/n), axis=0)
    saida[n] = y
    #print y, y.shape, n

    for i in xrange(w, N):
        m = n_m
        n_m = m + (dT[:,i]-m)/n - (dT[:,i-w]-m)/n
        S = S + ( (dT[:,i] - n_m) * (dT[:,i] - m) ) - ( ( dT[:,i-w] - m )*( dT[:,i-w] - n_m ) )

        y = np.concatenate((m, S/n), axis=0)
        saida[i] = y
    
    #print saida[n]

    if mean and variance:
        return saida[w:,:]

    if mean:
        return saida[w:,:saida.shape[1]/2]

    if variance:
        return saida[w:,saida.shape[1]/2:]

    return None

if __name__ == "__main__":

    import time

    a = np.random.random( (20000,34) )
    print a.shape

    st = time.time()
    b = to_texture(a, 100, variance=False)
    print b.shape
    print('it took %.4fs to calculate' % (time.time() - st))

