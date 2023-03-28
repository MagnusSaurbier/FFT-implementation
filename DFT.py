from cmath import cos, exp, sin, pi
import random
import time
import matplotlib.pyplot as plt
import numpy as np
class Function:
    def __init__(self, N, f, a=1, imperfection=0):
        self.N = N
        self.f = f
        self.a = a
        self.imperfection = imperfection
    def evaluate(self, x):
        return cos(2*pi*x*self.f/self.N)*self.a+ (2*random.random()*self.imperfection - self.imperfection)
    def __call__(self, x):
        return self.evaluate(x)
    def toArray(self, N=None):
        if N is None: N = self.N
        return [self.evaluate(x) for x in range(N)]
    
class SquareFunction(Function):
    def evaluate(self, x):
        return self.a if x*self.f % self.N < self.N//2 else -self.a
    
class CombinedFunction():
    def synthesize(self, N, freqList):
        self.fs = [Function(N, fa[0], fa[1]) for fa in freqList]
        return self
    def accumulate(self, fs):
        self.fs = fs
        return self
    def evaluate(self, x):
        return sum(f(x) for f in self.fs)
    def __call__(self, x):
        return self.evaluate(x)
    def toArray(self, N=None):
        if N is None: N = self.fs[0].N
        return [self.evaluate(x) for x in range(N)]
    
def DFT(F):
    """Discrete Fourier Transform of f, with N samples"""
    M = len(F)
    def evaluate(k):
        s = 0
        for n in range(M):
            s += F[n] * exp(-2j * pi * k * n / M) / M
        return s
    return [evaluate(k) for k in range(M)]

def IDFT(F):
    """Inverse Discrete Fourier Transform of F"""
    N = len(F)
    def evaluate(n):
        return sum(F[k] * exp(2j * pi * k * n / N)
            for k in range(N))
    return [evaluate(n) for n in range(N)]

def FFT(F):
    """Fast Fourier Transform of F, with M samples"""
    if len(F) == 1: return [F[0]]
    A0 = FFT(F[::2])
    A1 = FFT(F[1::2])
    w = exp(-2j*pi/len(F))
    m = len(F)//2
    return ([
        .5 * (A0[k%m] + A1[k%m] * w**k)
            for k in range(len(F))])

def FFTnp(F):
    """Fast Fourier Transform of F, with M samples using numpy"""
    M = F.shape[0]
    if M == 1: return F
    A0 = FFTnp(F[::2])
    A1 = FFTnp(F[1::2])
    Wk = np.exp(-2j*pi*np.arange(M//2)/M)
    return .5*(np.concatenate((A0+A1*Wk, A0-A1*Wk)))

def FFTnp2(F):
    """Fast Fourier Transform of F, with M samples using numpy"""

    #print("generating points")
    starttime = time.time()
    points = np.array(F)
    #print("points generated time:", time.time()-starttime)
    starttime = time.time()
    #print("numpy starting")
    result = np.fft.fft(points)
    #print("numpy done time:", time.time()-starttime)
    return result


def IFFT(F):
    """Inverse Fast Fourier Transform of Ak"""
    if len(F) == 1: return [F[0]]
    A0 = IFFT(F[::2])
    A1 = IFFT(F[1::2])
    w = exp(2j*pi/len(F))
    m = len(F)//2
    return ([
        (A0[k%m] + A1[k%m] * w**k)
            for k in range(len(F))])

def fourier(f:Function, fast=2):
    f = f.toArray()
    starttime = time.time()
    N = len(f)
    if fast==0: F = DFT(f)
    elif fast==1: F = FFT(f)
    elif fast==2: F = FFTnp(np.array(f))
    elif fast==3: F = FFTnp2(f)
    delta = time.time() - starttime
    #print(f"{'FFT'if fast else 'DFT'} done in {delta} seconds")
    #barchart
    maxFreq = min(100, N)

    # print("\nContributing frequencies:")
    # for i in range(1, f.N):
    #     if abs(F[i]) > 5: print("f =", i, "amplitude =", F[i])

    plt.figure(0)
    plt.bar(range(maxFreq), [abs(x) for x in F[:maxFreq]])
    plt.title(f"{'FFT'if fast else 'DFT'} of g")
    plt.xlabel("Frequency")
    plt.ylabel("Coefficient")
    #plt.show()

    # # Compression
    # quality = 0.25
    # plt.figure(1)
    # starttime = time.time()
    # if fast: A=IFFT(np.concatenate(F[:int(maxFreq*quality)], np.array([0]*(N-int(maxFreq*quality)))))
    # else: A=IDFT(F[:maxFreq//2]+[0]*(N-maxFreq//2))
    # delta = time.time() - starttime
    # print(f"Compressed {'IFFT'if fast else 'IDFT'} done in {delta} seconds")
    # plt.plot(range(N), A)
    # plt.title(f"Compressed {'IFFT'if fast else 'IDFT'} of g with quality {quality}")
    # plt.xlabel("x")
    # plt.ylabel("y")

    # IFFT
    starttime = time.time()
    if fast: A = IFFT(F)
    else: A = IDFT(F)
    delta = time.time() - starttime
    #print(f"{'IFFT'if fast else 'IDFT'} done in {delta} seconds")
    plt.figure(2)
    plt.plot(range(N), A)
    plt.title(f"{'IFFT'if fast else 'IDFT'} of g")
    plt.xlabel("x")
    plt.ylabel("y")

    # Original function
    plt.figure(3)
    plt.plot(range(N), [f[x] for x in range(N)])
    plt.title(f"Original function g")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

def speedTest():
    N = 2**18
    g = CombinedFunction().synthesize(N, [(1,50), (5, 10), (20, 3)])
    
    starttime=time.time()
    fourier(g, 1)
    print("FFT: ", time.time()-starttime)

    starttime=time.time()
    fourier(g, 2)
    print("FFTnp: ", time.time()-starttime)

    starttime=time.time()
    fourier(g, 3)
    print("FFTnp2: ", time.time()-starttime)

#speedTest()

# # Plot the FFT of a sine wave
# fast = 2
# N = 2**10
# nFreqs = 10
# maxFreq = 50
# amp = 50
# randList = []
# for _ in range(nFreqs):
#     f = random.randint(1, maxFreq)
#     a = random.randint(1, amp)/(f)
#     randList.append((f, a))
# g = CombinedFunction().synthesize(N, randList)

# fourier(g)
