from itertools import count
import struct
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pyaudio
import wave

from DFT import DFT, FFT, fourier, Function, CombinedFunction

#Constants
CHUNK = 1024*1
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

SECONDS = 5


p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK,
)


fig, ax = plt.subplots(2)
x = np.arange(0, 2 * CHUNK, 2)
x_freqs = np.linspace(0, RATE, CHUNK)

line, = ax[0].plot(x, np.random.rand(CHUNK))
ax[0].set_ylim(-1, 1)
ax[0].set_xlim(0, CHUNK)
freqs, = ax[1].semilogx(x_freqs, np.random.rand(CHUNK))
ax[1].set_ylim(-.2, .2)
ax[1].set_xlim(20, RATE/2)
ax[1].set_xscale('log')
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_ylabel('Coefficient (re)')



plt.ion()
plt.show()
while True:
    binary = stream.read(CHUNK, exception_on_overflow=False)
    value = np.frombuffer(binary, dtype=np.int16)*2**-12
    line.set_ydata(value)

    fft = FFT(value[:CHUNK])
    m = max(fft[1:CHUNK//4])
    n = fft.index(m)
    print(f"Frequency: {x_freqs[n]}")
    freqs.set_ydata(np.abs(fft))



    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.05)






# index = count()7
# def animate(i):
#     data = stream.read(CHUNK, exception_on_overflow=False)
#     value = np.array(struct.unpack(str(2 * CHUNK) + 'B', data), dtype='b')[::2]

#     plt.figure(1)
#     plt.cla()
#     plt.plot(np.arange(CHUNK), value)
#     fft = FFT(CHUNK, lambda x: value[x])
#     plt.figure(2)
#     plt.cla()
#     plt.plot(np.arange(CHUNK), [abs(y) for y in fft])

# ani = FuncAnimation(plt.gcf(), animate, interval=100)
# plt.show()


