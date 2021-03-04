import pyworld
import pysptk

#F0_FLOOR = 80
#F0_CEIL = 400
F0_FLOOR = 80
F0_CEIL = 520

def estimatef0(x, fs=16000, frame_period=20):
    f0, _ = pyworld.harvest(x, fs, f0_floor=40, f0_ceil=700, frame_period=frame_period)
    return f0

def analyze(x, fs=16000, frame_period=20):
    f0, time_axis = pyworld.harvest(x, fs, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=frame_period)
    spc = pyworld.cheaptrick(x, f0, time_axis, fs, fft_size=1024)
    ap = pyworld.d4c(x, f0, time_axis, fs, fft_size=1024)
    
    mcep = pysptk.sp2mc(spc, 24, 0.410)
    codeap = pyworld.code_aperiodicity(ap, fs)
    
    #return x, fs, f0, time_axis, spc, ap, mcep, codeap
    return f0, mcep, codeap

def synthesize(f0, mcep, codeap, fs=16000, frame_period=20):
    ap = pyworld.decode_aperiodicity(codeap, fs, 1024)
    spc = pysptk.mc2sp(mcep, 0.410, 1024)
    y = pyworld.synthesize(f0, spc, ap, fs, frame_period=frame_period)
    return y
