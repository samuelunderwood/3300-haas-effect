# app.py
# Streamlit demo of the Haas (precedence) effect
import io
import wave
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Haas (Precedence) Effect Demo", layout="centered")

# ---------- Helpers ----------
def db_to_amp(db):
    return 10 ** (db / 20.0)

def write_wav_bytes(stereo_float, fs=48000):
    """Convert float32 stereo [-1,1] to 16-bit WAV bytes."""
    # Safety limiter
    peak = np.max(np.abs(stereo_float))
    if peak > 1:
        stereo_float = stereo_float / (peak * 1.01)
    pcm = (stereo_float * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(fs)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()

def make_click(fs):
    """Single-sample impulse with a short 1 ms Hann fade to avoid DAC pops."""
    dur_ms = 5
    n = int(fs * dur_ms / 1000)
    x = np.zeros(n, dtype=np.float32)
    x[0] = 1.0
    # 1 ms Hann window
    w = int(fs * 0.001)
    if w > 2 and w < n:
        hann = np.hanning(w * 2)[:w]
        x[:w] *= hann[::-1]  # small fade-out
    x *= 0.25
    return x

def make_noise_burst(fs, burst_ms=10, band="white"):
    n = max(1, int(fs * burst_ms / 1000))
    if band == "white":
        x = np.random.randn(n).astype(np.float32)
    else:
        x = np.random.randn(n).astype(np.float32)
    # Apply 3 ms fade in/out to avoid clicks
    fade_ms = min(3, burst_ms/3)
    fns = max(1, int(fs * fade_ms / 1000))
    win = np.ones(n, dtype=np.float32)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fns, dtype=np.float32))
    win[:fns] = ramp
    win[-fns:] = ramp[::-1]
    x *= win
    x /= max(1e-9, np.max(np.abs(x)))
    x *= 0.25
    return x

def build_stereo_stimulus(
    fs,
    stim_type="Click",
    delay_ms=10.0,
    lead="Left",
    lag_rel_level_db=0.0,
    repetitions=3,
    io_interval_ms=700,
    noise_ms=10,
):
    """Return stereo float32 signal of shape (N, 2)."""
    # Make monophonic event
    if stim_type == "Click":
        ev = make_click(fs)
    else:
        ev = make_noise_burst(fs, burst_ms=noise_ms)

    delay_samps = int(round(fs * delay_ms / 1000.0))
    gap_samps = int(round(fs * io_interval_ms / 1000.0))
    lead_gain = 1.0
    lag_gain = db_to_amp(lag_rel_level_db)

    # One repetition length (enough to fit both lead & lag plus a small tail)
    rep_len = delay_samps + len(ev) + max(200, int(0.05 * fs))
    # Start each rep with a small onset pad so plots look nice
    onset_pad = int(0.02 * fs)

    # Build a single repetition stereo buffer
    L = np.zeros(onset_pad + rep_len, dtype=np.float32)
    R = np.zeros(onset_pad + rep_len, dtype=np.float32)

    if lead == "Left":
        L[onset_pad:onset_pad + len(ev)] += lead_gain * ev
        R[onset_pad + delay_samps:onset_pad + delay_samps + len(ev)] += lag_gain * ev
    else:
        R[onset_pad:onset_pad + len(ev)] += lead_gain * ev
        L[onset_pad + delay_samps:onset_pad + delay_samps + len(ev)] += lag_gain * ev

    # Tile repetitions with inter-onset interval (gap between starts)
    start_stride = max(len(L), gap_samps + len(ev) + delay_samps + onset_pad)
    total_len = start_stride * repetitions
    outL = np.zeros(total_len, dtype=np.float32)
    outR = np.zeros(total_len, dtype=np.float32)

    for i in range(repetitions):
        s = i * start_stride
        e = s + len(L)
        outL[s:e] += L
        outR[s:e] += R

    # Normalize gently to avoid clipping
    peak = max(1e-9, np.max(np.abs([outL, outR])))
    if peak > 0.98:
        outL /= peak * 1.02
        outR /= peak * 1.02

    stereo = np.stack([outL, outR], axis=1)  # (N,2)
    return stereo

def interpret_delay(delay_ms):
    d = delay_ms
    if d < 1:
        return "Near-simultaneous arrival; image tends to center (phantom center)."
    elif d < 5:
        return "Strong precedence: localizes to the leading ear/speaker."
    elif d < 40:
        return "Precedence region: still fused as one event, localized to the lead."
    else:
        return "Depending on relative level, a distinct separate echo may now be audible."

# ---------- UI ----------
st.title("Haas (Precedence) Effect — Interactive Demo")
st.subheader("Prepared for AREN 3300 by Sam Underwood")
st.markdown(
    """
Use **headphones**. This app plays a brief sound to both of your ears in quick succession. Using the 
control panel on the left side of the window, you can adjust the time gap and level difference between the two stimuli,
as well as which ear receives the sound first.

Using this demo, you can induce some interesting perceptual phenomena related to the Haas Effect. Pay attention to two things when you listen to the stimuli:

1) Do you hear two separate sounds? Or one sound?
2) Which direction do you perceive the sound to be coming from?

When the delay time is very short, your ear will likely integrate the sounds together into one sound. Depending on relative levels, you might perceive the sound is being centered in your head, or very close to either of your ears.

Up to roughly **~40 ms**, most listeners still perceive one sound (precedence). Beyond that, an **echo** may appear depending on the relative level between stimuli.

"""
)

with st.sidebar:
    st.header("Controls")
    stim_type = st.radio("Stimulus", ["Click", "Noise burst"], index=0)
    if stim_type == "Noise burst":
        noise_ms = st.slider("Noise burst duration (ms)", 5, 50, 10, step=1)
    else:
        noise_ms = 10  # ignored for click

    delay_ms = st.slider("Interaural/Speaker Delay (ms)", 0.0, 100.0, 10.0, step=0.1)
    lead = st.radio("Leading side", ["Left", "Right"], index=0)
    lag_rel_level_db = st.slider(
        "Lag level relative to lead (dB)",
        -6.0, 10.0, 0.0, step=0.5,
        help="Negative values make the lagging sound quieter; positive makes it louder."
    )
    repetitions = st.slider("Repetitions", 1, 6, 3, step=1, help="The number of times the dual stimuli repeat.")
    io_interval_ms = st.slider("Interval between repetitions (ms)", 300, 1500, 700, step=50)
    fs = st.select_slider("Sample rate (Hz)", options=[32000, 44100, 48000], value=48000, help="Sample rate of the .wav file generated (shouldn't need to adjust this)")

st.divider()



st.subheader("1) Generate & Play")
st.write("Press **Create stimulus** then click the play button. For best effect, keep your overall system volume moderate.")

if st.button("Create stimulus"):
    stereo = build_stereo_stimulus(
        fs=fs,
        stim_type=stim_type,
        delay_ms=delay_ms,
        lead=lead,
        lag_rel_level_db=lag_rel_level_db,
        repetitions=repetitions,
        io_interval_ms=io_interval_ms,
        noise_ms=noise_ms,
    )
    wav_bytes = write_wav_bytes(stereo, fs=fs)
    st.audio(wav_bytes, format="audio/wav")

    # Build audio
    st.write(f"**What you probably should be hearing given your current settings:**  \n{interpret_delay(delay_ms)}")
    st.caption("Tip: Try swapping the leading ear to see if the auditory 'image' jumps sides without changing the content itself.")
    # Download button
    st.download_button(
        "Download WAV",
        data=wav_bytes,
        file_name=f"haas_demo_{stim_type.lower()}_{lead.lower()}lead_{delay_ms:.1f}ms.wav",
        mime="audio/wav",
    )

    # Plot a short window so the delay is visible
    st.subheader("2) Waveform view")
    # Show first 120 ms (or entire signal if shorter)
    show_ms = 120
    n_show = min(len(stereo), int(fs * show_ms / 1000))
    t = np.arange(n_show) / fs * 1000.0  # ms
    fig = plt.figure(figsize=(7, 3))
    plt.plot(t, stereo[:n_show, 0], label="Left", linewidth=1.0)
    plt.plot(t, stereo[:n_show, 1], label="Right", linewidth=1.0)
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.title(f"Delay ≈ {delay_ms:.1f} ms (lead: {lead}).")
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    st.pyplot(fig)

st.divider()

# ---- Delay vs Lag Level map ----
st.subheader("Relative level vs. delay time")
st.caption("The blue marker shows your current settings. This plot is analogous to the Haas Effect plot shown in Lecture 7.")

fig2 = plt.figure(figsize=(6, 4))
plt.scatter([delay_ms], [lag_rel_level_db], s=80, marker="o")
plt.xlim(0, 100)        # delay time in ms
plt.ylim(-6, 10)        # +10 at top down to -6 at bottom
plt.xlabel("Delay time (ms)")
plt.ylabel("Relative level (lag - lead) dB")
plt.grid(True, alpha=0.3)

# Optional: label the point
plt.annotate(f"({delay_ms:.1f} ms, {lag_rel_level_db:.1f} dB)",
             xy=(delay_ms, lag_rel_level_db),
             xytext=(5, 5), textcoords="offset points", fontsize=9)

st.pyplot(fig2)

st.markdown(
    """
**What’s happening here?**

- The ear which receives the *first-arriving* sound dominates your perception of where the sound is coming from (the **Haas** / **precedence** effect).
- Under certain conditions, the *lagging* sound can be fused (i.e., integrated) perceptually with the leading sound.
- Making the lag **quieter** (negative “Lag level”) strengthens precedence and causes sounds to be integrated even with exaggerated delay times.
- Making the lag **louder** (positive "lag level") makes it more likely that the separate echo will be detected.

**OK, what is the practical application of this?**

- Designers of sound reinforcement systems in large venues must contend with the Haas Effect. You usually want the first sound 
arrival to come from the stage so that it takes precedence (i.e., the auditory perception matches your visual perception of the source location).
However, for large venues you also need to add reinforcement speakers to help get more sound coverage for back areas of the audience. 
These speakers need to be carefully placed on an electronic time delay so that they don't hit the audience first and take precedence.

- Understanding the Haas Effect will be critical when we start to discuss 'reverberation' in the next unit. 
In rooms, sound bounces all around very quickly (around 343 m/s, the speed of sound!) and an individual receiver 
inside a room receives multiple passing waves in rapid succession. Our ear does not usually detect the individual 
reflections because the delay time is so small. Instead our ear integrates the reflections into the perception of "reverberation."
Again, the precedence effect matters. In spite of all the reverberation, we will localize sound sources in the direction 
of the first arriving sound (which is often the straight-line direct path between source and receiver). In many room uses (particularly those involving music), harsh distinct echoes
are perceived as an acoustical defect, whereas reverberance can produce various subjective feelings of spaciousness and immersion in the sound field. The Haas Effect helps delineate the boundary between "echo" and "reverb".

"""
)

