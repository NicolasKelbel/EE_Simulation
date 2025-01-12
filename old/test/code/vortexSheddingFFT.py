import numpy as np
import matplotlib.pyplot as plt

def read_coefficients(file_path):
    
    # Reads the OpenFOAM 'coefficient.dat' file and extracts time and lift coefficient (Cl).
    
    # Args:
    #     file_path (str): Path to the coefficient.dat file.
    
    # Returns:
    #     t (numpy array): Time data.
    #     Cl (numpy array): Lift coefficient data.
    
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Detect the data start line (ignoring headers and comments)
    data_lines = [line.strip() for line in lines if not line.startswith("#") and line.strip()]

    # Convert to numerical values
    data = np.loadtxt(data_lines)

    # Extract columns (assuming OpenFOAM format: [time, Cd, Cl, Cm])
    t = data[:, 0]   # Time column
    Cl = data[:, 2]  # Lift coefficient column (adjust if your file has a different order)

    return t, Cl

def compute_fft(t, Cl):
   
    # Computes the FFT of the lift coefficient to determine the vortex shedding frequency.
    
    # Args:
    #     t (numpy array): Time data.
    #     Cl (numpy array): Lift coefficient data.
    
    # Returns:
    #     dominant_freq (float): The vortex shedding frequency (Hz).
    #     freqs (numpy array): Frequency axis.
    #     spectrum (numpy array): Magnitude of FFT.
    
    dt = t[1] - t[0]  # Time step
    Cl_fft = np.fft.fft(Cl)  # Compute FFT
    freqs = np.fft.fftfreq(len(t), dt)  # Compute frequency axis

    # Only keep positive frequencies
    positive_freqs = freqs[freqs > 0]
    spectrum = np.abs(Cl_fft[freqs > 0])

    # Find the dominant frequency (corresponding to vortex shedding)
    dominant_freq = positive_freqs[np.argmax(spectrum)]
    
    return dominant_freq, positive_freqs, spectrum

def plot_spectrum(freqs, spectrum, dominant_freq):
 
    # Plots the frequency spectrum of the lift coefficient.
    
    # Args:
    #     freqs (numpy array): Frequency axis.
    #     spectrum (numpy array): FFT magnitude.
    #     dominant_freq (float): Detected vortex shedding frequency.

    plt.figure(figsize=(8, 4))
    plt.plot(freqs, spectrum, label="FFT of $C_L$")
    plt.axvline(dominant_freq, color="r", linestyle="--", label=f"Vortex Shedding: {dominant_freq:.3f} Hz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Frequency Spectrum of Lift Coefficient")
    plt.legend()
    plt.grid()
    plt.show()

# === Main Execution ===
file_path = r"C:\Users\Nicolas\AppData\Local\Temp\Unnamed\case\postProcessing\ReportingFunction\0\coefficient.dat"  

# Read data
t, Cl = read_coefficients(file_path)

# Compute FFT
dominant_freq, freqs, spectrum = compute_fft(t, Cl)

# Print the result
print(f"Vortex Shedding Frequency: {dominant_freq:.3f} Hz")

# Plot the spectrum
plot_spectrum(freqs, spectrum, dominant_freq)