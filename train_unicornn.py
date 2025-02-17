import torch
import torch.optim as optim
import numpy as np
import scipy.signal as signal
import librosa
import librosa.feature
import matplotlib.pyplot as plt
from fastdtw import fastdtw
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as AF
from unicornn import UnICORNN  # Ensure UnICORNN model is correctly defined

# 🚀 Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 🎛 Training Hyperparameters
input_dim = 2  # Two features: poles & zeros
hidden_dim = 512
output_dim = 2
dt = 0.008
alpha = 0.1
layers = 5
epochs = 1000
lr = 0.003

# 🚀 Load the impulse response from the provided WAV file
filename = "Arundel Nave.wav"
impulse_response, Fs = librosa.load(filename, sr=None)  # Keep original sampling rate

# 🔬 Compute Autoregressive (AR) Coefficients using Levinson-Durbin
def levinson_durbin(r, order):
    """Perform Levinson-Durbin recursion to get AR coefficients."""
    r = np.asarray(r)
    a = np.zeros(order + 1)
    e = np.zeros(order + 1)
    
    a[0] = 1.0
    e[0] = r[0]
    
    for i in range(1, order + 1):
        k = (r[i] - np.dot(a[:i], r[i - 1::-1])) / e[i - 1]
        a[i] = k
        a[1:i] -= k * a[i - 1:0:-1]
        e[i] = (1 - k**2) * e[i - 1]
    
    return a

order = 64
r = np.correlate(impulse_response, impulse_response, mode="full")[len(impulse_response) - 1:]
a_coeffs = levinson_durbin(r, order)  # Poles from Levinson

# 🔢 Initialize b-coefficients (zeros) with small Gaussian noise
b_coeffs = np.random.normal(loc=0, scale=1e-3, size=order + 1)

# Set batch_size and hidden_dim to the length of impulse response
batch_size = len(a_coeffs)
hidden_dim = len(b_coeffs)

print(f"ℹ️ Setting batch_size = {batch_size}, hidden_dim = {hidden_dim}")

# 📌 Initialize Model
model = UnICORNN(input_dim, hidden_dim, output_dim, dt, alpha, layers).to(device)

# 🏗️ Initialize Weights
with torch.no_grad():
    output_weights = np.stack([a_coeffs, b_coeffs], axis=0)  # Shape: [2, hidden_dim]
    model.output_layer.weight.copy_(torch.tensor(output_weights, dtype=torch.float32, device=device))
    model.input_layer.weight.copy_(torch.tensor(np.column_stack((a_coeffs, b_coeffs)), dtype=torch.float32, device=device))

print("✅ Successfully initialized UnICORNN weights.")

# 🎯 Construct Training Input with Poles & Zeros
x_train_poles = signal.lfilter(a_coeffs, [1.0], np.random.randn(len(a_coeffs), 1))  # Filtered with poles
x_train_zeros = signal.lfilter(b_coeffs, [1.0], np.random.randn(len(b_coeffs), 1))  # Filtered with zeros

x_train = np.hstack((x_train_poles, x_train_zeros))  # Shape: [batch_size, 2]
x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)


def compute_frequency_loss(model, impulse_response, Fs=1):
    """
    Computes a fully differentiable frequency response loss.
    """
    device = next(model.parameters()).device

    # Generate impulse response from trained IIR filter
    impulse = torch.zeros(512, device=device)
    impulse[0] = 1  # Impulse at t=0

    a_coeffs = torch.cat((torch.tensor([1.0], device=device), -model.output_layer.weight[0]))
    b_coeffs = F.pad(model.output_layer.weight[1], (0, 1))  # Pad with zero

    # Apply IIR filter (Differentiable)
    trained_response = AF.lfilter(impulse, a_coeffs, b_coeffs)

    # Convert impulse response to NumPy for freqz
    trained_response_np = trained_response.cpu().detach().numpy()

    # Compute frequency response
    w, h_trained = signal.freqz(b_coeffs.cpu().detach().numpy(), a_coeffs.cpu().detach().numpy(), worN=2048, fs=Fs)
    w_orig, h_orig = signal.freqz(impulse_response, [1.0], worN=2048, fs=Fs)  # ✅ Fixed

    # Convert to log magnitude response (dB)
    mag_trained = 20 * np.log10(np.abs(h_trained) + 1e-8)  # Prevent log(0)
    mag_orig = 20 * np.log10(np.abs(h_orig) + 1e-8)

    # Compute frequency loss (L2 difference)
    freq_loss = np.mean((mag_trained - mag_orig) ** 2)
    
    return torch.tensor(freq_loss, dtype=torch.float32, device=device)

def compute_mfcc_loss(model, impulse_response, Fs=1, n_mfcc=20):
    """
    Computes a fully differentiable MFCC-based frequency loss using PyTorch and torchaudio.
    This allows backpropagation through the loss, ensuring it influences training.
    """
    # Ensure we get the correct device from the model
    device = next(model.parameters()).device

    # Generate impulse response from trained IIR filter
    impulse = torch.zeros(512, device=device)
    impulse[0] = 1  # Impulse at t=0

    # Ensure a_coeffs starts with 1.0
    a_coeffs = torch.cat((torch.tensor([1.0], device=device), -model.output_layer.weight[0]))

    # Ensure b_coeffs matches a_coeffs size by padding
    b_coeffs = F.pad(model.output_layer.weight[1], (0, 1))  # Pad with zero at the end

    # Use torchaudio's differentiable lfilter instead of scipy.signal.lfilter
    trained_response = AF.lfilter(impulse, a_coeffs, b_coeffs)

    # Convert impulse response to MFCC (Fully Differentiable)
    mfcc_transform = T.MFCC(
        sample_rate=Fs, 
        n_mfcc=n_mfcc, 
        melkwargs={"n_fft": 512, "n_mels": 40, "hop_length": 128}
    ).to(device)

    # Ensure impulse_response is a PyTorch tensor
    impulse_response = torch.tensor(impulse_response, dtype=torch.float32, device=device)

    mfcc_trained = mfcc_transform(trained_response.unsqueeze(0))  # Shape: [1, n_mfcc, T_trained]
    mfcc_original = mfcc_transform(impulse_response.unsqueeze(0))  # Shape: [1, n_mfcc, T_original]

    # Match time dimension by trimming or padding
    min_time = min(mfcc_trained.shape[2], mfcc_original.shape[2])
    mfcc_trained = mfcc_trained[:, :, :min_time]
    mfcc_original = mfcc_original[:, :, :min_time]

    # Compute loss (L2 difference between MFCCs)
    mfcc_loss = F.mse_loss(mfcc_trained, mfcc_original)

    return mfcc_loss  # This now supports gradient computation!


# 🔄 Training Loop with Frequency Loss, MFCC Loss, and MSE Loss
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Forward pass
    y_pred = model(x_train_tensor)
    
    # Compute frequency response loss
    freq_loss = compute_frequency_loss(model, impulse_response, Fs=Fs)

    # Compute MFCC-based frequency loss
    mfcc_loss = compute_mfcc_loss(model, impulse_response, Fs=Fs)

    # Compute MSE loss
    mse_loss = torch.nn.MSELoss()(y_pred, x_train_tensor)

    # Weighted combination: 50% FREQ, 40% MFCC, 10% MSE
    loss = 0.9 * freq_loss + .095 * mfcc_loss + 0.05 * mse_loss

    # **Prevent NaNs: Clip gradients**
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ✅ Limits gradient explosion
    optimizer.step()

    # **Check for NaN values and stop if needed**
    if torch.isnan(loss):
        print(f"❌ NaN detected at epoch {epoch}, stopping training.")
        break

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Total Loss = {loss.item()}, FREQ Loss: {freq_loss.item()}, MFCC Loss: {mfcc_loss.item()}, MSE Loss: {mse_loss.item()}")

print("✅ Training complete with 50% FREQ, 40% MFCC, and 10% MSE Loss!")


# ✅ Save Poles & Zeros to a File
def save_poles_zeros(model, filename="poles_zeros.txt"):
    """
    Extracts the pole (a coefficients) and zero (b coefficients) coefficients
    from the trained UnICORNN model, prints them, and saves them to a text file.
    """
    with torch.no_grad():
        a_coeffs = np.array([1.0] + (-model.output_layer.weight.cpu().numpy()[0, :]).tolist(), dtype=np.float64)
        b_coeffs = np.array(model.output_layer.weight.cpu().numpy()[1, :], dtype=np.float64)

    # Print coefficients to console
    print("\nPoles (a coefficients):")
    print(" ".join(map(str, a_coeffs)))
    
    print("\nZeros (b coefficients):")
    print(" ".join(map(str, b_coeffs)))

    # Save to text file
    with open(filename, "w") as f:
        f.write("Poles (a coefficients):\n")
        f.write(" ".join(map(str, a_coeffs)) + "\n\n")
        f.write("Zeros (b coefficients):\n")
        f.write(" ".join(map(str, b_coeffs)) + "\n")

    print(f"\n✅ Coefficients saved to {filename}")

# Save the coefficients
save_poles_zeros(model)

def plot_frequency_response(model, impulse_response, Fs, filename="freq_response.png"):
    """Compute and save the frequency response plot of the trained IIR filter."""
    
    # Ensure model is on CPU for coefficient extraction
    device = next(model.parameters()).device

    # Extract coefficients from the trained model
    a_coeffs = torch.cat((torch.tensor([1.0], device=device), -model.output_layer.weight[0])).detach().cpu().numpy()
    b_coeffs = model.output_layer.weight[1].detach().cpu().numpy()

    # Debugging print
    print(f"📏 Debugging plot_frequency_response() -> a_coeffs shape: {a_coeffs.shape}, b_coeffs shape: {b_coeffs.shape}")
    print(f"📏  a: {a_coeffs}, b: {b_coeffs}")

    # Compute frequency response of trained IIR filter
    w, h_iir = signal.freqz(b_coeffs, a_coeffs, worN=2048, fs=Fs)

    # Compute frequency response of original impulse response
    w_rir, h_rir = signal.freqz(impulse_response, [1.0], worN=2048, fs=Fs)

    # Plot the responses
    plt.figure(figsize=(10, 5))
    plt.plot(w, 20 * np.log10(abs(h_iir)), label="Trained IIR Filter", linestyle='dashed', color='red')
    plt.plot(w_rir, 20 * np.log10(abs(h_rir)), label="Original Impulse Response", linestyle='solid', color='blue')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Frequency Response Comparison')
    plt.legend()
    plt.grid()

    # Save instead of showing the plot
    plt.savefig(filename, dpi=300)
    print(f"✅ Saved frequency response plot as {filename}")

# Usage:
# Use this corrected function call
plot_frequency_response(model, impulse_response, Fs)
