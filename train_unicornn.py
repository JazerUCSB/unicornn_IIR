import torch
import torch.optim as optim
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import librosa
from unicornn import UnICORNN  # Ensure UnICORNN model is correctly defined

# 🚀 Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 🎛 Training Hyperparameters
input_dim = 2  # Two features: poles & zeros
hidden_dim = 128  # Will be adjusted dynamically
output_dim = 2  # Extracting poles and zeros
dt = 0.01
alpha = 0.1
layers = 3
epochs = 5000
lr = 0.001

# 🚀 Load the impulse response from the provided WAV file
filename = "Mesa_OS_4x12_57_m160.wav"
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

order = 84
r = np.correlate(impulse_response, impulse_response, mode="full")[len(impulse_response) - 1:]
a_coeffs = levinson_durbin(r, order)  # Poles from Levinson

# 🔢 Initialize b-coefficients (zeros) with small Gaussian noise
b_coeffs = np.random.normal(loc=0, scale=1e-3, size=order + 1)

# Set batch_size and hidden_dim to the length of impulse response
batch_size = len(a_coeffs)
hidden_dim = len(b_coeffs)  # Match the input length

print(f"ℹ️ Setting batch_size = {batch_size}, hidden_dim = {hidden_dim}")

# Ensure `a_coeffs` and `b_coeffs` match the new hidden dimension
#a_coeffs = np.pad(a_coeffs, (0, hidden_dim - len(a_coeffs)), 'constant', constant_values=0) if len(a_coeffs) < hidden_dim else a_coeffs[:hidden_dim]
#b_coeffs = np.pad(b_coeffs, (0, hidden_dim - len(b_coeffs)), 'constant', constant_values=0) if len(b_coeffs) < hidden_dim else b_coeffs[:hidden_dim]

# 📌 Initialize Model
model = UnICORNN(input_dim, hidden_dim, output_dim, dt, alpha, layers).to(device)

# 🔍 Debugging Model Layer Shapes
print(f"✅ Model Layers Shape:")
print(f"   output_layer.weight.shape: {model.output_layer.weight.shape}")  # Expected: [2, hidden_dim] (i.e., [2, 9056])
print(f"   input_layer.weight.shape: {model.input_layer.weight.shape}")    # Expected: [hidden_dim, 2] (i.e., [9056, 2])

# 🏗️ Correctly Reshape & Assign Weights
with torch.no_grad():
    # First row: Levinson-Durbin a-coefficients (poles)
    # Second row: Small Gaussian noise around zero (zeros)
    output_weights = np.stack([a_coeffs, b_coeffs], axis=0)  # Shape: [2, hidden_dim]
    
    model.output_layer.weight.copy_(torch.tensor(output_weights, dtype=torch.float32, device=device))  

    # Reshape `b_coeffs` to match input_layer's shape ([hidden_dim, input_dim])
    model.input_layer.weight.copy_(torch.tensor(np.column_stack((a_coeffs, b_coeffs)), dtype=torch.float32, device=device))  

print("✅ Successfully initialized UnICORNN weights.")

# 🎯 Construct Training Input with Poles & Zeros
x_train_poles = signal.lfilter(a_coeffs, [1.0], np.random.randn(len(a_coeffs), 1))  # Filtered with poles
x_train_zeros = signal.lfilter(b_coeffs, [1.0], np.random.randn(len(b_coeffs), 1))  # Filtered with zeros

x_train = np.hstack((x_train_poles, x_train_zeros))  # Shape: [1000, 2]
x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 🔄 Training Loop
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_train_tensor)
    loss = torch.nn.MSELoss()(y_pred, x_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")

# 📊 Frequency Response Comparison
def plot_frequency_response(model, impulse_response, Fs=1, filename="freq_response.png"):
    """Compute and save the frequency response plot of the trained IIR filter."""
    
    # Extract coefficients
    a_coeffs = np.array([1.0] + (-model.output_layer.weight.detach().cpu().numpy()[0, :]).tolist(), dtype=np.float64)
    b_coeffs = np.array(model.output_layer.weight.detach().cpu().numpy()[1, :], dtype=np.float64)

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

# Generate frequency response plot
plot_frequency_response(model, impulse_response, Fs=1, filename="freq_response.png")
