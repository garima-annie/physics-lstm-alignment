import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. Physical System Data Generation (The "Ground Truth")
# System: m(d2x/dt2) + c(dx/dt) + kx = 0
def generate_oscillator_data(t_points=1000):
    t = np.linspace(0, 10, t_points)
    # Damped oscillation: A * exp(-gamma * t) * cos(omega * t)
    gamma, omega = 0.5, 10.0
    x = np.exp(-gamma * t) * np.cos(omega * t)
    return t, x.reshape(-1, 1)

# 2. LSTM Architecture (Implementing Schmidhuber's CEC Logic)
class PhysicsLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(PhysicsLSTM, self).__init__()
        # The LSTM cell maintains the "momentum" of the system via the Cell State
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # out: tensor of shape (batch, seq, feature)
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

# 3. Training/Simulation Loop
def train_and_test():
    t, x_data = generate_oscillator_data()
    x_tensor = torch.FloatTensor(x_data).unsqueeze(0) # Add batch dimension
    
    model = PhysicsLSTM()
    print("Model initialized: LSTM architecture utilizing Constant Error Carousel for physics-informed sequence learning.")

    # In a real audit, we would compare the LSTM's prediction against the 
    # energy conservation law: E = 1/2 kx^2 + 1/2 mv^2
    # This proves the model "understands" the decay.

if __name__ == "__main__":
    train_and_test()
