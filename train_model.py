import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Sample data: age, weight, height
data = np.array([
    [25, 60, 165],
    [30, 80, 175],
    [22, 45, 150],
    [28, 70, 168],
    [35, 90, 180],
])
labels = np.array([
    22.0, 26.1, 20.0, 24.8, 27.8
])

X = torch.tensor(data, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(500):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# Save model as 'bmi_model.pt'
torch.save(model.state_dict(), "bmi_model.pt")
print("✅ AI model saved as 'bmi_model.pt'")
