import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("outputs/val_details.csv")

# Extract data
steps = df.index.values
loss = df['loss']
psnr = df['psnr']

# Calculate epochs
epochs = steps // 10

# Plotting
plt.figure(figsize=(16,9))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(steps, loss, color='blue')
plt.title('Val Loss')
plt.xlabel('Time Step')
plt.ylabel('Loss')
# for epoch in set(epochs):
#     plt.axvline(x=epoch*20, color='gray', linestyle='--')
#     plt.text(epoch*20 + 2, max(loss)*0.9, f'Epoch {epoch}', rotation=90)

# Plot PSNR
plt.subplot(1, 2, 2)
plt.plot(steps, psnr, color='red')
plt.title('Val PSNR')
plt.xlabel('Time Step')
plt.ylabel('PSNR')
# for epoch in set(epochs):
#     plt.axvline(x=epoch*20, color='gray', linestyle='--')
#     plt.text(epoch*20 + 2, max(psnr)*0.9, f'Epoch {epoch}', rotation=90)

plt.tight_layout()
plt.savefig('outputs/val_plot.png')
plt.show()
