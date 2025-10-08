import torch
import torch.nn as nn
from utils.fixseed import fixseed
from utils.parser_util import train_args, classifier_args
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
import torch.nn.functional as F
import os


import json
from train.training_loop import TrainLoop
from utils.model_util import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
os.system('export clearml_log_level=ERROR')



class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.fc = nn.Linear(int(hidden_size/2), num_classes)

    def forward(self, x):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Reshape the output tensor for linear layer
        out = out[:, -1, :]

        # Forward propagate through the linear layer
        out = self.fc1(out)
        out = self.fc(out)
        return out


args = classifier_args()#train_args()
fixseed(args.seed)
dist_util.setup_dist(args.device)
print("creating data loader...")
data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames,
                          data_mode=args.data_mode)

train_platform_type = eval(args.train_platform_type)
train_platform = train_platform_type(args.save_dir)
train_platform.report_args(args, name='Args')

save_dir = args.save_dir
overwrite = args.overwrite

# Set the hyperparameters
input_size = 68*3  # Replace with your input size
hidden_size = 128  # Replace with the desired hidden size
num_layers = 1  # Replace with the desired number of LSTM layers
num_classes = 28#5  # Replace with the number of action classes

# Create an instance of the LSTMClassifier
model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)

model.to(dist_util.dev())

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss().to(dist_util.dev())
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Generate some dummy input data (replace with your own data loading pipeline)
batch_size = 32
sequence_length = 150


input_data = torch.randn(batch_size, sequence_length, input_size)
targets = torch.randint(0, num_classes, (batch_size,))
from tqdm import tqdm

# Training loop
num_epochs = 150
for epoch in range(num_epochs):
    # Forward pass
    correct = 0
    total_data = 0
    for motion, cond in tqdm(data):
        motion = motion.to(dist_util.dev()).squeeze(2).permute(0, 2, 1)
        outputs = model(motion)
        targets = cond['y']['action'].squeeze(1).to(dist_util.dev())
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        index_pred = torch.argmax(F.softmax(outputs), dim=1)
        correct += torch.sum(index_pred == targets).item()
        total_data += motion.shape[0]

        # Print the loss for monitoring progress
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Correct: {100*(correct/total_data)}%")

# Save the trained model
torch.save(model.state_dict(), "action_recognition_model.pth")
train_platform.close()
