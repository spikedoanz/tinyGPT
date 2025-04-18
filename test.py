from tinygrad.device import Device

# Set environment variables (optional)
import os
os.environ["HOST"] = "localhost:6667"  # If connecting to a remote server
# or leave unset to start a local server automatically

# Set the default device to CLOUD
os.environ["DEVICE"] = "CLOUD"

# Use tinygrad as normal - the CloudDevice will handle communication
from tinygrad import Tensor
x = Tensor([1, 2, 3])
y = x * 2
print(y.numpy())  # This will be processed on the cloud server
