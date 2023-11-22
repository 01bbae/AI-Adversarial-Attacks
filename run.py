import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import timeit
from tqdm import tqdm

start = timeit.default_timer()

fgsm_epsilons = [0, .05, .1, .15, .2, .25, .3]
# epsilons = [0]
pgd_epsilons = [0, 15/255, 25/255, 35/255]
# Parameters given in Assignment
epsilon_param = 25/255
alpha_param = 10/255
max_iter = 15
pretrained_model = "data/lenet_mnist_model.pth"
# Set random seed for reproducibility
torch.manual_seed(42)

# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])),
        batch_size=1, shuffle=True)
use_cuda = False
use_mps = False
# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
print("MPS Available: ",torch.backends.mps.is_available())
if torch.backends.mps.is_available() and use_mps:
    device = torch.device("mps")
    print("Using MPS as device")
elif torch.cuda.is_available() and use_cuda:
    device = torch.device("cuda")
    print("Using CUDA as device")
else:
    device = torch.device("cpu")
    print ("Default CPU will be used as device")

# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location=device))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

# FGSM attack code
def fgsm_attack(model, loss_fn, image, label, epsilon):
    # Forward pass the data through the model
    output = model(image)
    init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

    # If the initial prediction is wrong, don't bother attacking, just move on
    if init_pred != label.item():
        return init_pred, image
        
    # Calculate the loss
    loss = loss_fn(output, label)

    # Zero all existing gradients
    model.zero_grad()

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect ``datagrad``
    data_grad = image.grad.data

    # Restore the data to its original scale
    original_image = denorm(image)

    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = original_image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return init_pred, perturbed_image

# PGD attack code
def pgd_attack(model, loss_fn, image, label, epsilon, max_iter, alpha):
    # epsilon is based on inf norm

    # Haven't applied perturbations yet but used as a copy for the loop
    perturbed_image = image
    # Create a random adversarial target that is different from current target label
    adv_target = label
    while adv_target == label:
        adv_target = torch.floor(torch.rand(1)*(10)).type(torch.LongTensor)
    
    # Repeat PGD until end condition or max iter
    for i in range(max_iter):
        output = model(perturbed_image)

        if i == 0:
            # Store initial prediction of non perturbed image
            init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            # If the initial prediction is wrong, don't bother attacking, just move on
            if init_pred != label.item():
                return init_pred, perturbed_image

        model.zero_grad()
        loss = loss_fn(output, adv_target).to(device)
        loss.backward(retain_graph=True)
        perturbed_image = denorm(perturbed_image)

        # Collect the element-wise sign of the data gradient
        sign_grad = perturbed_image.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = perturbed_image + alpha*sign_grad
        # Keep perturbations under epsilon value
        inf_norm = abs(torch.linalg.vector_norm(torch.reshape(perturbed_image - image, (28,28)), ord=np.inf))
        if inf_norm > epsilon:
            perturbations = torch.clip(perturbed_image - image, min=-epsilon, max=+epsilon)
        else:
            break
        # Adding perturbations to image and clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image + perturbations, 0, 1)
        # Renormalize perturbed image for new gradient
        perturbed_image = transforms.Normalize((0.1307,), (0.3081,))(perturbed_image)


    # Return the perturbed image
    return init_pred, perturbed_image

# restores the tensors to their original scale
def denorm(batch, mean=[0.1307], std=[0.3081]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def fgsm_test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []
    index = 0
        
    # Loop over all examples in test set
    for data, target in tqdm(test_loader):

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        loss = nn.NLLLoss()

        # Call FGSM Attack
        init_pred, perturbed_data = fgsm_attack(model, loss, data, target, epsilon)
        index += 1

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

def pgd_test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []
    index = 0
        
    # Loop over all examples in test set
    for data, target in tqdm(test_loader):

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        loss = nn.NLLLoss()

        # Call PGD Attack
        # print("New image:", index)
        init_pred, perturbed_data = pgd_attack(model, loss, data, target, epsilon, max_iter, alpha_param)
        index += 1

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

fgsm_accuracies = []
fgsm_examples = []

pgd_accuracies = []
pgd_examples = []


# Run test for each epsilon for fgsm
print("FGSM")
for eps in epsilons:
    acc, ex = fgsm_test(model, device, test_loader, eps)

    fgsm_accuracies.append(acc)
    fgsm_examples.append(ex)

# print("PGD")
# for eps in epsilons:
#     acc, ex = pgd_test(model, device, test_loader, eps)

#     pgd_accuracies.append(acc)
#     pgd_examples.append(ex)
end = timeit.default_timer()

print("elapsed time: ", end - start)

plt.figure(figsize=(5,5))
plt.plot(epsilons, fgsm_accuracies, "*-")
# plt.plot(epsilons, pgd_accuracies, "-")
# plt.legend()
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon for FGSM and PGD")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")

plt.show()