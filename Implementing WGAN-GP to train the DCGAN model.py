import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5), std=(0.5))
])

mnist_dataset = torchvision.datasets.MNIST(
    root='./', train=True, transform=transform, download=False
)

def make_generator_network_wgan(input_size, n_filters):
    model = nn.Sequential(
        nn.ConvTranspose2d(input_size, n_filters*4, 4, 1, 0, bias=False),
        nn.InstanceNorm2d(n_filters*4),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(n_filters*4, n_filters*2, 3, 2, 1, bias=False),
        nn.InstanceNorm2d(n_filters*2),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(n_filters*2, n_filters, 4, 2, 1, bias=False),
        nn.InstanceNorm2d(n_filters),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(n_filters, 1, 4, 2, 1, bias=False),
        nn.Tanh()
    )
    return model

class Discriminator(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, n_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters, n_filters*2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(n_filters*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters*2, n_filters*4, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(n_filters*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters*4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.network(input)
        return output.view(-1, 1).squeeze(0)

def create_noise(batch_size, z_size, mode_z):
    if mode_z == 'uniform':
        input_z = torch.rand(batch_size, z_size, 1, 1) * 2 - 1
    elif mode_z == 'normal':
        input_z = torch.randn(batch_size, z_size, 1, 1)
    return input_z

z_size = 100
n_filter = 32
lambda_gp = 10.0
num_epochs = 100
torch.manual_seed(1)
critic_iterations = 5
batch_size = 64
image_size = (28, 28)

mnist_dl = DataLoader(
    dataset=mnist_dataset,
    batch_size=batch_size,
    shuffle=True,
)

gen_model = make_generator_network_wgan(z_size, n_filter).to(device)
disc_model = Discriminator(n_filter).to(device)
g_optimizer = torch.optim.Adam(gen_model.parameters(), lr=0.0002)
d_optimizer = torch.optim.Adam(disc_model.parameters(), lr=0.0002)

def gradient_penalty(real_data, generated_data):
    batch_size = real_data.size(0)

    alpha = torch.rand(real_data.shape[0], 1, 1, 1, requires_grad=True ,device=device)
    interpolated = alpha * real_data + (1 - alpha) * generated_data

    proba_interpolated = disc_model(interpolated)

    gradients = torch.autograd.grad(
        outputs=proba_interpolated, inputs=interpolated,
        grad_outputs=torch.ones(proba_interpolated.size(), device=device),
        create_graph=True, retain_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = gradients.norm(2, dim=1)
    return lambda_gp * ((gradients_norm - 1) ** 2).mean()

def d_train_wgan(x):
    disc_model.zero_grad()

    batch_size = x.size(0)
    x = x.to(device)

    d_real = disc_model(x)
    input_z = create_noise(batch_size, z_size, 'uniform').to(device)
    g_output = gen_model(input_z)
    d_generated = disc_model(g_output)
    d_loss = d_generated.mean() - d_real.mean() + gradient_penalty(x.data, g_output.data)
    d_loss.backward()
    d_optimizer.step()
    return d_loss.data.item()

def g_train_wgan(x):
    gen_model.zero_grad()

    batch_size = x.size(0)
    input_z = create_noise(batch_size, z_size, 'uniform').to(device)
    g_output = gen_model(input_z)
    d_generated = disc_model(g_output)
    g_loss = -d_generated.mean()

    g_loss.backward()
    g_optimizer.step()
    return g_loss.data.item()

fixed_z = create_noise(batch_size, z_size, 'uniform').to(device)
def create_samples(gen_model, input_z):
    g_output = gen_model(input_z)
    images = torch.reshape(g_output, (batch_size, *image_size))
    return (images+1)/2.0

epoch_samples_wgan = []
for epoch in range(1, num_epochs+1):
    gen_model.train()
    d_losses, g_losses = [], []
    for i, (x, _) in enumerate(tqdm(mnist_dl)):
        for _ in range(critic_iterations):
            d_loss = d_train_wgan(x)
        d_losses.append(d_loss)
        g_losses.append(g_train_wgan(x))

    print(f'Epoch {epoch:03d} | D Loss >> {torch.FloatTensor(d_losses).mean():.4f}')
    gen_model.eval()
    epoch_samples_wgan.append(
        create_samples(gen_model, fixed_z).detach().cpu().numpy()
    )