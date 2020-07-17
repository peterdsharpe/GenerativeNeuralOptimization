import jax
import jax.numpy as jnp
import numpy as onp
import flax

latent_space_dimensionality = 100
hidden_layer_dimensionality = 50
design_space_dimensionality = 2
training_iterations = 300
batch_size = 1000


class Network(flax.nn.Module):
    def apply(self, x):

        x = flax.nn.Dense(x, features=hidden_layer_dimensionality)
        x = flax.nn.relu(x)
        x = flax.nn.Dense(x, features=hidden_layer_dimensionality)
        x = flax.nn.relu(x)
        x = flax.nn.Dense(x, features=hidden_layer_dimensionality)
        x = flax.nn.relu(x)
        x = flax.nn.Dense(x, features=hidden_layer_dimensionality)
        x = flax.nn.relu(x)
        x = flax.nn.Dense(x, features=design_space_dimensionality)
        return x


_, initial_params = Network.init_by_shape(
    jax.random.PRNGKey(0),
    [
        (
            (latent_space_dimensionality,),
            jnp.float32
        )
    ]
)
model = flax.nn.Model(Network, initial_params)

optimizer = flax.optim.Adam(learning_rate=0.01).create(model)


@jax.jit
@jax.vmap
def objective_function(x_design):
    # return jnp.sum(jnp.abs(x_design - 4))
    # return jnp.sum(jnp.abs(x_design[0] - 4))
    bin_1 = jnp.sum(jnp.abs(x_design - jnp.array([4, 0])))
    bin_2 = jnp.sum(jnp.abs(x_design - jnp.array([0, 4])))
    return jnp.minimum(bin_1, bin_2)


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

sns.set(palette=sns.color_palette("husl"))
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
iters = []
samples = []



### Training loop
for iter in range(training_iterations):
    x_latent_space = onp.random.randn(batch_size, latent_space_dimensionality)


    def loss_function(model):
        x_design_space = model(x_latent_space)
        loss = jnp.mean(objective_function(x_design_space))
        return loss


    loss, loss_gradient = jax.value_and_grad(loss_function)(optimizer.target)
    optimizer = optimizer.apply_gradient(loss_gradient)
    if iter % 20 == 0:
        print(f"Iter {iter:4} | Loss = {loss}")

        x_latent_space = onp.random.randn(1000, latent_space_dimensionality)
        x_design_space = optimizer.target(x_latent_space)

        iters.append(iter)
        samples.append(x_design_space)

        ax = sns.kdeplot(
            onp.array(x_design_space[:, 0], dtype="f8"),
            onp.array(x_design_space[:, 1], dtype="f8"),
            shade=True,
            shade_lowest=False,
            cmap="Reds",
        )
        X1_obj, X2_obj = onp.meshgrid(onp.linspace(-3, 7), onp.linspace(-3, 7), indexing="ij")

        plt.title(f"Generative Neural Optimization\nIteration {iter}")
        plt.xlim(-3, 7)
        plt.ylim(-3, 7)
        plt.show()

### Make animation
# def animate(index):
#     plt.clf()
#     ax = sns.kdeplot(
#         onp.array(samples[index][:, 0], dtype="f8"),
#         onp.array(samples[index][:, 1], dtype="f8"),
#         shade=True,
#         shade_lowest=False,
#         cmap="Reds",
#     )
#     plt.title(f"Generative Neural Optimization\nIteration {iters[index]}")
#     plt.xlim(-3, 7)
#     plt.ylim(-3, 7)
#
# ani = FuncAnimation(fig, animate, range(len(iters)), cache_frame_data=False)
# ani.save("animation.mp4", fps=20)