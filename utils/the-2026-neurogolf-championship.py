# %%
# !pip install -q numpy==2.4.4 2>/dev/null
# !pip install -q onnx==1.21.0 2>/dev/null
# !pip install -q onnxruntime==1.24.4 2>/dev/null
# !pip install -q onnx-tool==1.0.1 2>/dev/null

# %% [markdown]
# This starter notebook for the [2026 NeuroGolf Championship](https://www.kaggle.com/competitions/neurogolf-2026) is designed to help contestants verify the functional correctness of their networks.  It allows one to load example pairs for a task, visualize them, and test whether a candidate network produces expected results across all public competition benchmarks.
#
# ## Enter a task number (between 1 and 400):

# %%
task_num = 0  # Task 0 is just an illustrative example (and not eligible for points)

# %% [markdown]
# ## Color legend

# %%
import sys
sys.path.append("/kaggle/input/competitions/neurogolf-2026/neurogolf_utils")
try:
  # Local repository/notebook execution from the project root.
  from utils.neurogolf_utils import *
except ModuleNotFoundError as error:
  if error.name != "utils":
    raise
  # Kaggle, or direct execution as `python utils/the-2026-...py`.
  from neurogolf_utils import *
show_legend()

# %% [markdown]
# ## Example <input, output> pairs

# %%
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])


# %% [markdown]
# ## Define your network to solve the task
#
# The code below uses a utility we've provided to help you create [single-layer convolutional networks](https://en.wikipedia.org/wiki/Convolutional_layer).  It may be possible to solve other tasks in a similar way, but most will require more complicated architectures!

# %%
def weight(channel_out, channel_in, kernel_coord):
  if kernel_coord == ( 0,  0) and channel_in == channel_out: return 1.0
  if kernel_coord == ( 0,  0) and channel_in != 5 and channel_out == 0: return -1.0
  if kernel_coord == (-1, -1) and channel_in != 5 and channel_out == 0: return 1.0
  if kernel_coord == (-1, -1) and channel_in != 5 and channel_out == 5: return -1.0
  return 0.0

network = single_layer_conv2d_network(weight, kernel_size=3)

# %% [markdown]
# ## Verify your network

# %%
verify_network(network, task_num, examples)
