import tensorflow as tf


# -------------------------Parameters-------------------------

tf.flags.DEFINE_float(
	"dev_sample_percentage",  # flag_name
	.1,  # default_value
	"Percentage of the training data to use for validation"  # docstr
)
tf.flags.DEFINE_integer(
	"embedding_dim",
	200,
	"Dimensionality of character embedding (default: 128)"
)
tf.flags.DEFINE_string(
	"filter_sizes",
	"3",
	"Comma-separated filter sizes (default: '3,4,5')"
)
tf.flags.DEFINE_integer(
	"num_filters",
	128,
	"Number of filters per filter size (default: 128)"
)
tf.flags.DEFINE_float(
	"dropout_keep_prob",
	0.3,
	"Dropout keep probability (default: 0.5)"
)
tf.flags.DEFINE_integer(
    "batch_size",
    64,
    "Batch Size (default: 64)"
)
tf.flags.DEFINE_integer(
    "num_epochs",
    25,
    "Number of training epochs (default: 200)"
)
tf.flags.DEFINE_integer(
	"evaluate_every",
	100,
	"Evaluate model on dev set after this many steps (default: 100)"
)
tf.flags.DEFINE_integer(
	"checkpoint_every",
	100,
	"Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer(
	"num_checkpoints",
	5,
	"Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_boolean(
	"allow_soft_placement",
	True,
	"Allow device soft device placement")
tf.flags.DEFINE_boolean(
	"log_device_placement",
	False,
	"Log placement of ops on devices"
)
tf.flags.DEFINE_integer(
	"vocabulary_size",
	200000,
	# 262144,  # i.e. 2e18
	"Vocbulary size"
)
tf.flags.DEFINE_integer(
	"sequence_length",
	100,
	"Sequence length"
)
tf.flags.DEFINE_string(
	"pad",
	"right",
	"Direction for padding in input sequence"
)
tf.flags.DEFINE_string(
	"out_dir",
	"askubuntu_runs",
	"Directory for saving summaries and checkpoints"
)

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
# for attr, value in FLAGS.__flags.items():
#     print("{}={}".format(attr.upper(), value))
